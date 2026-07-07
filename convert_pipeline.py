#!/usr/bin/env python3
"""AI-READI full-v3 ETL — optimized for a 32-core / 256 GB RAM / 8 TB disk box.

Architecture (the old notebook did 24k per-file `azcopy cp` spawns — hours of pure overhead):
  1. BULK DOWNLOAD  — 5 recursive `azcopy cp` calls (one per datatype), not 24k per-file spawns.
                      azcopy parallelizes internally; `--overwrite=ifSourceNewer` => resumable.
  2. BULK PROCESS   — ProcessPool(N_WORKERS=32) over the manifest, reading LOCAL raw files
                      (no download latency during compute). garmin/ecg GIL-bound flattens finally
                      use all 32 cores. Writes per-participant shards (atomic, resumable).
  3. MERGE          — parallel-read (ThreadPool) / single streaming ParquetWriter at ZSTD_LEVEL=1
                      (disk is abundant -> favour fast compression over size).
  4. ARCHIVE        — rclone canonical -> gdrive.

All paths are RELATIVE to BASE (default ./ai_readi) — run this script from a directory on your
8 TB disk. Existing shards on disk are reused (resume); raw is re-bulk-downloaded (the old run
deleted it per-participant). Override knobs via env vars (see Config).

Usage:
    python convert.py                 # full pipeline
    AI_READI_SKIP_DOWNLOAD=1 python convert.py   # skip phase 1 (raw already on disk)
    AI_READI_SKIP_ARCHIVE=1 python convert.py    # skip the gdrive rclone step
    AI_READI_BASE=/mnt/data/ai_readi python convert.py
"""

import subprocess, json, os, sys, time, shutil, re, threading, warnings
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ============================================================================
# Config — all relative; no hardcoded absolute paths.
# ============================================================================
DATASET   = os.environ.get("AI_READI_DATASET", "full")              # "mini" | "full"

# Load secrets from .env if present (gitignored) so SAS URLs never live in source.
# Either create .env (see .env.example) or export AI_READI_SAS_* in your shell.
def _load_dotenv():
    p = Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_dotenv()

SAS = {
 "mini": os.environ.get("AI_READI_SAS_MINI") or "",
 "full": os.environ.get("AI_READI_SAS_FULL") or "",
}
BASE      = Path(os.environ.get("AI_READI_BASE", "ai_readi"))       # relative; lives on the 8 TB disk
N_WORKERS       = min(32, os.cpu_count() or 32)                     # threads (env/dex) + procs (garmin/ecg)
N_PROC_WORKERS  = N_WORKERS                                          # ProcessPool for garmin/ecg (RAM is not a constraint)
N_MERGE_WORKERS = N_WORKERS                                          # parallel shard reads in merge
N_RCLONE        = N_WORKERS                                          # rclone upload concurrency
ZSTD_LEVEL      = 1                                                  # merge writer: 8 TB disk -> favour speed over size
USE_ARROW_ENV   = True
GDRIVE_REMOTE   = os.environ.get("AI_READI_GDRIVE", f"gdrive:AI_READI/{DATASET}/AI_READI")
SKIP_DOWNLOAD   = os.environ.get("AI_READI_SKIP_DOWNLOAD", "")  not in ("", "0", "false")
SKIP_ARCHIVE    = os.environ.get("AI_READI_SKIP_ARCHIVE", "")   not in ("", "0", "false")

RAW        = BASE / "raw"        / DATASET
CANONICAL  = BASE / "ai_readi"   / DATASET / "AI_READI"
MANIFEST   = BASE / "manifests"  / DATASET
SHARD_ROOT = BASE / "shards"     / DATASET
for p in (RAW, CANONICAL, MANIFEST, SHARD_ROOT): p.mkdir(parents=True, exist_ok=True)

_base, _q, _query = SAS[DATASET].partition("?")
def blob_url(path):  return f"{_base}/{path}{_q}{_query}"            # literal slashes; AI-READI paths have no chars needing quote
def dir_url(prefix): return f"{_base}/{prefix}{_q}{_query}"          # dir prefix

ROOT_GUID = None   # set in main(); passed explicitly to workers as a pickled arg (no fork-inheritance dependency)
def dl_one(file_path, dest, root_guid):
    """Per-file azcopy fallback. root_guid is passed explicitly so it's correct under fork OR spawn.
    Returns (ok, err) so workers can surface the azcopy stderr on failure."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["azcopy","cp", blob_url(f"{root_guid}/{file_path}"), str(dest)],
                       capture_output=True, text=True)
    ok = r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    return ok, ("" if ok else (r.stderr + r.stdout)[-300:])

def diag_download(fm, root):
    """One per-file download in the PARENT, so we know dl_one works before fanning out 32 workers.
    Prints the azcopy stderr on failure — that's the real error, not a generic 'missing'."""
    fp = fm.iloc[0]["file_path"]
    dest = RAW / "_diag" / Path(fp).name
    ok, err = dl_one(fp, dest, root)
    print(f"[diag] dl_one('{fp}') -> ok={ok}" + (f" | azcopy: {err[:200]}" if not ok else ""))
    if ok:
        try: dest.unlink()
        except Exception: pass

DT_KEEP   = {"clinical_data","environment","wearable_blood_glucose","wearable_activity_monitor","cardiac_ecg"}
LABEL_MAP = {"healthy":0,"pre_diabetes_lifestyle_controlled":1,
             "oral_medication_and_or_non_insulin_injectable_medication_controlled":2,"insulin_dependent":3}

# ============================================================================
# Deps + azcopy bootstrap
# ============================================================================
def ensure_deps():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pyarrow","pandas","orjson","wfdb","tqdm","numpy"], check=True)
    global np, pd, pa, pq, pacsv, wfdb, orjson
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.csv as pacsv, wfdb, orjson
    AZCOPY = Path.home()/".local"/"bin"/"azcopy"
    os.environ["PATH"] = str(Path.home()/".local"/"bin") + ":" + os.environ["PATH"]
    if not AZCOPY.exists():
        AZCOPY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["wget","-q","-O","/tmp/azc.tgz","https://aka.ms/downloadazcopy-v10-linux"], check=True)
        subprocess.run(["tar","-xzf","/tmp/azc.tgz","-C","/tmp/"], check=True)
        subprocess.run(["cp", str(next(Path("/tmp").glob("azcopy_linux_amd64_*/azcopy"))), str(AZCOPY)], check=True)
        AZCOPY.chmod(0o755)
    v = subprocess.run(["azcopy","--version"], capture_output=True, text=True).stdout.strip()
    print(f"azcopy: {v} | pyarrow {pa.__version__} | pandas {pd.__version__} | orjson {orjson.__version__} | wfdb {wfdb.__version__}")

# ============================================================================
# Core helpers (schema, shards, merge) — proven logic from the notebook
# ============================================================================
failures = []
_fail_lock = threading.Lock()
def log_failure(label, msg):
    with _fail_lock: failures.append((label, msg))

def enforce(df, spec):
    cols = [c for c,_ in spec]
    for c in cols:
        if c not in df.columns: df[c] = pd.NA
    df = df[cols].copy()
    for c, dt in spec:
        if   dt == "datetime_utc": df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        elif dt == "str":          df[c] = df[c].astype("string")
        elif dt == "float":       df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        elif dt == "int":          df[c] = pd.array(pd.to_numeric(df[c], errors="coerce"), dtype="Int32")
    return df

def shard_path(modality, pid):
    p = SHARD_ROOT / modality / f"{pid}.parquet"; p.parent.mkdir(parents=True, exist_ok=True); return p

def shard_valid(sp):
    try: return pq.ParquetFile(sp).metadata is not None
    except Exception: return False

def merge_shards(modality, spec, out_path, label=""):
    shards = sorted((SHARD_ROOT/modality).glob("*.parquet"))
    if not shards:
        print(f"[{label}] NO SHARDS to merge"); return
    schema = pa.Table.from_pandas(enforce(pd.DataFrame([{c:None for c,_ in spec}]), spec),
                                  preserve_index=False).schema
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    def _read_cast(sp):
        try: return pq.read_table(sp).cast(schema)             # zstd decompress releases the GIL
        except Exception as e: log_failure(modality, f"merge {sp.name}: {type(e).__name__}: {e}"); return None
    writer = None; n = 0; pid = 0; t0 = time.time(); chunk = N_MERGE_WORKERS
    try:
        with ThreadPoolExecutor(max_workers=N_MERGE_WORKERS) as ex:
            for i in range(0, len(shards), chunk):
                for f in [ex.submit(_read_cast, sp) for sp in shards[i:i+chunk]]:
                    tbl = f.result()
                    if tbl is None: continue
                    if writer is None: writer = pq.ParquetWriter(str(out_path), schema, compression="zstd", compression_level=ZSTD_LEVEL)
                    writer.write_table(tbl); n += tbl.num_rows; pid += 1
                if pid % 20 == 0 and pid:
                    print(f"  [{label}] merged {pid}/{len(shards)} shards, {n} rows ({time.time()-t0:.0f}s)")
    finally:
        if writer: writer.close()
    print(f"[{label}] MERGED {pid}/{len(shards)} shards -> {n} rows ({out_path.name})")

# ---- column specs ----
ENV_SPEC = [("person_id","int"),("ts","datetime_utc"),
            ("lch0","float"),("lch1","float"),("lch2","float"),("lch3","float"),
            ("lch6","float"),("lch7","float"),("lch8","float"),("lch9","float"),
            ("lch10","float"),("lch11","float"),
            ("pm1","float"),("pm2.5","float"),("pm4","float"),("pm10","float"),
            ("hum","float"),("temp","float"),("voc","float"),("nox","float"),
            ("screen","float"),("ff","float"),("inttemp","float")]
DEX_SPEC = [("person_id","int"),("timestamp","datetime_utc"),("blood_glucose","float"),
            ("unit","str"),("event_type","str"),("source_device_id","str"),("transmitter_id","str")]
GARMIN_SPEC = {
 "heart_rate":               [("person_id","int"),("timestamp","datetime_utc"),("heart_rate","float"),("unit","str")],
 "oxygen_saturation":        [("person_id","int"),("timestamp","datetime_utc"),("oxygen_saturation","float"),("unit","str"),("measurement_method","str")],
 "respiratory_rate":         [("person_id","int"),("timestamp","datetime_utc"),("respiratory_rate","float"),("unit","str")],
 "stress":                   [("person_id","int"),("timestamp","datetime_utc"),("stress_level","float"),("unit","str")],
 "sleep":                    [("person_id","int"),("start_time","datetime_utc"),("end_time","datetime_utc"),("sleep_stage_state","str")],
 "physical_activity":        [("person_id","int"),("start_time","datetime_utc"),("end_time","datetime_utc"),("activity_name","str"),("steps","float"),("step_unit","str")],
 "physical_activity_calorie":[("person_id","int"),("timestamp","datetime_utc"),("calories","float"),("unit","str")],
}
SPECS = {"environment": ENV_SPEC, "dexcom": DEX_SPEC, **GARMIN_SPEC}

# ============================================================================
# Parsers (module-level -> picklable for ProcessPool fork)
# ============================================================================
def _json_load(path):
    data = open(path, "rb").read()
    try: return orjson.loads(data)
    except Exception: return json.loads(data)   # stdlib tolerates NaN/Infinity (source has invalid-JSON NaN)
def _num(v):
    try: return float(v)
    except: return None

def parse_env(path, pid):
    try:
        tbl = pacsv.read_csv(str(path), read_options=pacsv.ReadOptions(skip_rows=45))
        df = tbl.to_pandas()
        if "ts" not in df.columns: raise ValueError(f"no 'ts' col, got {list(df.columns)[:5]}")
    except Exception:
        df = pd.read_csv(path, skiprows=45)
    df.insert(0, "person_id", pid); return df

def parse_dex(path, pid):
    j = _json_load(path); items = j.get("body",{}).get("cgm",[]); recs = []
    for it in items:
        ti = it.get("effective_time_frame",{}).get("time_interval",{}); bg = it.get("blood_glucose",{}) or {}
        recs.append({"person_id":pid,"timestamp":ti.get("start_date_time") or ti.get("end_date_time"),
                     "blood_glucose":bg.get("value"),"unit":bg.get("unit","mg/dL"),"event_type":it.get("event_type"),
                     "source_device_id":it.get("source_device_id"),"transmitter_id":it.get("transmitter_id")})
    return pd.DataFrame(recs)

def _flat_heart_rate(pid, body):
    return [{"person_id":pid,"timestamp":it.get("effective_time_frame",{}).get("date_time"),
             "heart_rate":_num((it.get("heart_rate",{}) or {}).get("value")),
             "unit":(it.get("heart_rate",{}) or {}).get("unit","beats/min")} for it in body.get("heart_rate",[])]
def _flat_oxygen_saturation(pid, body):
    return [{"person_id":pid,"timestamp":it.get("effective_time_frame",{}).get("date_time"),
             "oxygen_saturation":_num((it.get("oxygen_saturation",{}) or {}).get("value")),
             "unit":(it.get("oxygen_saturation",{}) or {}).get("unit","%"),
             "measurement_method":it.get("measurement_method")} for it in body.get("breathing",[])]
def _flat_respiratory_rate(pid, body):
    return [{"person_id":pid,"timestamp":it.get("effective_time_frame",{}).get("date_time"),
             "respiratory_rate":_num((it.get("respiratory_rate",{}) or {}).get("value")),
             "unit":(it.get("respiratory_rate",{}) or {}).get("unit","breaths/min")} for it in body.get("breathing",[])]
def _flat_stress(pid, body):
    return [{"person_id":pid,"timestamp":it.get("effective_time_frame",{}).get("date_time"),
             "stress_level":_num((it.get("stress",{}) or {}).get("value")),
             "unit":(it.get("stress",{}) or {}).get("unit","stress level")} for it in body.get("stress",[])]
def _flat_sleep(pid, body):
    return [{"person_id":pid,
             "start_time":it.get("effective_time_frame",{}).get("time_interval",{}).get("start_date_time"),
             "end_time":it.get("effective_time_frame",{}).get("time_interval",{}).get("end_date_time"),
             "sleep_stage_state":it.get("sleep_stage_state")} for it in body.get("sleep",[])]
def _flat_physical_activity(pid, body):
    return [{"person_id":pid,
             "start_time":it.get("effective_time_frame",{}).get("time_interval",{}).get("start_date_time"),
             "end_time":it.get("effective_time_frame",{}).get("time_interval",{}).get("end_date_time"),
             "activity_name":it.get("activity_name"),
             "steps":_num((it.get("base_movement_quantity",{}) or {}).get("value")),
             "step_unit":(it.get("base_movement_quantity",{}) or {}).get("unit","steps")} for it in body.get("activity",[])]
def _flat_physical_activity_calorie(pid, body):
    return [{"person_id":pid,"timestamp":it.get("effective_time_frame",{}).get("date_time"),
             "calories":_num((it.get("calories_value",{}) or {}).get("value")),
             "unit":(it.get("calories_value",{}) or {}).get("unit","kcal")} for it in body.get("activity",[])]
GARMIN_FLATTEN = {"heart_rate":_flat_heart_rate,"oxygen_saturation":_flat_oxygen_saturation,
                  "respiratory_rate":_flat_respiratory_rate,"stress":_flat_stress,"sleep":_flat_sleep,
                  "physical_activity":_flat_physical_activity,"physical_activity_calorie":_flat_physical_activity_calorie}
GARMIN_SUFFIX = {"heart_rate":"_heartrate.json","oxygen_saturation":"_oxygensaturation.json",
                 "respiratory_rate":"_respiratoryrate.json","stress":"_stress.json","sleep":"_sleep.json",
                 "physical_activity":"_activity.json","physical_activity_calorie":"_calorie.json"}

def parse_garmin(path, pid, modality):
    return pd.DataFrame(GARMIN_FLATTEN[modality](pid, _json_load(path).get("body",{})))

# ============================================================================
# Workers (ProcessPool). row is a dict with _modality/file_path/_pid/_ext.
# Reads LOCAL raw (bulk-downloaded in phase 1) -> parse -> enforce -> atomic shard.
# ============================================================================
def _shard_worker(row, root_guid):
    modality = row["_modality"]
    try: pid = int(row["_pid"])
    except Exception as e: return ("pid-err", modality, str(row.get("_pid")), f"{e}")
    sp = shard_path(modality, pid)
    if sp.exists() and shard_valid(sp): return ("resume", modality, pid)
    src = RAW / row["file_path"]
    if not src.exists():
        ok, err = dl_one(row["file_path"], src, root_guid)
        if not ok: return ("missing", modality, pid, f"{row['file_path']} | azcopy: {err}")
    try:
        if   modality == "environment": df = parse_env(src, pid)
        elif modality == "dexcom":      df = parse_dex(src, pid)
        else:                           df = parse_garmin(src, pid, modality)
        if df is None or len(df) == 0: return ("empty", modality, pid)
        df = enforce(df, SPECS[modality])
        tmp = sp.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, compression="zstd")
        os.replace(str(tmp), str(sp))               # atomic; overwrites any corrupt shard
        return ("wrote", modality, pid)
    except Exception as e:
        return ("parse-err", modality, pid, f"{type(e).__name__}: {e}")

def _ecg_worker(row, root_guid):
    try: pid = int(row["_pid"])
    except Exception as e: return None, None, f"pid parse {row.get('_pid')!r}: {e}"
    rec_id = Path(row["file_path"]).stem
    hea = RAW / row["file_path"]; dat = RAW / (row["file_path"][:-4] + ".dat")
    if not hea.exists(): dl_one(row["file_path"], hea, root_guid)
    if not dat.exists(): dl_one(row["file_path"][:-4] + ".dat", dat, root_guid)
    if not hea.exists() or not dat.exists(): return None, None, f"pid={pid} missing raw {row['file_path']}"
    try:
        rec = wfdb.rdrecord(str(hea)[:-4])
        sig = np.asarray(rec.p_signal, dtype=np.float32).T
        idx = {"rec_id":rec_id,"person_id":pid,"fs":int(rec.fs),"sig_len":int(rec.sig_len),
               "n_sig":int(rec.n_sig),"sig_name":",".join(rec.sig_name),"units":",".join(rec.units),
               "comments":"\n".join(rec.comments)}
        return sig, idx, None
    except Exception as e:
        return None, None, f"pid={pid} {type(e).__name__}: {e}"

# ============================================================================
# Pipeline
# ============================================================================
def detect_root_guid():
    GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    proc = subprocess.Popen(["azcopy","list", SAS[DATASET]],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    root = None
    while True:
        line = proc.stdout.readline()
        if not line: break
        if "INFO:" in line or "/" not in line: continue
        top = line.split("/")[0].strip(" ;")
        if GUID_RE.match(top): root = top; break
    proc.terminate(); proc.wait()
    assert root, "could not detect ROOT_GUID — check SAS token / container access"
    print("ROOT_GUID =", root); return root

def download_manifests(root):
    for blob in [f"{root}/file-manifest.tsv", f"{root}/dataset/participants.tsv"]:
        dest = MANIFEST / blob.rsplit("/",1)[-1]
        if dest.exists() and dest.stat().st_size > 0: continue
        r = subprocess.run(["azcopy","cp", blob_url(blob), str(dest)], capture_output=True, text=True)
        assert r.returncode == 0 and dest.exists() and dest.stat().st_size > 0, f"manifest dl failed: {blob}\n{r.stderr[-300:]}"
    print("manifest present:", sorted(os.listdir(MANIFEST)))

def _dt_shard_count(dt):
    if dt == "environment": return len(list((SHARD_ROOT/"environment").glob("*.parquet")))
    if dt == "wearable_blood_glucose": return len(list((SHARD_ROOT/"dexcom").glob("*.parquet")))
    if dt == "wearable_activity_monitor":
        return sum(len(list((SHARD_ROOT/m).glob("*.parquet"))) for m in GARMIN_FLATTEN)
    return 0

def _dt_done(dt, fm):
    """Skip download if this datatype is already fully sharded/output (avoids re-pulling env's ~50GB)."""
    if dt == "environment": return _dt_shard_count(dt) >= len(fm[(fm._dt=="environment") & (fm._ext=="csv")])
    if dt == "wearable_blood_glucose": return _dt_shard_count(dt) >= len(fm[(fm._dt=="wearable_blood_glucose") & (fm._ext=="json")])
    if dt == "wearable_activity_monitor": return _dt_shard_count(dt) >= len(fm[(fm._dt=="wearable_activity_monitor") & (fm._ext=="json")])
    if dt == "cardiac_ecg": return (CANONICAL/"ecg"/"recordings.npy").exists() and (CANONICAL/"ecg"/"index.parquet").exists()
    if dt == "clinical_data":
        rows = fm[(fm._dt=="clinical_data") & (fm._ext=="csv")]
        return all((CANONICAL/"clinical"/(Path(r["file_path"]).name.replace(".csv",".parquet"))).exists()
                   for r in rows.to_dict("records"))
    return False

def bulk_download(root, fm):
    """One recursive azcopy call per datatype, from {root}/dataset/{dt} (the manifest file_path
    starts with 'dataset/'). Skips datatypes already fully sharded. --overwrite=ifSourceNewer => resumable.
    Stragglers missed by bulk are fetched on-demand by dl_one() in the workers."""
    t0 = time.time()
    for dt in DT_KEEP:
        if _dt_done(dt, fm):
            print(f"[download] {dt}: already complete -> skip"); continue
        dest = RAW / "dataset" / dt; dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[download] {dt}  (recursive, resumable)")
        r = subprocess.run(["azcopy","cp", dir_url(f"{root}/dataset/{dt}"), str(dest),
                            "--recursive","--overwrite=ifSourceNewer","--cap-mbps","0"], check=False)
        if r.returncode != 0:
            print(f"  [download] {dt}: azcopy rc={r.returncode} (workers will fetch per-file via dl_one)")
    print(f"[download] done in {time.time()-t0:.0f}s")

def load_manifest():
    fm = pd.read_csv(MANIFEST/"file-manifest.tsv", sep="\t")
    def dt_of(p):
        for s in p.split("/"):
            if s in DT_KEEP: return s
        return None
    fm["_dt"]  = fm["file_path"].map(dt_of)
    fm = fm.dropna(subset=["_dt"]).copy()
    fm["_ext"] = fm["file_path"].map(lambda p: p.rsplit(".",1)[-1].lower() if "." in p else "")
    fm["_pid"] = fm["file_path"].map(lambda p: p.split("/")[-2] if "/" in p else "")
    def modality_of(r):
        dt, ext, fp = r["_dt"], r["_ext"], r["file_path"]
        if dt == "environment" and ext == "csv": return "environment"
        if dt == "wearable_blood_glucose" and ext == "json": return "dexcom"
        if dt == "wearable_activity_monitor":
            for mod, suf in GARMIN_SUFFIX.items():
                if fp.endswith(suf): return mod
        return None
    fm["_modality"] = fm.apply(modality_of, axis=1)
    print(f"manifest: {len(fm)} files | dt: {fm['_dt'].value_counts().to_dict()}")
    return fm

def process_shard_modality(modality, rows, out_path, label, root_guid):
    """ProcessPool(N_PROC_WORKERS) over rows -> per-participant shards -> merge."""
    SHARD_ROOT.joinpath(modality).mkdir(parents=True, exist_ok=True)
    total = len(rows); done = 0; t0 = time.time(); counts = {}
    with ProcessPoolExecutor(max_workers=N_PROC_WORKERS) as ex:
        futs = [ex.submit(_shard_worker, r, root_guid) for r in rows]
        for f in as_completed(futs):
            res = f.result(); st = res[0]; counts[st] = counts.get(st,0)+1
            if st not in ("wrote","resume","empty"):
                log_failure(modality, f"{res[0]} pid={res[2]} {res[3] if len(res)>3 else ''}")
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{label}] {done}/{total} ({time.time()-t0:.0f}s) {counts}")
    merge_shards(modality, SPECS[modality], out_path, label)

def process_clinical(fm, root_guid):
    (CANONICAL/"clinical").mkdir(parents=True, exist_ok=True)
    rows = fm[(fm._dt=="clinical_data") & (fm._ext=="csv")].to_dict("records")
    for row in rows:
        tbl = Path(row["file_path"]).name; src = RAW / row["file_path"]
        if not src.exists(): dl_one(row["file_path"], src, root_guid)
        if not src.exists(): log_failure("clinical", row["file_path"]); continue
        out = CANONICAL/"clinical"/(tbl.replace(".csv",".parquet"))
        if out.exists() and out.stat().st_size > 0: continue
        drop = [c for c in ("Unnamed: 0","X") if c in pd.read_csv(src, nrows=1).columns]
        df = pd.read_csv(src).drop(columns=drop, errors="ignore")
        df.to_parquet(out, compression="zstd")
        print(f"  [clinical/{tbl}] {len(df)} rows x {len(df.columns)} cols")

def process_ecg(fm, root_guid):
    (CANONICAL/"ecg").mkdir(parents=True, exist_ok=True)
    out_npy = CANONICAL/"ecg"/"recordings.npy"; out_idx = CANONICAL/"ecg"/"index.parquet"
    if out_npy.exists() and out_idx.exists():
        print("[ecg] already converted (recordings.npy + index.parquet present) -> skip"); return
    rows = fm[(fm._dt=="cardiac_ecg") & (fm._ext=="hea")].to_dict("records")
    arrays, idx_rows = [], []; t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=N_PROC_WORKERS) as ex:
        futs = [ex.submit(_ecg_worker, r, root_guid) for r in rows]
        for f in as_completed(futs):
            sig, idx, err = f.result(); done += 1
            if err: log_failure("ecg", err)
            elif sig is not None: arrays.append(sig); idx_rows.append(idx)
            if done % 50 == 0 or done == len(rows):
                print(f"  [ecg] {done}/{len(rows)} ({time.time()-t0:.0f}s)")
    if not arrays: print("[ecg] NO recordings converted"); return
    max_len = max(a.shape[1] for a in arrays); n_sig = arrays[0].shape[0]
    out_arr = np.full((len(arrays), n_sig, max_len), np.nan, dtype=np.float32); placed = 0
    for i, a in enumerate(arrays):
        if a.shape[0] != n_sig:
            log_failure("ecg", f"rec={idx_rows[i]['rec_id']} n_sig={a.shape[0]}!={n_sig} skip"); continue
        out_arr[placed, :, :a.shape[1]] = a; placed += 1
    if placed < len(arrays): out_arr = out_arr[:placed]
    np.save(out_npy, out_arr)
    pd.DataFrame(idx_rows[:placed]).to_parquet(out_idx, compression="zstd")
    print(f"[ecg] {placed}/{len(arrays)} recordings -> {out_arr.shape}, max_len={max_len}")

def participants():
    (CANONICAL/"metadata").mkdir(parents=True, exist_ok=True)
    pt = pd.read_csv(MANIFEST/"participants.tsv", sep="\t")
    pt = pt[["person_id","study_group","age","clinical_site","study_visit_date","recommended_split"]].copy()
    pt["label"] = pt["study_group"].map(LABEL_MAP)
    unmapped = pt.loc[pt["label"].isna(), "study_group"].unique()
    assert len(unmapped) == 0, f"Unmapped study_group: {list(unmapped)}"
    pt.to_parquet(CANONICAL/"metadata"/"participants.parquet", compression="zstd")
    print(f"[participants] {len(pt)} rows | labels: {pt['label'].value_counts().to_dict()}")
    return pt

def provenance(fm, pt, root):
    prov = CANONICAL/"metadata"/"manifests"; prov.mkdir(parents=True, exist_ok=True)
    for _, row in fm[fm._ext=="tsv"].drop_duplicates("file_path").iterrows():
        dest = prov / f"{row._dt}_manifest.tsv"
        if not dest.exists(): subprocess.run(["azcopy","cp", blob_url(f"{root}/{row.file_path}"), str(dest)], capture_output=True, text=True)
    shutil.copy(MANIFEST/"file-manifest.tsv", prov/"file-manifest.tsv")
    info = {"dataset":DATASET,"generated_at":datetime.now(timezone.utc).isoformat(),
            "root_guid":root,"n_participants":int(len(pt)),"label_map":LABEL_MAP,
            "workers":{"threads":N_WORKERS,"procs":N_PROC_WORKERS,"merge_readers":N_MERGE_WORKERS,"rclone":N_RCLONE,"zstd_level":ZSTD_LEVEL},
            "sentinels":{"heart_rate=0":"sensor off","stress in {-1,-2}":"invalid","respiratory_rate in {-1,-2}":"invalid","oxygen_saturation=0":"invalid"},
            "ecg":{"array":"ecg/recordings.npy float32 (N,12,max_len) NaN-padded (physical mV, NOT bit-exact)","true_sig_len_in":"ecg/index.parquet"},
            "timestamps":"tz-aware UTC","architecture":"bulk-download(5 azcopy) -> bulk-process(ProcessPool 32, local raw) -> parallel-merge(zstd level 1) -> rclone"}
    json.dump(info, open(CANONICAL/"metadata"/"dataset_info.json","w"), indent=2)
    print("provenance + dataset_info.json written.")

def validate():
    print("========== VALIDATION =========="); total_mb = 0
    for f in sorted(CANONICAL.rglob("*.parquet")):
        m = pq.ParquetFile(f).metadata; mb = f.stat().st_size/1024/1024; total_mb += mb
        print(f"  {f.relative_to(CANONICAL)}: {m.num_rows} rows, {m.num_columns} cols, {m.num_row_groups} rg  ({mb:.1f} MiB)")
    if (CANONICAL/"ecg"/"recordings.npy").exists():
        arr = np.load(CANONICAL/"ecg"/"recordings.npy", mmap_mode="r")
        idx = pd.read_parquet(CANONICAL/"ecg"/"index.parquet")
        assert arr.shape[0] == len(idx), f"ECG array {arr.shape[0]} != index {len(idx)}"
        print(f"  ecg/recordings.npy: {arr.shape} aligns with index ({len(idx)} rows)")
    print(f"\nTotal parquet: {total_mb:.1f} MiB | failures: {len(failures)}")
    for f_ in failures[:30]: print("  ", f_)
    assert not failures, f"{len(failures)} failures — see above"
    print("\n✓ VALIDATION OK — canonical tree at", CANONICAL)

def archive():
    print(f"[archive] rclone copy {CANONICAL} -> {GDRIVE_REMOTE} ...")
    rc = subprocess.run(["rclone","copy", str(CANONICAL), GDRIVE_REMOTE,
                    "--transfers",str(N_RCLONE),"--checkers",str(N_RCLONE),"--progress"], check=False).returncode
    if rc != 0:
        log_failure("archive", f"rclone rc={rc} — is the 'gdrive' remote configured? run `rclone config` to add it (OAuth, one-time)")
        print(f"[archive] FAILED rc={rc} — configure gdrive via `rclone config`, then re-run")
    else:
        print(f"[archive] done (verify: {GDRIVE_REMOTE})")

def main():
    t0 = time.time()
    ensure_deps()
    global ROOT_GUID
    ROOT_GUID = detect_root_guid()
    root = ROOT_GUID
    download_manifests(root)
    fm = load_manifest()
    diag_download(fm, root)
    pt = participants()
    if not SKIP_DOWNLOAD:
        bulk_download(root, fm)
    else:
        print("[download] SKIP_DOWNLOAD=1 -> using raw already on disk")
    # ---- process (local raw, ProcessPool 32) ----
    process_clinical(fm, root)
    process_shard_modality("environment", fm[(fm._modality=="environment")].to_dict("records"),
                           CANONICAL/"environment"/"environment.parquet", "env", root)
    process_shard_modality("dexcom", fm[(fm._modality=="dexcom")].to_dict("records"),
                           CANONICAL/"dexcom"/"cgm.parquet", "dex", root)
    for mod in GARMIN_FLATTEN:
        rows = fm[fm._modality==mod].to_dict("records")
        process_shard_modality(mod, rows, CANONICAL/"garmin"/f"{mod}.parquet", mod, root)
    process_ecg(fm, root)
    # ---- finalize ----
    provenance(fm, pt, root)
    validate()
    if not SKIP_ARCHIVE: archive()
    print(f"\nALL DONE in {time.time()-t0:.0f}s | canonical: {CANONICAL}")

if __name__ == "__main__":
    main()
