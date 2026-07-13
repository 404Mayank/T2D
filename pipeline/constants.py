"""Derived constants and small pure helpers from config."""

from __future__ import annotations

from typing import Any


def site_tz_map(cfg: dict) -> dict[str, str]:
    return dict(cfg["time"]["site_tz"])


def survey_sentinel_set(cfg: dict) -> set[float]:
    return {float(x) for x in cfg["sentinels"]["survey_codes"]}


def classify_prefix(prefix: str, cfg: dict) -> str:
    """Return class label for a source_value prefix. First matching rule wins by priority."""
    p = (prefix or "").strip()
    cl = cfg["classify"]

    def starts(plist: list[str]) -> bool:
        return any(p == s or p.startswith(s) for s in plist)

    # Priority: hard_exclude > retinal > labs > metadata > keep blocks > keep_survey default-ish
    if p in set(cl.get("hard_exclude_exact") or []):
        return "hard_exclude"
    if starts(cl.get("hard_exclude_prefixes") or []):
        return "hard_exclude"
    if starts(cl.get("retinal_drop_prefixes") or []):
        return "retinal_drop"
    labs = list(cl.get("labs_upper_bound_prefixes") or []) + list(
        cl.get("labs_upper_bound_extra_prefixes") or []
    )
    if starts(labs):
        return "labs_upper_bound"
    # metadata by prefix or suffix (paidstartts, cesmpdat, …)
    if starts(cl.get("metadata_drop_prefixes") or []):
        return "metadata_drop"
    for sfx in cl.get("metadata_drop_suffixes") or []:
        if p.endswith(sfx):
            return "metadata_drop"
    # keep rules — mhoccur_pdr already hard_excluded above
    if starts(cl.get("keep_onboarding_prefixes") or []):
        return "keep_onboarding"
    if starts(cl.get("keep_comorbidity_prefixes") or []):
        return "keep_comorbidity"
    if starts(cl.get("keep_mood_prefixes") or []):
        return "keep_mood"
    if starts(cl.get("keep_diet_prefixes") or []):
        return "keep_diet"
    if starts(cl.get("keep_survey_prefixes") or []):
        return "keep_survey"

    # Heuristic leftovers — force review on diabetes-adjacent unknowns
    low = p.lower()
    if any(
        k in low
        for k in (
            "hba1c",
            "a1c",
            "insulin",
            "glucose",
            "c_peptide",
            "predm",
            "dm2",
            "dm1",
            "retino",
            "dri",
            "glc",
            "neuro",
        )
    ):
        return "borderline"
    if p.startswith("cmtrt_"):
        return "borderline"
    if p.startswith("import_"):
        return "borderline"
    if p.startswith("mhterm_") or p.startswith("mh_"):
        return "borderline"

    return (cfg.get("classify") or {}).get("unmatched_class") or "borderline"


def block_for_prefix(prefix: str, class_name: str, cfg: dict) -> str | None:
    """Which feature block a kept prefix belongs to."""
    if not class_name.startswith("keep"):
        return None
    if class_name == "keep_onboarding":
        return "onboarding"
    if class_name == "keep_comorbidity":
        return "comorbidity"
    if class_name == "keep_mood":
        return "mood"
    if class_name == "keep_diet":
        return "diet"
    # keep_survey: try block prefix lists
    blocks = cfg.get("clinical", {}).get("blocks") or {}
    for block, key in (
        ("onboarding", "onboarding_prefixes"),
        ("comorbidity", "comorbidity_prefixes"),
        ("mood", "mood_prefixes"),
        ("diet", "diet_prefixes"),
    ):
        prefs = blocks.get(key) or []
        if any(prefix == s or prefix.startswith(s) for s in prefs):
            return block
    return "other_keep"


def activity_intensity(name: str | None, cfg: dict) -> float | None:
    imap = (cfg.get("intervals") or {}).get("activity", {}).get("intensity_map") or {}
    key = (name or "").strip().lower()
    if key == "" or key not in imap:
        blank_as = (cfg.get("intervals") or {}).get("activity", {}).get("blank_name_as", "unknown")
        key = blank_as if key == "" else key
    if key not in imap:
        return None
    val = imap[key]
    if val is None:
        return None
    return float(val)


def report_default() -> dict[str, Any]:
    return {
        "stages": {},
        "counts": {},
        "warnings": [],
        "errors": [],
    }
