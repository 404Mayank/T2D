"""attn_lstm_64 — BiLSTM + mask attention + day-level glucose head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLSTM64(nn.Module):
    def __init__(
        self,
        d_in: int,
        *,
        hidden: int = 64,
        n_classes: int = 4,
        n_glu: int = 8,
        dropout: float = 0.2,
        bidirectional: bool = True,
        green_dim: int = 0,
    ):
        super().__init__()
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.green_dim = int(green_dim)
        self.input = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0,
        )
        lstm_out = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(lstm_out, hidden) if bidirectional else nn.Identity()
        self.attn = nn.Linear(hidden, 1)
        # Late-fuse person GREEN into class head only (glu stays on h_t).
        self.class_head = nn.Linear(hidden + self.green_dim, n_classes)
        self.glu_head = nn.Linear(hidden, n_glu)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self, x: torch.Tensor, watch_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,T,d], watch_mask: [B,T] bool
        returns h_t [B,T,H], z [B,H], attn weights [B,T]
        """
        b, t, _ = x.shape
        h = self.input(x)
        # lengths from mask (at least 1 to satisfy pack)
        lengths = watch_mask.long().sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=t
        )
        h_t = self.proj(out)
        h_t = self.dropout(h_t)

        scores = self.attn(h_t).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(~watch_mask, -1e9)
        # all-masked safety
        all_bad = ~watch_mask.any(dim=1)
        if all_bad.any():
            scores = scores.clone()
            scores[all_bad, 0] = 0.0
        alpha = torch.softmax(scores, dim=1)
        z = torch.sum(alpha.unsqueeze(-1) * h_t, dim=1)
        return h_t, z, alpha

    def forward(
        self,
        x: torch.Tensor,
        watch_mask: torch.Tensor,
        green: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_t, z, alpha = self.encode(x, watch_mask)
        if self.green_dim > 0:
            if green is None:
                raise ValueError("model expects green [B, green_dim]")
            z_cls = torch.cat([z, green], dim=-1)
        else:
            z_cls = z
        logits = self.class_head(z_cls)
        glu_pred = self.glu_head(h_t)  # [B,T,8] — same h_t as attention
        return {"logits": logits, "glu_pred": glu_pred, "z": z, "alpha": alpha}


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """pred/target [B,T,C], mask [B,T]. Mean over masked elements & channels."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not mask.any():
        return pred.new_zeros(())
    m = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target)[m]
    return diff.pow(2).mean()


def ce_loss(
    logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor | None
) -> torch.Tensor:
    if class_weights is not None:
        return F.cross_entropy(logits, y, weight=class_weights)
    return F.cross_entropy(logits, y)
