# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class ResidualAdaptor(nn.Module):
    """
    Residual adaptor module that learns a small delta and adds it to the input,
    followed by LayerNorm. Intended for use as a lightweight task-specific adapter.
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 32,
        dropout: float = 0.1,
        activation: Callable = F.relu,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        self.activation = activation

        # weight initialization (helpful for small adapters)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: input tensor of shape (B, ..., dim)

        Returns:
            Tensor of same shape as z after applying adaptor and layer norm.
        """
        out = self.down(z)
        out = self.activation(out)
        out = self.dropout(out)
        delta = self.up(out)
        return self.ln(z + delta)


class GateResidualAdaptor(nn.Module):
    """
    Gated residual adaptor. Learns a delta like ResidualAdaptor but scales it
    with a learned scalar gate (constrained to (0,1) with sigmoid).
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 32,
        dropout: float = 0.1,
        activation: Callable = F.relu,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        # a single scalar gate parameter; sigmoid(self.alpha) produces gate in (0,1)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.down(z)
        out = self.activation(out)
        out = self.dropout(out)
        delta = self.up(out)
        gate = torch.tanh(self.alpha)  # Keep identity at the beginning with a parameter
        return self.ln(z + gate * delta)


class SoftShareModule(nn.Module):
    """
    Soft-sharing multi-target module:
      - an encoder produces shared representation z
      - a list of per-target adaptors refine z -> z_k
      - a list of per-target heads map z_k -> scalar predictions

    Usage:
        module = SoftShareModule(encoder, dim, num_targets=4)
        preds = module(input)   # shape (B, num_targets)
    """
    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        num_targets: int = 4,
        adaptor_cls: Optional[Callable] = ResidualAdaptor,
        adaptor_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dim = dim
        self.num_targets = int(num_targets)
        adaptor_kwargs = adaptor_kwargs or {}

        self.adaptors = nn.ModuleList(
            adaptor_cls(dim, **adaptor_kwargs) for _ in range(self.num_targets)
        )
        self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.num_targets)])

        # initialize heads
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input to encoder

        Returns:
            Tensor of shape (B, num_targets) with one scalar per target.
        """
        z = self.encoder(x)
        outputs = []
        for adaptor, head in zip(self.adaptors, self.heads):
            z_k = adaptor(z)        # adapted representation
            y_k = head(z_k)         # (B, 1)
            outputs.append(y_k)
        return torch.cat(outputs, dim=1)  # (B, num_targets)


class GateSoftShareModule(nn.Module):
    """
    Accepts either:
      - z: torch.Tensor of shape (B, seq_len, d)  -> treated as shared representation, OR
      - raw_input: passed to self.encoder(...) if self.encoder is set.

    If constructed with encoder != None, old behavior remains supported.
    """
    def __init__(self, encoder: Optional[nn.Module], d: int, num_targets: int = 4):
        super().__init__()
        self.encoder = encoder  # optional; if provided, we can call it
        self.d = d
        self.num_targets = num_targets

        # per-target adaptors and heads (heads producing scalars by default)
        self.adaptors = nn.ModuleList([GateResidualAdaptor(d) for _ in range(num_targets)])
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(num_targets)])

    def forward(self, x, *args, **kwargs):
        """
        If x is a Tensor with last-dim == self.d, treat x as z (shared representation).
        Otherwise, if self.encoder is not None, call self.encoder(x) to obtain z.
        """
        # If the user passed a Tensor with expected hidden dim -> treat as z
        if isinstance(x, torch.Tensor) and x.dim() >= 2 and x.size(-1) == self.d:
            z = x
        else:
            if self.encoder is None:
                raise ValueError(
                    "GateSoftShareModule received non-tensor input but no encoder was provided."
                )
            # legacy behavior: encoder(batched_data, **kwargs) -> z
            z = self.encoder(x, *args, **kwargs)

        # Now z should be (B, seq_len, d) or (B, d) depending on your adaptor expectations.
        # If adaptors expect per-node/per-position inputs, keep seq_len; if they expect pooled z, pool here.
        # We'll assume adaptors operate on the per-position tensor and heads produce scalars by pooling.
        outputs = []
        for adaptor, head in zip(self.adaptors, self.heads):
            z_k = adaptor(z)            # (B, seq_len, d) or (B, d)
            # If z_k is (B, seq_len, d) and heads expect (B, d), you may want to pool:
            # pooled = z_k.mean(dim=1)  # simple mean pooling
            pooled = z_k.mean(dim=1) if z_k.dim() == 3 else z_k
            y_k = head(pooled)         # (B, 1)
            outputs.append(y_k)
        return torch.cat(outputs, dim=1)  # (B, num_targets)
