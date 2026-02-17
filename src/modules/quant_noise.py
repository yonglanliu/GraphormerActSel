# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fake_quantize_8bit_symmetric(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-tensor symmetric fake quantization to int8 range [-127, 127] with STE."""
    max_abs = w.detach().abs().amax()
    scale = torch.clamp(max_abs / 127.0, min=eps)
    q = torch.clamp((w / scale).round(), -127, 127) * scale
    return w + (q - w).detach()


def quant_noise(module: nn.Module, q_noise: float, qn_block_size: int) -> nn.Module:
    """
    Training-time regularizer: randomly fake-quantize blocks of the weight matrix.

    Args:
        module: typically nn.Linear to be wrapped
        q_noise: probability of quantizing a block (e.g., 0.05)
        qn_block_size: block size along both output and input dims (e.g., 8)

    Notes:
        - Applies only in module.train() mode.
        - If weight dims aren't divisible by qn_block_size, falls back to block_size=1.
        - Exposes .weight and .bias attributes for init/checkpoint compatibility.
    """
    if q_noise is None or q_noise <= 0:
        return module

    if not isinstance(module, nn.Linear):
        raise TypeError("quant_noise currently supports nn.Linear only.")

    class QuantNoiseLinear(nn.Module):
        def __init__(self, base: nn.Linear, p: float, block: int):
            super().__init__()
            self.base = base
            self.p = float(p)
            self.block = int(block)

        @property
        def weight(self) -> torch.Tensor:
            return self.base.weight

        @property
        def bias(self) -> Optional[torch.Tensor]:
            return self.base.bias

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.base.weight
            b = self.base.bias

            # Only apply during training
            if (not self.training) or self.p <= 0:
                return F.linear(x, w, b)

            out_dim, in_dim = w.shape
            block = max(1, self.block)

            # If not divisible, degrade gracefully (or implement padding if you prefer)
            if (out_dim % block) != 0 or (in_dim % block) != 0:
                block = 1

            bo, bi = out_dim // block, in_dim // block

            # block mask: 1 => quantize this block
            block_mask = (torch.rand(bo, bi, device=w.device) < self.p).to(w.dtype)
            mask = block_mask.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)

            w_q = _fake_quantize_8bit_symmetric(w)
            w_noisy = w * (1.0 - mask) + w_q * mask

            return F.linear(x, w_noisy, b)

    return QuantNoiseLinear(module, q_noise, qn_block_size)
