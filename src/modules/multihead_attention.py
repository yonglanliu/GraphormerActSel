# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from modules.quant_noise import quant_noise


class MultiheadAttention(nn.Module):
    """Multi-headed self-attention (fairseq-free)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        self_attention: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        # Keep the same attribute names as fairseq version for checkpoint compatibility
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        self.reset_parameters()
        self.onnx_trace = False  # kept for API parity

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # self-attention: key/value are query
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q = q * self.scaling

        # (T,B,C) -> (B*H, T, D)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Workaround: Optional mask sometimes passed as scalar tensor
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights_float)

        attn = torch.bmm(attn_probs.type_as(v), v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out

    def apply_sparse_mask(self, attn_weights: Tensor, tgt_len: int, src_len: int, bsz: int) -> Tensor:
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        # unchanged; this is pure tensor/key manipulation
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in list(state_dict.keys()):
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]
                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict:
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
                    keys_to_remove.append(k_bias)

        for k in keys_to_remove:
            del state_dict[k]
        for key, value in items_to_add.items():
            state_dict[key] = value
