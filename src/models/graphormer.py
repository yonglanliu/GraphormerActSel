# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license found in the LICENSE file.

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.graphormer_graph_encoder import (
    init_graphormer_params,
    GraphormerGraphEncoder,
)
from modules.soft_sharing import (
    GateSoftShareModule
)

from pretrain.load_pretrained_model import load_pretrained_state_dict

logger = logging.getLogger(__name__)

# -----------------------------
# Lightweight registry
# -----------------------------

MODEL_REGISTRY = {}
ARCH_REGISTRY = {}


def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def register_model_architecture(model_name: str, arch_name: str):
    def deco(fn):
        ARCH_REGISTRY[(model_name, arch_name)] = fn
        return fn
    return deco


def safe_hasattr(obj, attr: str) -> bool:
    return hasattr(obj, attr)


AVAILABLE_ACTIVATIONS = (
    "relu", "gelu", "gelu_fast", "tanh", "sigmoid", "silu", "swish"
)


def get_activation_fn(name: str) -> Callable:
    name = (name or "gelu").lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "gelu_fast":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name in ("silu", "swish"):
        return F.silu
    raise ValueError(f"Unknown activation_fn='{name}'")


# -----------------------------
# Model
# -----------------------------

@register_model("graphormer")
class GraphormerModel(nn.Module):
    def __init__(self, cfg, encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

        if getattr(cfg, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

        self.encoder_embed_dim = cfg.encoder_embed_dim

        # optional pretrained loading
        if getattr(cfg, "pretrained_model_name", "none") != "none":
            state_dict = load_pretrained_state_dict(cfg.pretrained_model_name)
            self.load_state_dict(state_dict, strict=False)

            if not getattr(cfg, "load_pretrained_model_output_layer", False):
                if hasattr(self.encoder, "reset_output_layer_parameters"):
                    self.encoder.reset_output_layer_parameters()

    def max_nodes(self):
        return self.encoder.max_nodes()

    @classmethod
    def build_model(cls, cfg):
        base_architecture(cfg)

        if not safe_hasattr(cfg, "max_nodes"):
            cfg.max_nodes = getattr(cfg, "tokens_per_sample", None)
            if cfg.max_nodes is None:
                raise ValueError("cfg.max_nodes must be set")

        logger.info(cfg)

        encoder = GraphormerEncoder(cfg)
        return cls(cfg, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._max_nodes = cfg.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=cfg.num_atoms,
            num_in_degree=cfg.num_in_degree,
            num_out_degree=cfg.num_out_degree,
            num_edges=cfg.num_edges,
            num_spatial=cfg.num_spatial,
            num_edge_dis=cfg.num_edge_dis,
            edge_type=cfg.edge_type,
            multi_hop_max_dist=cfg.multi_hop_max_dist,
            num_encoder_layers=cfg.encoder_layers,
            embedding_dim=cfg.encoder_embed_dim,
            ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
            num_attention_heads=cfg.encoder_attention_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.act_dropout,
            encoder_normalize_before=cfg.encoder_normalize_before,
            pre_layernorm=cfg.pre_layernorm,
            apply_graphormer_init=cfg.apply_graphormer_init,
            activation_fn=cfg.activation_fn,
        )

        self.share_input_output_embed = cfg.share_encoder_input_output_embed
        self.embed_out: Optional[nn.Linear] = None
        self.lm_output_learned_bias: Optional[nn.Parameter] = None

        self.load_softmax = not getattr(cfg, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(
            cfg.encoder_embed_dim, cfg.encoder_embed_dim
        )
        self.activation_fn = get_activation_fn(cfg.activation_fn)
        self.layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)

        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            self.embed_out = nn.Linear(
                cfg.encoder_embed_dim,
                cfg.num_classes,
                bias=False,
            )

        # -----------------------------
        # Gate soft-sharing integration
        # -----------------------------
        # Configurable flags:
        #   cfg.use_gate_soft_sharing (bool): enable GateSoftShareModule
        #   cfg.num_targets (int): number of soft-shared targets (default 4)
        self.use_gate_soft_sharing = getattr(cfg, "use_gate_soft_sharing", False)
        self.num_targets = int(getattr(cfg, "num_targets", 4))

        # Use soft sharing networks
        if self.use_gate_soft_sharing:
            # instantiate GateSoftShareModule WITHOUT an encoder
            self.soft_share = GateSoftShareModule(
                encoder=None,  # not needed now; forward will take z
                d=cfg.encoder_embed_dim,
                num_targets=self.num_targets,
            )
        else:
            self.soft_share = None

    def reset_output_layer_parameters(self):
        if self.embed_out is not None:
            self.embed_out.reset_parameters()
        if self.lm_output_learned_bias is not None:
            with torch.no_grad():
                self.lm_output_learned_bias.zero_()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):

        # Original behavior: compute encoder z, transform through LM head and optionally head
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        if masked_tokens is not None:
            raise NotImplementedError

        z = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        ##---------------Edit by Yonglan Liu----------------------------
        # If soft sharing is enabled, pass z (not batched_data) to the soft-share module
        if self.soft_share is not None:
            # soft_share.forward accepts z directly now
            return self.soft_share(z)
        
        # Otherwise continue with existing LM head behavior
        x=z
        ##--------------------------------------------------------------
        if self.embed_out is not None:
            x = self.embed_out(x)

        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

    def max_nodes(self):
        return self._max_nodes


# -----------------------------
# Architectures
# -----------------------------

@register_model_architecture("graphormer", "graphormer")
def base_architecture(cfg):
    cfg.dropout = getattr(cfg, "dropout", 0.1)
    cfg.attention_dropout = getattr(cfg, "attention_dropout", 0.1)
    cfg.act_dropout = getattr(cfg, "act_dropout", 0.0)

    cfg.encoder_ffn_embed_dim = getattr(cfg, "encoder_ffn_embed_dim", 4096)
    cfg.encoder_layers = getattr(cfg, "encoder_layers", 6)
    cfg.encoder_attention_heads = getattr(cfg, "encoder_attention_heads", 8)

    cfg.encoder_embed_dim = getattr(cfg, "encoder_embed_dim", 1024)
    cfg.share_encoder_input_output_embed = getattr(cfg, "share_encoder_input_output_embed", False)

    cfg.apply_graphormer_init = getattr(cfg, "apply_graphormer_init", False)
    cfg.activation_fn = getattr(cfg, "activation_fn", "gelu")
    cfg.encoder_normalize_before = getattr(cfg, "encoder_normalize_before", True)
    cfg.pre_layernorm = getattr(cfg, "pre_layernorm", False)

    # defaults for soft sharing
    cfg.use_gate_soft_sharing = getattr(cfg, "use_gate_soft_sharing", False)
    cfg.num_targets = getattr(cfg, "num_targets", 4)


@register_model_architecture("graphormer", "graphormer_base")
def graphormer_base_architecture(cfg):
    cfg.encoder_embed_dim = getattr(cfg, "encoder_embed_dim", 768)
    cfg.encoder_layers = getattr(cfg, "encoder_layers", 12)
    cfg.encoder_attention_heads = getattr(cfg, "encoder_attention_heads", 32)
    cfg.encoder_ffn_embed_dim = getattr(cfg, "encoder_ffn_embed_dim", 768)

    base_architecture(cfg)
