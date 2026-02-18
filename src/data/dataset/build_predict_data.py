# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset.pyg_dataset import PYGDataset
from data.data_wrapper import graphormer_collate, make_preprocess_item


PathLike = Union[str, Path]


def _normalize_smiles(smiles: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(smiles, str):
        return [smiles]
    return [str(s) for s in smiles]


def build_inference_dataset(
    smiles: Union[str, Sequence[str]],
    *,
    cache_root: PathLike,
    num_target: int,
    max_node: int = 256,
    multi_hop_max_dist: int = 20,
    spatial_pos_max: int = 20,
    transform=None,
) -> PYGDataset:
    """
    Build a cached dataset from SMILES for inference.
    - y_reg is filled with NaNs (shape [N, num_target]) so code paths expecting y/mask still work.
    """
    smiles_list = _normalize_smiles(smiles)

    # Dummy targets: NaNs with shape (num_target,)
    y_reg = [ [float("nan")] * num_target for _ in range(len(smiles_list)) ]

    if transform is None:
        transform = make_preprocess_item(int(multi_hop_max_dist))

    ds = PYGDataset(
        root=str(cache_root),
        X=smiles_list,
        y_reg=y_reg,
        max_node=max_node,
        multi_hop_max_dist=multi_hop_max_dist,
        spatial_pos_max=spatial_pos_max,
        seed=123,
        use_scaffold_split=False,   # for inference splits aren't important
        cache_splits=False,         # optional: speed up
        transform=transform,
    )
    return ds


def build_loader_for_inference(
    ds: PYGDataset,
    *,
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=graphormer_collate,
    )
