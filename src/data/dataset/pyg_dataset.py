# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

"""

A single-file PyG InMemoryDataset that can:
  1) Build + cache graphs to <root>/processed/data.pt
  2) Build + cache split indices to <root>/processed/split_*.pt
  3) Export EVERYTHING (data+slices+splits+meta) into ONE file: <out>.pt
  4) Load from that ONE file without needing original X/y_reg

Usage (build):
  ds = PYGDataset(
        root=cache_root,
        X=smiles_list,
        y_reg=targets,
        max_node=256,
        multi_hop_max_dist=20,
        spatial_pos_max=20,
        seed=123,
        use_scaffold_split=True,
      )
  ds.save_all("scaffold_seed123.pt")

Usage (load one-file artifact):
  ds = PYGDataset.load_all("scaffold_seed123.pt", transform=make_preprocess_item(...))
  train, valid, test = ds.train, ds.valid, ds.test
"""

from __future__ import annotations

from typing import List, Sequence, Optional, Dict, Any
from pathlib import Path
import os

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import rdmolops

# project imports
from data.data_wrapper import make_preprocess_item
from utils.mol import smiles2graph
from utils.data_split import scaffold_split


class PYGDataset(InMemoryDataset):
    """
    Cached PyG InMemoryDataset with optional cached split indices.

    Caches:
      - graphs to: <root>/processed/data.pt
      - splits to: <root>/processed/split_<tag>_seed<seed>_<frac>.pt

    Can also export a single portable file containing:
      - data, slices, split indices, and meta/config
    """

    def __init__(
        self,
        root: str | Path,
        X: Sequence[str],
        y_reg: Sequence,
        max_node: int,
        multi_hop_max_dist: int,
        spatial_pos_max: int,
        *,
        seed: int = 123,
        use_scaffold_split: bool = False,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        y_dtype: torch.dtype = torch.float32,
        cache_splits: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.X = [str(s) for s in X]
        self.y_reg = list(y_reg)
        if len(self.X) != len(self.y_reg):
            raise ValueError(
                f"X and y_reg must have same length. Got len(X)={len(self.X)} len(y_reg)={len(self.y_reg)}"
            )

        self.seed = int(seed)
        self.use_scaffold_split = bool(use_scaffold_split)
        self.frac_train = float(frac_train)
        self.frac_valid = float(frac_valid)
        self.frac_test = float(frac_test)
        self.y_dtype = y_dtype
        self.cache_splits = bool(cache_splits)

        total = self.frac_train + self.frac_valid + self.frac_test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("frac_train + frac_valid + frac_test must sum to 1.0")

        # Compute dataset-level meta (also stores self.multi_hop_max_dist)
        self.__get_graph_metainfo(max_node, multi_hop_max_dist, spatial_pos_max)

        # If user didn't pass a transform, use one that fixes edge_input distance dim to self.multi_hop_max_dist
        if transform is None:
            transform = make_preprocess_item(int(self.multi_hop_max_dist))

        super().__init__(str(root), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self._load_or_make_splits()
        self.train = self._subset(self.train_idx)
        self.valid = self._subset(self.valid_idx)
        self.test = self._subset(self.test_idx)

    # -------------------------
    # Meta / processing
    # -------------------------
    def __get_graph_metainfo(self, max_node: int, multi_hop_max_dist: int, spatial_pos_max: int):
        mols = [Chem.MolFromSmiles(smi) for smi in self.X]
        if any(m is None for m in mols):
            bad = [self.X[i] for i, m in enumerate(mols) if m is None][:5]
            raise ValueError(f"RDKit failed to parse some SMILES, e.g.: {bad}")

        max_nodes_actual = max(m.GetNumAtoms() for m in mols)
        self.max_node = min(int(max_node), int(max_nodes_actual))

        max_dist = 0
        for m in mols:
            dist = rdmolops.GetDistanceMatrix(m)  # numpy
            max_dist = max(max_dist, int(np.amax(dist)))

        self.multi_hop_max_dist = min(int(multi_hop_max_dist), int(max_dist))
        self.spatial_pos_max = min(int(spatial_pos_max), int(max_dist))

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        data_list: List[Data] = []

        for smi, target in zip(self.X, self.y_reg):
            g = smiles2graph(smi)
            if isinstance(g, Data):
                d = g
            elif isinstance(g, dict):
                d = Data(**g)
            else:
                raise TypeError(f"smiles2graph must return a PyG Data or dict, got {type(g)}")

            d.smiles = smi
            d.y = torch.as_tensor(target, dtype=self.y_dtype).view(1, -1)
            data_list.append(d)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    # -------------------------
    # Splits
    # -------------------------
    def _split_cache_path(self) -> str:
        tag = "scaffold" if self.use_scaffold_split else "random"
        frac_tag = f"{self.frac_train:.3f}_{self.frac_valid:.3f}_{self.frac_test:.3f}".replace(".", "p")
        return os.path.join(self.processed_dir, f"split_{tag}_seed{self.seed}_{frac_tag}.pt")

    def _load_or_make_splits(self):
        split_path = self._split_cache_path()

        if self.cache_splits and os.path.exists(split_path):
            obj = torch.load(split_path)
            self.train_idx = obj["train_idx"]
            self.valid_idx = obj["valid_idx"]
            self.test_idx = obj["test_idx"]
            return

        n = self.len()

        if self.use_scaffold_split:
            train_idx, valid_idx, test_idx = scaffold_split(
                smiles_list=self.X,
                frac_train=self.frac_train,
                frac_valid=self.frac_valid,
                seed=self.seed,
            )
            self.train_idx = torch.as_tensor(train_idx, dtype=torch.long)
            self.valid_idx = torch.as_tensor(valid_idx, dtype=torch.long)
            self.test_idx = torch.as_tensor(test_idx, dtype=torch.long)
        else:
            gen = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=gen)
            n_train = int(n * self.frac_train)
            n_valid = int(n * self.frac_valid)
            self.train_idx = perm[:n_train]
            self.valid_idx = perm[n_train : n_train + n_valid]
            self.test_idx = perm[n_train + n_valid :]

        if self.cache_splits:
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(
                {"train_idx": self.train_idx, "valid_idx": self.valid_idx, "test_idx": self.test_idx},
                split_path,
            )

    def _subset(self, idx: torch.Tensor) -> "PYGSubset":
        return PYGSubset(self, idx)

    # -------------------------
    # One-file save/load (DATA + SPLITS in same file)
    # -------------------------
    def _meta_dict(self) -> Dict[str, Any]:
        return dict(
            seed=int(self.seed),
            use_scaffold_split=bool(self.use_scaffold_split),
            frac_train=float(self.frac_train),
            frac_valid=float(self.frac_valid),
            frac_test=float(self.frac_test),
            max_node=int(getattr(self, "max_node", -1)),
            multi_hop_max_dist=int(getattr(self, "multi_hop_max_dist", -1)),
            spatial_pos_max=int(getattr(self, "spatial_pos_max", -1)),
            y_dtype=str(self.y_dtype),
        )

    def save_all(self, out_path: str | Path):
        """
        Save graphs + split indices + meta into ONE .pt file.
        """
        out_path = str(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Make sure indices exist
        if not hasattr(self, "train_idx"):
            self._load_or_make_splits()

        payload = {
            "data": self._data,
            "slices": self.slices,
            "train_idx": self.train_idx.cpu(),
            "valid_idx": self.valid_idx.cpu(),
            "test_idx": self.test_idx.cpu(),
            "meta": self._meta_dict(),
        }
        torch.save(payload, out_path)

    @classmethod
    def load_all(
        cls,
        path: str | Path,
        *,
        transform=None,
        root: Optional[str | Path] = None,
        map_location: str | torch.device = "cpu",
    ) -> "PYGDataset":
        """
        Load graphs + splits from a single .pt file.
        Does NOT require original X/y_reg.
        """
        obj_dict = torch.load(str(path), map_location=map_location)

        # Build a blank object without calling __init__
        obj = cls.__new__(cls)

        # Minimal attributes to satisfy InMemoryDataset.__init__ and any property access
        # (these prevent accidental calls to process() that depend on X/y_reg)
        obj.X = []              # minimal placeholder
        obj.y_reg = []          # minimal placeholder
        obj.pre_transform = None
        obj.pre_filter = None

        # Choose a root just to satisfy InMemoryDataset internals.
        # If provided, we use it; otherwise we use the directory of the file.
        if root is None:
            root = str(Path(path).resolve().parent / "_loaded_dataset_root")
        os.makedirs(str(root), exist_ok=True)

        # Provide a default transform if user didn't pass one, based on meta
        if transform is None:
            mh = int(obj_dict.get("meta", {}).get("multi_hop_max_dist", 20))
            transform = make_preprocess_item(mh)

        # Now initialize InMemoryDataset (safe because we've set required attrs)
        InMemoryDataset.__init__(obj, root=str(root), transform=transform)

        # Load internal storage that PyG expects (_data / slices) or (data, slices) pair
        # If you saved using self._data, use that; otherwise the saved payload has 'data'/'slices'
        # The 'save_all' we provided stored 'data' and 'slices' (or 'data' could be _data).
        obj.data = obj_dict["data"]
        obj.slices = obj_dict["slices"]

        # Load split indices
        obj.train_idx = obj_dict["train_idx"].long()
        obj.valid_idx = obj_dict["valid_idx"].long()
        obj.test_idx = obj_dict["test_idx"].long()

        # Load meta
        meta = obj_dict.get("meta", {})
        obj.seed = int(meta.get("seed", 123))
        obj.use_scaffold_split = bool(meta.get("use_scaffold_split", False))
        obj.frac_train = float(meta.get("frac_train", 0.8))
        obj.frac_valid = float(meta.get("frac_valid", 0.1))
        obj.frac_test = float(meta.get("frac_test", 0.1))
        obj.max_node = int(meta.get("max_node", -1))
        obj.multi_hop_max_dist = int(meta.get("multi_hop_max_dist", -1))
        obj.spatial_pos_max = int(meta.get("spatial_pos_max", -1))
        obj.y_dtype = torch.float32
        obj.cache_splits = True

        # Build subsets
        obj.train = obj._subset(obj.train_idx)
        obj.valid = obj._subset(obj.valid_idx)
        obj.test = obj._subset(obj.test_idx)
        return obj



class PYGSubset(torch.utils.data.Dataset):
    """
    IMPORTANT: Do NOT call preprocess_item here.
    The base dataset's `transform` already applies preprocessing on __getitem__.
    """

    def __init__(self, base: PYGDataset, idx: torch.Tensor):
        self.base = base
        self.idx = idx

    def __len__(self) -> int:
        return int(self.idx.numel())

    def __getitem__(self, idx: int) -> Data:
        base_idx = int(self.idx[idx])
        item = self.base[base_idx]  # already transformed
        item.idx = base_idx
        return item


