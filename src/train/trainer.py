#!/usr/bin/env python3
# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

"""
Train Graphormer on multitask regression with:
- argparse CLI
- separate LR for encoder vs adaptor/head
- optional pretrained checkpoint load
- cosine or plateau scheduler
- dict-batch training loop in train.train_reg
"""

import argparse
from pathlib import Path


import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

from models.graphormer import GraphormerModel
from train.train_reg import set_seed, make_param_groups, train_with_eval_earlystop
from data.dataset.pyg_dataset import PYGDataset
from data.data_wrapper import graphormer_collate, make_preprocess_item
from config.config import get_config



# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Train Graphormer multitask regression")

    # Data
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--root_path", type=Path, required=True)

    # Output
    p.add_argument("--save_dir", type=Path, default=Path("./runs"))
    p.add_argument("--run_name", type=str, default="graphormer_regress")

    # Pretrained checkpoint
    p.add_argument("--checkpoint", type=Path, default=None)

    # Training
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_aux", type=float, default=0.3)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--debug_nan", action="store_true")

    # LRs
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--adaptor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--freeze_encoder", action="store_true")

    # Scheduler
    p.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    p.add_argument("--tmax", type=int, default=50)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_patience", type=int, default=3)

    # Dataset-specific knobs
    p.add_argument("--use_scaffold_split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_node", type=int, default=512)
    p.add_argument("--multi_hop_max_dist", type=int, default=32)
    p.add_argument("--spatial_pos_max", type=int, default=32)
    p.add_argument("--cached_dataset_path", type=Path, default=None)

    return p.parse_args()


def _fingerprint_smiles(df: pd.DataFrame, n: int = 25) -> str:
    # stable string for quick sanity check
    xs = df["SMILES"].astype(str).head(n).tolist()
    return "|".join(xs)

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seed(args.seed)

    # Load dataframe (make ordering stable)
    df = pd.read_parquet(args.data_path)
    df = df.sort_values("SMILES").reset_index(drop=True)

    # Dataset
    if os.path.exists(args.cached_dataset_path):
        root_t = args.cached_dataset_path.parent.parent 
        ds = PYGDataset.load_all(str(args.cached_dataset_path), transform=make_preprocess_item(args.multi_hop_max_dist), root=str(root_t))
        print("Loaded cached dataset:", args.cached_dataset_path)
    else:
        ds = PYGDataset(
            root=str(args.root_path),
            X=df["SMILES"].tolist(),
            y_reg=df[["ISOA", "ISOB", "ISOC", "ISOD"]].to_numpy().tolist(),
            use_scaffold_split=args.use_scaffold_split,
            seed=args.seed,
            max_node=args.max_node,
            multi_hop_max_dist=args.multi_hop_max_dist,
            spatial_pos_max=args.spatial_pos_max,
        )
        ds.save_all(args.cached_dataset_path)

    train_loader = DataLoader(
        ds.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=graphormer_collate,
    )
    valid_loader = DataLoader(
        ds.valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=graphormer_collate,
    )

    # Config Model
    cfg = get_config()
    model = GraphormerModel.build_model(cfg).to(device)

    # Load checkpoint (optional)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state_dict.pop("encoder.embed_out.weight", None)
        state_dict.pop("encoder.lm_output_learned_bias", None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded:", args.checkpoint)
        print("Missing:", missing)
        print("Unexpected:", unexpected)

    # Optimizer with param groups
    param_groups = make_param_groups(
        model,
        encoder_lr=args.encoder_lr,
        adaptor_lr=args.adaptor_lr,
        weight_decay=args.weight_decay,
        adaptor_name_keywords=("soft_share", "adaptor", "adapter", "head", "heads"),
        freeze_encoder=args.freeze_encoder,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True,
        )
    else:
        scheduler = None

    # Train
    history = train_with_eval_earlystop(
        model,
        train_loader,
        valid_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        lambda_aux=args.lambda_aux,
        huber_delta=args.huber_delta,
        save_dir=str(args.save_dir),
        run_name=args.run_name,
        save_every_epochs=1,
        patience=args.patience,
        min_delta=args.min_delta,
        debug_nan=args.debug_nan,
    )

    print("Done. Saved to:", str(Path(args.save_dir) / args.run_name))
    return history


if __name__ == "__main__":
    main()
