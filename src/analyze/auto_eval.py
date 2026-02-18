# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path
from typing import Union, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from config.config import get_config
from data.dataset.pyg_dataset import PYGDataset
from data.data_wrapper import graphormer_collate, make_preprocess_item
from analyze.analyzer import GraphormerAnalyzer


PathLike = Union[str, Path]


def _build_test_loader_and_analyzer(
    data_path: PathLike,
    checkpoint_path: PathLike,
    *,
    batch_size: int = 32,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
):
    cfg = get_config()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = Path(data_path)
    root = data_path.parent.parent

    ds = PYGDataset.load_all(
        str(data_path),
        transform=make_preprocess_item(cfg.multi_hop_max_dist),
        root=str(root),
    )

    test_loader = DataLoader(
        ds.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=graphormer_collate,
    )

    analyzer = GraphormerAnalyzer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        device=device,
    )

    return cfg, device, ds, test_loader, analyzer


def _get_task_names(test_data, task_names: Optional[Sequence[str]] = None) -> list[str]:
    if task_names is not None:
        return list(task_names)
    T = test_data.pred_y.size(1)
    return [f"task{i}" for i in range(T)]


def run_test_eval(data_path: PathLike, checkpoint_path: PathLike, *, batch_size: int = 16):
    _, _, _, test_loader, analyzer = _build_test_loader_and_analyzer(
        data_path,
        checkpoint_path,
        batch_size=batch_size,
        num_workers=1,
    )

    test_loss = analyzer.evaluate_loss(test_loader)
    print(f"Test loss: {test_loss:.6f}")

    test_data = analyzer.predict(test_loader)
    task_names = _get_task_names(test_data)

    metrics = analyzer.evaluate_all_metrics(test_data, task_names)
    analyzer.print_metrics(metrics)

    return test_loss, metrics


def run_test_eval_to_csv(
    data_path: PathLike,
    checkpoint_path: PathLike,
    *,
    batch_size: int = 32,
    out_path: PathLike = "test_metrics.csv",
    task_names: Optional[Sequence[str]] = None,
):
    _, _, _, test_loader, analyzer = _build_test_loader_and_analyzer(
        data_path,
        checkpoint_path,
        batch_size=batch_size,
        num_workers=2,
    )

    test_loss = analyzer.evaluate_loss(test_loader)
    print(f"Test loss: {test_loss:.6f}")

    test_data = analyzer.predict(test_loader)
    task_names = _get_task_names(test_data, task_names)

    metrics = analyzer.evaluate_all_metrics(test_data, task_names)
    out_path = Path(out_path)
    analyzer.to_csv(out_path, metrics)

    print(f"Saved: {out_path}")

    return test_loss, metrics


def run_selectivity_eval(
    data_path: PathLike,
    checkpoint_path: PathLike,
    *,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    batch_size: int = 32,
    out_path: PathLike = "selectivity_pairs.csv",
    task_names: Optional[Sequence[str]] = None,
):
    _, _, _, test_loader, analyzer = _build_test_loader_and_analyzer(
        data_path,
        checkpoint_path,
        batch_size=batch_size,
        num_workers=2,
    )

    test_data = analyzer.predict(test_loader)
    task_names = _get_task_names(test_data, task_names)

    if pairs is None:
        # default example pairs (only if those tasks exist)
        pairs = []
        if len(task_names) >= 2:
            pairs.append((task_names[0], task_names[1]))
        if len(task_names) >= 4:
            pairs.append((task_names[2], task_names[3]))

    sel = analyzer.evaluate_selectivity_pairs(test_data, list(pairs), task_names)
    analyzer.print_metrics(sel)

    out_path = Path(out_path)
    analyzer.to_csv(out_path, sel)
    print(f"Saved: {out_path}")

    return sel


def auto_analyze_testset(
    checkpoint_path: PathLike,
    data_path: PathLike,
    *,
    batch_size: int = 32,
    out_dir: PathLike = "eval_outputs",
    task_names: Optional[Sequence[str]] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, test_loader, analyzer = _build_test_loader_and_analyzer(
        data_path,
        checkpoint_path,
        batch_size=batch_size,
        num_workers=2,
    )

    test_loss = analyzer.evaluate_loss(test_loader)
    print(f"Test loss: {test_loss:.6f}")

    test_data = analyzer.predict(test_loader)
    task_names = _get_task_names(test_data, task_names)

    metrics = analyzer.evaluate_all_metrics(test_data, task_names)
    analyzer.to_csv(out_dir / "test_metrics.csv", metrics)
    print(f"Saved: {out_dir / 'test_metrics.csv'}")

    return test_loss, metrics

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Graphormer evaluation on test set")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data.pt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="eval_outputs", help="Output directory")

    args = parser.parse_args()

    auto_analyze_testset(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )
