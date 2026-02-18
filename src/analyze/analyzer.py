# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from config.config import get_config
from models.graphormer import GraphormerModel

from data.dataset.pyg_dataset import PYGDataset
from data.data_wrapper import graphormer_collate, make_preprocess_item

from train.train_reg import enforce_graphormer_batch, sanitize_labels
from loss.Losses import loss_regression_plus_selectivity


@dataclass
class CollectedData:
    true_y: torch.Tensor   # (N, T) CPU
    pred_y: torch.Tensor   # (N, T) CPU
    mask: torch.Tensor     # (N, T) bool CPU


@dataclass
class GraphormerAnalyzer:
    model: "GraphormerModel"
    cfg: Any
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        cfg: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "GraphormerAnalyzer":
        checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if cfg is None:
            cfg = get_config()

        model = GraphormerModel.build_model(cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return cls(model=model, cfg=cfg, device=device)

    @staticmethod
    @torch.no_grad()
    def collect_y_from_loader(loader, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_y = []
        all_mask = []

        for batch in loader:
            batch = enforce_graphormer_batch(batch, device)

            y, mask = sanitize_labels(batch["y"])
            y = y.view(y.size(0), -1)
            mask = mask.view(mask.size(0), -1).bool()

            all_y.append(y.detach().cpu())
            all_mask.append(mask.detach().cpu())

        y_all = torch.cat(all_y, dim=0)
        mask_all = torch.cat(all_mask, dim=0)
        return y_all, mask_all

    @torch.no_grad()
    def predict(self, data_loader) -> CollectedData:
        self.model.eval()

        all_pred, all_y, all_mask = [], [], []

        for batch in data_loader:
            batch = enforce_graphormer_batch(batch, self.device)

            y, mask = sanitize_labels(batch["y"])
            out = self.model(batch)

            pred = out[:, 0, :] if out.dim() == 3 else out  # (B,T)

            y = y.view(y.size(0), -1)
            mask = mask.view(mask.size(0), -1).bool()

            all_pred.append(pred.detach().cpu())
            all_y.append(y.detach().cpu())
            all_mask.append(mask.detach().cpu())

        pred_all = torch.cat(all_pred, dim=0)
        y_all = torch.cat(all_y, dim=0)
        mask_all = torch.cat(all_mask, dim=0)

        return CollectedData(true_y=y_all, pred_y=pred_all, mask=mask_all)

    @staticmethod
    def collect_data(
        pred_y: Union[torch.Tensor, np.ndarray, list],
        true_y: Union[torch.Tensor, np.ndarray, list],
        mask: Union[torch.Tensor, np.ndarray, list],
    ) -> CollectedData:
        def to_tensor(x: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            if isinstance(x, list):
                return torch.tensor(x)
            raise TypeError(f"Unsupported type: {type(x)}")

        pred_t = to_tensor(pred_y).float()
        true_t = to_tensor(true_y).float()
        mask_t = to_tensor(mask).bool()

        pred_t = pred_t.view(pred_t.size(0), -1) if pred_t.dim() > 1 else pred_t.unsqueeze(1)
        true_t = true_t.view(true_t.size(0), -1) if true_t.dim() > 1 else true_t.unsqueeze(1)
        mask_t = mask_t.view(mask_t.size(0), -1) if mask_t.dim() > 1 else mask_t.unsqueeze(1)

        return CollectedData(true_y=true_t, pred_y=pred_t, mask=mask_t)

    @torch.no_grad()
    def evaluate_loss(
        self,
        data_loader,
        loss_fn=loss_regression_plus_selectivity,
    ) -> float:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in data_loader:
            batch = enforce_graphormer_batch(batch, self.device)

            y, mask = sanitize_labels(batch["y"])
            y = y.view(y.size(0), -1)
            mask = mask.view(mask.size(0), -1).bool()

            out = self.model(batch)
            pred = out[:, 0, :] if out.dim() == 3 else out  # (B,T)

            loss = loss_fn(pred, y, mask)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    @staticmethod
    def evaluate_all_metrics(data: CollectedData, task_names: Union[list[str], tuple[str, ...]]) -> Dict[str, Dict[str, float]]:
        y, pred, mask = data.true_y, data.pred_y, data.mask
        results: Dict[str, Dict[str, float]] = {}

        T = pred.size(1)
        for i in range(T):
            name = task_names[i] if i < len(task_names) else f"task{i}"
            valid = mask[:, i]
            n_valid = int(valid.sum().item())

            if n_valid == 0:
                results[name] = {"N": 0.0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan, "Spearman": np.nan}
                continue

            y_i = y[valid, i].detach().cpu().numpy()
            p_i = pred[valid, i].detach().cpu().numpy()

            mae = float(np.mean(np.abs(p_i - y_i)))
            rmse = float(np.sqrt(np.mean((p_i - y_i) ** 2)))

            if n_valid < 2 or np.std(y_i) < 1e-12:
                r2 = np.nan
                pearson = np.nan
                spearman = np.nan
            else:
                r2 = float(r2_score(y_i, p_i))
                pearson = float(pearsonr(y_i, p_i)[0])
                spearman = float(spearmanr(y_i, p_i)[0])

            results[name] = {"N": float(n_valid), "MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pearson, "Spearman": spearman}

        return results

    @staticmethod
    def evaluate_selectivity_pairs(
        data: CollectedData,
        pairs: Union[list[Tuple[str, str]], tuple[Tuple[str, str], ...]],
        task_names: Union[list[str], tuple[str, ...]],
    ) -> Dict[str, Dict[str, float]]:
        name_to_idx = {n: i for i, n in enumerate(task_names)}
        y, pred, mask = data.true_y, data.pred_y, data.mask

        results: Dict[str, Dict[str, float]] = {}

        for a, b in pairs:
            if a not in name_to_idx or b not in name_to_idx:
                raise ValueError(f"Unknown task in pair {(a, b)}. Known: {list(name_to_idx)}")

            ia, ib = name_to_idx[a], name_to_idx[b]
            valid = mask[:, ia] & mask[:, ib]
            n_valid = int(valid.sum().item())

            if n_valid == 0:
                results[f"{a}-{b}"] = {
                    "N": 0.0,
                    "DeltaMeanTrue": np.nan,
                    "DeltaStdTrue": np.nan,
                    "DeltaVarTrue": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Pearson": np.nan,
                    "Spearman": np.nan,
                }
                continue

            ya = y[valid, ia].detach().cpu().numpy()
            yb = y[valid, ib].detach().cpu().numpy()
            pa = pred[valid, ia].detach().cpu().numpy()
            pb = pred[valid, ib].detach().cpu().numpy()

            y_delta = ya - yb
            p_delta = pa - pb

            delta_mean_true = float(np.mean(y_delta))
            delta_std_true = float(np.std(y_delta))
            delta_var_true = float(np.var(y_delta))

            err = p_delta - y_delta
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))

            if n_valid < 2 or delta_std_true < 1e-12:
                r2 = np.nan
                pear = np.nan
                spear = np.nan
            else:
                r2 = float(r2_score(y_delta, p_delta))
                pear = float(pearsonr(y_delta, p_delta)[0])
                spear = float(spearmanr(y_delta, p_delta)[0])

            results[f"{a}-{b}"] = {
                "N": float(n_valid),
                "DeltaMeanTrue": delta_mean_true,
                "DeltaStdTrue": delta_std_true,
                "DeltaVarTrue": delta_var_true,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Pearson": pear,
                "Spearman": spear,
            }

        return results

    @staticmethod
    def print_metrics(results: Dict[str, Dict[str, float]]) -> None:
        for task, v in results.items():
            print(f"\n{task}")
            for metric_name, value in v.items():
                try:
                    val = float(value)
                except Exception:
                    val = np.nan
                print(f"{metric_name:12s}: {val:.4f}")

    @staticmethod
    def to_csv(path: Union[Path, str], results: Dict[str, Dict[str, float]]) -> None:
        metrics = set()
        for v in results.values():
            metrics.update(v.keys())

        metrics = sorted(metrics)
        data: Dict[str, list] = {"Metrics": metrics}

        for task, v in results.items():
            data[task] = [float(v.get(m, float("nan"))) for m in metrics]

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)


def main():
    cfg = get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- dataset / loaders ----
    data_path = Path("./data.pt")
    root = data_path.parent.parent

    ds = PYGDataset.load_all(
        str(data_path),
        transform=make_preprocess_item(cfg.multi_hop_max_dist),
        root=str(root),
    )

    test_loader = DataLoader(
        ds.test,
        batch_size=16,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=graphormer_collate,
    )

    # ---- analyzer ----
    # TODO: set your checkpoint path here
    checkpoint_path = Path("YOUR_CHECKPOINT.pt")
    analyzer = GraphormerAnalyzer.from_checkpoint(checkpoint_path, cfg=cfg, device=device)

    # ---- task names ----
    # If your dataset/config provides task names, use that. Fallback to generic.
    # You can replace this with: task_names = ds.task_names  (if it exists)
    # or cfg.task_names, etc.
    num_tasks_guess = getattr(cfg, "num_tasks", None)
    if num_tasks_guess is None:
        # best-effort: read one batch to infer T from y
        y0, m0 = GraphormerAnalyzer.collect_y_from_loader(test_loader, device=device)
        T = y0.size(1)
    else:
        T = int(num_tasks_guess)
    task_names = [f"task{i}" for i in range(T)]

    # ---- evaluate loss ----
    test_loss = analyzer.evaluate_loss(test_loader)
    print(f"\nTest loss: {test_loss:.6f}")

    # ---- predictions + metrics ----
    test_data = analyzer.predict(test_loader)
    metrics = analyzer.evaluate_all_metrics(test_data, task_names)
    analyzer.print_metrics(metrics)
    analyzer.to_csv("metrics_test.csv", metrics)

    # ---- optional: selectivity pairs ----
    # Example: evaluate pairs for first 2 tasks if you want
    # pairs = [("task0", "task1")]
    # sel = analyzer.evaluate_selectivity_pairs(test_data, pairs, task_names)
    # analyzer.print_metrics(sel)
    # analyzer.to_csv("selectivity_pairs.csv", sel)


if __name__ == "__main__":
    main()
