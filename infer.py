#!/usr/bin/env python3
"""
infer.py — Graphormer multi-isoform inference with optional MC Dropout uncertainty.

Features:
- Single SMILES inference:  --smiles "..."
- Batch inference from CSV: --input in.csv --output out.csv
- GPU auto-detection
- Optional MC Dropout:      --mc 30  (returns mean ± std)
- Pairwise selectivity:     S(i,j) = pIC50_i - pIC50_j
- Selectivity uncertainty:  computed from MC samples (mean ± std)

You MUST adapt two parts to your codebase:
1) load_model(): how to construct your model + load checkpoint
2) featurize_smiles(): how to convert SMILES -> model inputs

Recommended CSV input schema:
- Column: smiles
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from rdkit import Chem
except Exception as e:
    Chem = None


# ----------------------------
# Chemistry helpers
# ----------------------------

def canonicalize_smiles(smiles: str) -> str:
    if Chem is None:
        raise ImportError("RDKit not available. Install rdkit (conda-forge recommended) to canonicalize SMILES.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


# ----------------------------
# Selectivity helpers
# ----------------------------

def compute_pairwise_selectivity(values: Dict[str, float], prefix: str = "sel") -> Dict[str, float]:
    """
    Given per-isoform predictions, compute pairwise selectivity:
      sel_isoA_isoB = pIC50_isoA - pIC50_isoB
    """
    isoforms = list(values.keys())
    out: Dict[str, float] = {}
    for i in range(len(isoforms)):
        for j in range(i + 1, len(isoforms)):
            a, b = isoforms[i], isoforms[j]
            out[f"{prefix}_{a}_{b}"] = values[a] - values[b]
    return out


def compute_selectivity_from_samples(samples_df: pd.DataFrame, prefix: str = "sel") -> pd.DataFrame:
    """
    samples_df: rows = MC samples, columns = isoform predictions
    returns a DF with pairwise selectivity columns (same number of rows).
    """
    isoforms = list(samples_df.columns)
    cols = {}
    for i in range(len(isoforms)):
        for j in range(i + 1, len(isoforms)):
            a, b = isoforms[i], isoforms[j]
            cols[f"{prefix}_{a}_{b}"] = samples_df[a] - samples_df[b]
    return pd.DataFrame(cols)


# ----------------------------
# MC Dropout helpers
# ----------------------------

def enable_dropout(model: torch.nn.Module) -> None:
    """
    Enable dropout layers during inference for Monte Carlo sampling.
    This sets only Dropout modules to train mode, leaving everything else in eval mode.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


# ----------------------------
# Model + featurization (YOU EDIT)
# ----------------------------

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model weights.

    You MUST modify this to match your checkpoint + model code.

    Expected checkpoint examples:
    - {"state_dict": ..., "model_args": {...}, "isoform_names": [...]}
    - or a raw state_dict directly

    Replace `from src.model import GraphormerModel` with your actual import.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # ---- Example pattern A: dict checkpoint ----
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        model_args = ckpt.get("model_args", {})

        # TODO: replace with your actual model class
        try:
            from src.model import GraphormerModel  # <-- CHANGE THIS
        except Exception as e:
            raise ImportError(
                "Could not import your model. Edit load_model() to import the correct class.\n"
                f"Import error: {e}"
            )

        model = GraphormerModel(**model_args)
        model.load_state_dict(state_dict, strict=True)
    # ---- Example pattern B: raw state_dict ----
    elif isinstance(ckpt, dict):
        # TODO: replace with your actual model class + args if needed
        try:
            from src.model import GraphormerModel  # <-- CHANGE THIS
        except Exception as e:
            raise ImportError(
                "Checkpoint looks like a raw state_dict. You still need to construct the model class.\n"
                f"Import error: {e}"
            )
        model = GraphormerModel()
        model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError("Unsupported checkpoint format. Edit load_model() to match your checkpoint.")

    model.to(device)
    model.eval()
    return model


def featurize_smiles(smiles: str) -> Dict[str, Any]:
    """
    Convert SMILES -> model input.

    You MUST modify this to match your Graphormer data pipeline.
    Common outputs might include:
      - atom features, edge index, edge features, shortest path distances, etc.

    For now, this is a placeholder.
    """
    # TODO: Replace with your actual graph construction / dataset code
    return {"smiles": smiles}


def model_forward(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Run model forward pass and return a dict of isoform->prediction tensors (shape []).
    You MUST adapt this to your model's forward output format.

    Assumptions (edit as needed):
    - model(batch) returns a dict: {isoform_name: tensor_scalar}
      OR returns a tensor of shape [K] plus you have isoform names somewhere.
    """
    # Move tensors to device if your batch contains tensors
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    out = model(batch)

    # Case 1: dict output
    if isinstance(out, dict):
        return out

    # Case 2: tensor output (needs isoform names)
    if torch.is_tensor(out):
        # TODO: set your isoform names in code/config/checkpoint
        raise ValueError(
            "Model returned a tensor, but isoform names are unknown. "
            "Edit model_forward() to map tensor indices to isoform names."
        )

    raise ValueError(f"Unsupported model output type: {type(out)}")


# ----------------------------
# Prediction API
# ----------------------------

@torch.no_grad()
def predict_once(
    model: torch.nn.Module,
    smiles: str,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    smi = canonicalize_smiles(smiles)
    batch = featurize_smiles(smi)
    out = model_forward(model, batch, device)

    preds = {k: float(v.detach().cpu().item()) for k, v in out.items()}
    sels = compute_pairwise_selectivity(preds, prefix="sel")
    return preds, sels


@torch.no_grad()
def predict_mc(
    model: torch.nn.Module,
    smiles: str,
    device: torch.device,
    mc_samples: int = 30,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Returns:
      activity_mean, activity_std, selectivity_mean, selectivity_std
    """
    if mc_samples < 2:
        raise ValueError("--mc must be >= 2 to compute std reliably.")

    smi = canonicalize_smiles(smiles)
    batch = featurize_smiles(smi)

    # Keep model in eval, enable only Dropout layers
    model.eval()
    enable_dropout(model)

    rows: List[Dict[str, float]] = []

    for _ in range(mc_samples):
        out = model_forward(model, batch, device)
        rows.append({k: float(v.detach().cpu().item()) for k, v in out.items()})

    samples_df = pd.DataFrame(rows)  # cols = isoforms
    act_mean = samples_df.mean(axis=0).to_dict()
    act_std = samples_df.std(axis=0, ddof=1).to_dict()

    sel_samples_df = compute_selectivity_from_samples(samples_df, prefix="sel")
    sel_mean = sel_samples_df.mean(axis=0).to_dict()
    sel_std = sel_samples_df.std(axis=0, ddof=1).to_dict()

    return act_mean, act_std, sel_mean, sel_std


# ----------------------------
# CLI + IO
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graphormer multi-isoform inference (with optional MC Dropout).")

    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smiles", type=str, help="Single SMILES string.")
    grp.add_argument("--input", type=str, help="CSV with a 'smiles' column for batch inference.")

    p.add_argument("--output", type=str, default=None, help="Output CSV for batch mode.")
    p.add_argument("--mc", type=int, default=0, help="MC Dropout samples. 0 disables MC Dropout.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    p.add_argument("--json", action="store_true", help="Print single-SMILES outputs as JSON.")

    return p.parse_args()


def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single(model: torch.nn.Module, smiles: str, device: torch.device, mc: int, as_json: bool) -> None:
    if mc and mc > 0:
        act_mean, act_std, sel_mean, sel_std = predict_mc(model, smiles, device, mc_samples=mc)

        payload = {
            "smiles": smiles,
            "device": str(device),
            "mc_samples": mc,
            "activity_mean": act_mean,
            "activity_std": act_std,
            "selectivity_mean": sel_mean,
            "selectivity_std": sel_std,
        }

        if as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return

        print(f"Using device: {device}")
        print(f"MC Dropout samples: {mc}")

        print("\nPredicted Activity (mean ± std):")
        for k in sorted(act_mean.keys()):
            print(f"  {k}: {act_mean[k]:.3f} ± {act_std[k]:.3f}")

        print("\nPredicted Selectivity (mean ± std):")
        for k in sorted(sel_mean.keys()):
            print(f"  {k}: {sel_mean[k]:+.3f} ± {sel_std[k]:.3f}")

    else:
        preds, sels = predict_once(model, smiles, device)

        payload = {
            "smiles": smiles,
            "device": str(device),
            "activity": preds,
            "selectivity": sels,
        }

        if as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return

        print(f"Using device: {device}")
        print("\nPredicted Activity:")
        for k in sorted(preds.keys()):
            print(f"  {k}: {preds[k]:.3f}")

        print("\nPredicted Selectivity:")
        for k in sorted(sels.keys()):
            print(f"  {k}: {sels[k]:+.3f}")


def run_batch(
    model: torch.nn.Module,
    input_csv: str,
    output_csv: Optional[str],
    device: torch.device,
    mc: int,
) -> None:
    df = pd.read_csv(input_csv)
    if "smiles" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'smiles' column. Found: {list(df.columns)}")

    records: List[Dict[str, Any]] = []
    n = len(df)

    for idx, raw_smi in enumerate(df["smiles"].astype(str).tolist(), start=1):
        try:
            if mc and mc > 0:
                act_mean, act_std, sel_mean, sel_std = predict_mc(model, raw_smi, device, mc_samples=mc)

                row: Dict[str, Any] = {"smiles": raw_smi}
                # activity mean/std
                for k, v in act_mean.items():
                    row[f"pIC50_{k}_mean"] = v
                for k, v in act_std.items():
                    row[f"pIC50_{k}_std"] = v
                # selectivity mean/std
                for k, v in sel_mean.items():
                    row[f"{k}_mean"] = v
                for k, v in sel_std.items():
                    row[f"{k}_std"] = v

            else:
                preds, sels = predict_once(model, raw_smi, device)
                row = {"smiles": raw_smi}
                for k, v in preds.items():
                    row[f"pIC50_{k}"] = v
                row.update(sels)

            records.append(row)

        except Exception as e:
            # Keep going; record error for that SMILES
            records.append({"smiles": raw_smi, "error": str(e)})

        if idx % 50 == 0 or idx == n:
            print(f"[{idx}/{n}] processed", file=sys.stderr)

    out_df = pd.DataFrame(records)

    if output_csv:
        out_df.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")
    else:
        print(out_df.head(10).to_string(index=False))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model = load_model(args.checkpoint, device)

    if args.smiles:
        run_single(model, args.smiles, device, mc=args.mc, as_json=args.json)
    else:
        run_batch(model, args.input, args.output, device, mc=args.mc)


if __name__ == "__main__":
    main()
