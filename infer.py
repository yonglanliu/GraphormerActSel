import argparse
import torch
import pandas as pd
import numpy as np
from rdkit import Chem

# -------------------------
# Utilities
# -------------------------

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def compute_selectivity(pred_dict):
    """
    Compute pairwise selectivity:
    S(i,j) = pIC50_i - pIC50_j
    """
    isoforms = list(pred_dict.keys())
    selectivity = {}
    for i in range(len(isoforms)):
        for j in range(i + 1, len(isoforms)):
            key = f"{isoforms[i]}_vs_{isoforms[j]}"
            selectivity[key] = (
                pred_dict[isoforms[i]] - pred_dict[isoforms[j]]
            )
    return selectivity


# -------------------------
# Model Loading
# -------------------------

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Replace this with your actual model class
    from src.model import GraphormerModel  # adjust if needed

    model = GraphormerModel(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model


# -------------------------
# Featurization
# -------------------------

def featurize(smiles):
    """
    Convert SMILES into model input.
    Replace this with your graph construction pipeline.
    """
    # Example placeholder
    return {"smiles": smiles}


# -------------------------
# Prediction
# -------------------------

@torch.no_grad()
def predict(model, smiles, device):
    smiles = canonicalize_smiles(smiles)
    inputs = featurize(smiles)

    # Replace this with your forward logic
    outputs = model(inputs)

    # Assume outputs is a dict: {isoform_name: value}
    pred_dict = {k: float(v) for k, v in outputs.items()}

    selectivity = compute_selectivity(pred_dict)

    return pred_dict, selectivity


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smiles", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    # Single SMILES mode
    if args.smiles:
        preds, sels = predict(model, args.smiles, device)

        print("\nPredicted Activity:")
        for k, v in preds.items():
            print(f"  {k}: {v:.3f}")

        print("\nPredicted Selectivity:")
        for k, v in sels.items():
            print(f"  {k}: {v:.3f}")

    # Batch mode
    elif args.input:
        df = pd.read_csv(args.input)
        results = []

        for smi in df["smiles"]:
            preds, sels = predict(model, smi, device)
            row = {"smiles": smi}
            row.update(preds)
            row.update(sels)
            results.append(row)

        out_df = pd.DataFrame(results)

        if args.output:
            out_df.to_csv(args.output, index=False)
            print(f"Saved predictions to {args.output}")
        else:
            print(out_df.head())

    else:
        raise ValueError("Provide either --smiles or --input")


if __name__ == "__main__":
    main()
