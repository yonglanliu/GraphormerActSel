import streamlit as st
import pandas as pd
import torch

from config.config import get_config
from predict.predictor import GraphormerPredictor
import itertools

# ---------- Page Config ----------
st.set_page_config(page_title="Predictor", layout="wide")
st.title("Protein Activity & Selectivity Predictor")

# ---------- Load Model (cached) ----------
@st.cache_resource
def load_model():
    cfg = get_config()
    predictor = GraphormerPredictor(
        checkpoint_path="./checkpoint_best.pt",
        cfg=cfg,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        cache_root="./streamlit_cache",
        task_names=["ISOA", "ISOB", "ISOC"],  # adjust
    )
    return predictor

predictor = load_model()

# ---------- Input ----------
smiles_input = st.text_area(
    "Enter SMILES (one per line)",
    height=150,
)

predict_button = st.button("Predict")

if predict_button:
    if not smiles_input.strip():
        st.warning("Please enter at least one SMILES.")
    else:
        smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
        df = pd.DataFrame({"smiles": smiles_list})

        with st.spinner("Running prediction..."):
            result = predictor.predict_activities(df)

        st.success("Prediction complete!")

        # ==============================
        # Activity Predictions
        # ==============================
        st.subheader("Activity Predictions")
        st.dataframe(result.df)

        # ==============================
        # All Pairwise Selectivity
        # ==============================
        st.subheader("All Pairwise Selectivity (Δ = A − B)")

        task_names = ["ISOA", "ISOB", "ISOC"]
        pred_df = result.df.copy()

        # generate all unique pairs
        pairs = list(itertools.combinations(task_names, 2))

        for a, b in pairs:
            col_a = f"pred_{a}"
            col_b = f"pred_{b}"

            if col_a in pred_df.columns and col_b in pred_df.columns:
                pred_df[f"delta_{a}_minus_{b}"] = pred_df[col_a] - pred_df[col_b]

        # Show only selectivity columns
        selectivity_cols = ["smiles"] + [
            c for c in pred_df.columns if c.startswith("delta_")
        ]

        st.dataframe(pred_df[selectivity_cols])
