from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Tuple
import pandas as pd

from config.config import get_config
from predict.predictor import GraphormerPredictor

app = FastAPI()

cfg = get_config()
predictor = GraphormerPredictor(
    checkpoint_path=".checkpoint_best.pt",
    cfg=cfg,
    device="cuda:0",              # or "cpu"
    cache_root="./api_cache",
    batch_size=64,
)

class PredictRequest(BaseModel):
    smiles: List[str]
    pairs: Optional[List[Tuple[str, str]]] = None  # optional selectivity pairs

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame({"smiles": req.smiles})
    if req.pairs:
        out = predictor.predict_selectivity(df, req.pairs).df
    else:
        out = predictor.predict_activities(df).df
    return out.to_dict(orient="records")
