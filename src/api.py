import os, numpy as np, pandas as pd, joblib
from fastapi import FastAPI
from pydantic import BaseModel

PACK = joblib.load(os.environ.get("HYBRID_PACK", "models/recommender/hybrid.pkl"))
app = FastAPI(title="Hybrid Recommender API")

class RecIn(BaseModel):
    user: str
    topk: int = 5

@app.post("/recommend")
def recommend(inp: RecIn):
    ucode = PACK["user_codes"].get(inp.user)
    if ucode is None:
        return {"items": []}
    model = PACK["lightfm"]
    n_items = len(PACK["item_codes"])
    scores = model.predict(ucode, np.arange(n_items))
    topk_idx = np.argsort(-scores)[:inp.topk]
    inv_items = {v:k for k,v in PACK["item_codes"].items()}
    items = [inv_items[i] for i in topk_idx]
    return {"items": items, "scores": [float(scores[i]) for i in topk_idx]}
