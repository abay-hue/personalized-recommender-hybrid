# Hybrid Recommender (K-Means + CF)

K-Means segments + LightFM collaborative filtering with FastAPI/Streamlit demo.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Structure
```
src/        # core code
app/        # UI/API (if any)
notebooks/  # EDA & experiments
data/       # raw/processed (gitignored)
models/     # checkpoints (gitignored)
reports/    # metrics & figures
configs/    # yaml/json configs
```
