import argparse, os, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from lightfm import LightFM
import joblib

def build_interactions(df, user_col, item_col, rating_col):
    users = pd.Categorical(df[user_col])
    items = pd.Categorical(df[item_col])
    mat = coo_matrix((df[rating_col].astype(float), (users.codes, items.codes)))
    return mat, users, items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_csv", default="data/processed/interactions.csv")  # user,item,rating
    ap.add_argument("--user_features_csv", default="data/processed/user_features.csv") # user, f1,f2,...
    ap.add_argument("--out", default="models/recommender")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ui = pd.read_csv(args.interactions_csv)
    mat, users, items = build_interactions(ui, "user", "item", "rating")

    # LightFM CF
    model = LightFM(loss="warp")
    model.fit(mat, epochs=10, num_threads=4)

    # User clustering
    uf = pd.read_csv(args.user_features_csv).set_index("user")
    kmeans = KMeans(n_clusters=5, random_state=42).fit(uf.values)

    joblib.dump({
        "lightfm": model,
        "user_codes": dict(zip(users.categories, range(len(users.categories)))),
        "item_codes": dict(zip(items.categories, range(len(items.categories)))),
        "kmeans": kmeans,
        "user_feature_columns": uf.columns.tolist()
    }, os.path.join(args.out, "hybrid.pkl"))
    print("✅ Saved:", os.path.join(args.out, "hybrid.pkl"))

if __name__ == "__main__":
    main()
