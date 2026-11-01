"""Huấn luyện toàn bộ mô hình của Task 1-4 và lưu vào models/bundle.joblib.

Chạy:  python app/train_models.py
"""
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from features import (CLF_EXCLUDE, REG_EXCLUDE, build_matrix, make_schema,
                      prepare_training_frame)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "cgv_eda_cleaned.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
BUNDLE_PATH = MODEL_DIR / "bundle.joblib"
RANDOM_STATE = 42


def describe_cluster(sub: pd.DataFrame) -> str:
    top = sub["MovieName"].value_counts(normalize=True)
    name = top.index[0]
    short = name if len(name) <= 30 else name[:28] + "…"
    if top.iloc[0] >= 0.6:
        return f"{short} ({sub['Rating'].mode().iloc[0]})"
    return f"Nhiều phim – {sub['Rating'].mode().iloc[0]}"


def train() -> dict:
    t0 = time.time()
    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = prepare_training_frame(raw)
    schema = make_schema(df)
    X_full = build_matrix(df, schema)

    # ------------------------------------------------------------------ Task 1: PCA
    scaler = StandardScaler().fit(X_full)
    X_scaled = scaler.transform(X_full)
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_components = int(np.argmax(cumulative >= 0.90) + 1)            # giữ >= 90% phương sai
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # ------------------------------------------------------------------ Task 2: K-Means
    scan = []
    for k in range(2, 11):
        km_k = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_pca)
        scan.append({"K": k, "WCSS": km_k.inertia_,
                     "Silhouette": silhouette_score(X_pca, km_k.labels_)})
    scan = pd.DataFrame(scan)
    best_k = int(scan[scan["K"] >= 3].sort_values("Silhouette", ascending=False).iloc[0]["K"])
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_pca)
    labels = kmeans.labels_

    prof_rows = []
    for k in range(best_k):
        sub = df[labels == k]
        sub_x = X_full[labels == k]
        prof_rows.append({
            "Cụm": k,
            "Mô tả": describe_cluster(sub),
            "Số suất": len(sub),
            "Giá vé TB": round(float(sub["TicketPrice"].mean())),
            "Tỉ lệ lấp đầy TB": round(float(sub_x["OccupancyRate"].mean()), 3),
            "Thời lượng TB (phút)": round(float(sub["DurationMinutes"].mean()), 1),
            "% cuối tuần": round(float(sub_x["IsWeekend"].mean()) * 100, 1),
            "% phòng đặc biệt": round(float(sub_x["IsPremiumRoom"].mean()) * 100, 1),
            "Phim chính": "; ".join(f"{i} ({v:.0%})" for i, v in
                                    sub["MovieName"].value_counts(normalize=True).head(2).items()),
        })
    cluster_profile = pd.DataFrame(prof_rows).set_index("Cụm")

    viz = pd.DataFrame(X_pca[:, :3], columns=["PC1", "PC2", "PC3"])
    viz["Cluster"] = labels
    viz["MovieName"] = df["MovieName"].values
    viz["Cinema"] = df["Cinema"].values
    viz["ShowTime"] = df["ShowTime"].values

    # ------------------------------------------------------------------ Task 3: hồi quy
    revenue = df["TicketPrice"] * (df["TotalSeats"] - df["AvailableSeats"])
    X_reg = X_full.drop(columns=REG_EXCLUDE)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, revenue, test_size=0.2,
                                                  random_state=RANDOM_STATE)
    reg_models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()).fit(Xr_tr, yr_tr),
    }
    grid_r = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {"max_depth": [2, 3, 4, 5, 6, 8, 10], "min_samples_leaf": [5, 10, 20, 40]},
        cv=5, scoring="r2").fit(Xr_tr, yr_tr)
    reg_models["Decision Tree"] = grid_r.best_estimator_

    reg_rows, reg_pred_test = [], {}
    for name, mdl in reg_models.items():
        pred = mdl.predict(Xr_te)
        mse = mean_squared_error(yr_te, pred)
        reg_rows.append({
            "Mô hình": name, "MSE (test)": mse, "RMSE (test)": float(np.sqrt(mse)),
            "R² (test)": r2_score(yr_te, pred),
            "R² (train)": r2_score(yr_tr, mdl.predict(Xr_tr)),
            "R² (CV 5-fold)": cross_val_score(mdl, X_reg, revenue, cv=5, scoring="r2").mean(),
        })
        reg_pred_test[name] = pred
    reg_metrics = pd.DataFrame(reg_rows).set_index("Mô hình")

    # ------------------------------------------------------------------ Task 4: phân loại
    occupancy = X_full["OccupancyRate"]
    threshold = float(occupancy.median())
    y_clf = pd.Series(np.where(occupancy > threshold, "Cao", "Không cao"), index=df.index)
    X_clf = X_full.drop(columns=CLF_EXCLUDE)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clf, y_clf, test_size=0.2,
                                                  random_state=RANDOM_STATE, stratify=y_clf)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        ).fit(Xc_tr, yc_tr),
    }
    grid_c = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [5, 10, 20, 40]},
        scoring="f1_macro", cv=cv).fit(Xc_tr, yc_tr)
    clf_models["Decision Tree"] = grid_c.best_estimator_

    labels_order = ["Không cao", "Cao"]
    clf_rows, confusion = [], {}
    for name, mdl in clf_models.items():
        pred = mdl.predict(Xc_te)
        clf_rows.append({
            "Mô hình": name,
            "Accuracy": accuracy_score(yc_te, pred),
            "Precision (Cao)": precision_score(yc_te, pred, pos_label="Cao", zero_division=0),
            "Recall (Cao)": recall_score(yc_te, pred, pos_label="Cao", zero_division=0),
            "F1 (Cao)": f1_score(yc_te, pred, pos_label="Cao", zero_division=0),
            "F1 macro": f1_score(yc_te, pred, average="macro"),
            "F1 macro (CV)": cross_val_score(mdl, X_clf, y_clf, cv=cv, scoring="f1_macro").mean(),
        })
        confusion[name] = confusion_matrix(yc_te, pred, labels=labels_order)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xc_tr, yc_tr)
    dpred = dummy.predict(Xc_te)
    clf_rows.insert(0, {
        "Mô hình": "Cơ sở (luôn đoán lớp đa số)",
        "Accuracy": accuracy_score(yc_te, dpred), "Precision (Cao)": 0.0, "Recall (Cao)": 0.0,
        "F1 (Cao)": 0.0, "F1 macro": f1_score(yc_te, dpred, average="macro"),
        "F1 macro (CV)": cross_val_score(DummyClassifier(strategy="most_frequent"),
                                         X_clf, y_clf, cv=cv, scoring="f1_macro").mean(),
    })
    clf_metrics = pd.DataFrame(clf_rows).set_index("Mô hình")

    # ------------------------------------------------------------------ thông tin cho UI
    movies = (df.groupby("MovieName")
                .agg(Rating=("Rating", "first"), Genre=("Genre", "first"),
                     DurationMinutes=("DurationMinutes", "first"),
                     ReleaseDate=("ReleaseDate", "first"), So_suat=("MovieName", "size"))
                .sort_values("So_suat", ascending=False))
    options = {
        "cinemas": schema["Cinema"], "ratings": schema["Rating"],
        "format_techs": schema["FormatTech"], "genres": schema["Genre"],
        "total_seats": sorted(int(v) for v in df["TotalSeats"].unique()),
        "price_min": int(df["TicketPrice"].min()), "price_max": int(df["TicketPrice"].max()),
        "price_median": int(df["TicketPrice"].median()),
        "date_min": str(df["ShowDate"].min()), "date_max": str(df["ShowDate"].max()),
    }

    bundle = {
        "schema": schema, "scaler": scaler, "pca": pca, "kmeans": kmeans,
        "n_components": n_components, "k": best_k,
        "explained": explained, "cumulative": cumulative, "kmeans_scan": scan,
        "cluster_profile": cluster_profile, "viz": viz,
        "reg_models": reg_models, "reg_metrics": reg_metrics,
        "reg_cols": list(X_reg.columns), "reg_test": (yr_te.values, reg_pred_test),
        "clf_models": clf_models, "clf_metrics": clf_metrics,
        "clf_cols": list(X_clf.columns), "confusion": confusion,
        "occupancy_threshold": threshold,
        "movies": movies, "options": options,
        "best_params": {"reg": grid_r.best_params_, "clf": grid_c.best_params_},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    print(f"Đã huấn luyện và lưu {BUNDLE_PATH} ({time.time() - t0:.1f}s)")
    print(f"PCA: {n_components} thành phần | K-Means: K={best_k} | "
          f"ngưỡng nhu cầu: {threshold:.0%}")
    return bundle


def load_or_train() -> dict:
    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)
    return train()


if __name__ == "__main__":
    train()
