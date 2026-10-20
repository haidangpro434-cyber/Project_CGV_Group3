from __future__ import annotations

from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
NUMERIC_FEATURES = (
    "DurationMinutes",
    "ShowMonth",
    "ShowDayOfWeek",
    "ShowHour",
    "ShowMinute",
    "ReleaseMonth",
    "DaysFromReleaseToShow",
)
CATEGORICAL_FEATURES = ("Cinema", "Format", "Rating", "Genre", "Language")


def create_models() -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def load_resources() -> dict[str, object]:
    pca_path = DATA_DIR / "cgv_pca_transformed.csv"
    labels_path = DATA_DIR / "cgv_clustered_suatchieu.csv"
    raw_data_path = DATA_DIR / "cgv_eda_cleaned.csv"
    scaler_path = DATA_DIR / "scaler_model.joblib"
    pca_model_path = DATA_DIR / "pca_model.joblib"
    required_paths = (
        pca_path,
        labels_path,
        raw_data_path,
        scaler_path,
        pca_model_path,
    )
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Thiếu dữ liệu hoặc mô hình trong Stage2: " + ", ".join(missing_paths)
        )

    features = pd.read_csv(pca_path)
    clustered_data = pd.read_csv(labels_path)
    raw_data = pd.read_csv(raw_data_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_model_path)

    if "Cluster_Label" not in clustered_data:
        raise ValueError("cgv_clustered_suatchieu.csv thiếu cột Cluster_Label.")
    if len(features) != len(clustered_data):
        raise ValueError(
            "Số dòng dữ liệu PCA và nhãn cụm không khớp; "
            "hãy chạy lại Task 1 và Task 2."
        )
    if features.isna().any().any() or clustered_data["Cluster_Label"].isna().any():
        raise ValueError("Dữ liệu huấn luyện còn giá trị thiếu.")
    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError(
            "Scaler không chứa tên đặc trưng; hãy chạy lại Task 1 để lưu mô hình."
        )

    encoded_columns = list(scaler.feature_names_in_)
    if len(encoded_columns) != scaler.n_features_in_:
        raise ValueError("Số đặc trưng của scaler không hợp lệ.")
    if features.shape[1] != pca.n_components_:
        raise ValueError(
            "Số thành phần PCA của Task 4 không khớp với mô hình PCA đã lưu."
        )
    expected_components = [
        f"PC{index}" for index in range(1, pca.n_components_ + 1)
    ]
    if list(features.columns) != expected_components:
        raise ValueError("Tên/cột thành phần PCA trong dữ liệu Task 4 không hợp lệ.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        clustered_data["Cluster_Label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=clustered_data["Cluster_Label"],
    )
    metrics: dict[str, dict[str, float]] = {}
    models = create_models()
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics[name] = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "F1-score (Macro)": float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            ),
        }
        model.fit(features, clustered_data["Cluster_Label"])

    return {
        "features": features,
        "labels": clustered_data["Cluster_Label"],
        "raw_data": raw_data,
        "clustered_data": clustered_data,
        "encoded_columns": encoded_columns,
        "scaler": scaler,
        "pca": pca,
        "models": models,
        "metrics": metrics,
    }


def build_encoded_input(
    *,
    duration_minutes: float,
    show_date: date,
    show_time: time,
    release_date: date,
    categories: dict[str, str],
    encoded_columns: list[str],
) -> pd.DataFrame:
    if duration_minutes <= 0:
        raise ValueError("Thời lượng phim phải lớn hơn 0 phút.")
    missing_categories = set(CATEGORICAL_FEATURES) - categories.keys()
    if missing_categories:
        raise ValueError(
            "Thiếu thuộc tính phân loại: " + ", ".join(sorted(missing_categories))
        )

    numeric_values = {
        "DurationMinutes": float(duration_minutes),
        "ShowMonth": float(show_date.month),
        "ShowDayOfWeek": float(show_date.weekday()),
        "ShowHour": float(show_time.hour),
        "ShowMinute": float(show_time.minute),
        "ReleaseMonth": float(release_date.month),
        "DaysFromReleaseToShow": float((show_date - release_date).days),
    }
    categorical_values = {
        f"{name}_{categories[name]}": 1
        for name in CATEGORICAL_FEATURES
    }
    encoded_values = {
        column: numeric_values.get(column, categorical_values.get(column, 0))
        for column in encoded_columns
    }
    return pd.DataFrame([encoded_values], columns=encoded_columns, dtype=float)


def transform_input(
    encoded_input: pd.DataFrame,
    scaler: StandardScaler,
    pca: object,
) -> pd.DataFrame:
    scaled_input = scaler.transform(encoded_input)
    pca_values = pca.transform(scaled_input)
    component_names = [f"PC{index}" for index in range(1, pca_values.shape[1] + 1)]
    return pd.DataFrame(pca_values, columns=component_names)


def category_options(raw_data: pd.DataFrame, column: str) -> list[str]:
    if column not in raw_data:
        raise ValueError(f"Dữ liệu gốc thiếu cột {column}.")
    options = sorted(raw_data[column].dropna().astype(str).unique())
    if not options:
        raise ValueError(f"Không có lựa chọn hợp lệ cho thuộc tính {column}.")
    return options


def category_label(value: str) -> str:
    return " ".join(value.split())
