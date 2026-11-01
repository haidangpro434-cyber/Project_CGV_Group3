"""Suy luận: từ thông tin một suất chiếu mới -> 3 kết quả (giá trị, cụm, nhãn)."""
import pandas as pd

from features import build_matrix


def predict_all(bundle: dict, inputs: dict, reg_name: str, clf_name: str) -> dict:
    """inputs: Cinema, ShowDate, ShowTime('HH:MM'), FormatTech, IsPremiumRoom,
    TicketPrice, AvailableSeats, TotalSeats, Rating, Genre('A, B'), ReleaseDate,
    DurationMinutes."""
    row = pd.DataFrame([inputs])
    X_full = build_matrix(row, bundle["schema"])

    # --- Task 3: hồi quy (doanh thu dự đoán của suất chiếu)
    reg = bundle["reg_models"][reg_name]
    revenue = float(reg.predict(X_full[bundle["reg_cols"]])[0])
    rmse = float(bundle["reg_metrics"].loc[reg_name, "RMSE (test)"])

    # --- Task 1 + 2: chuẩn hóa -> PCA -> K-Means
    X_pca = bundle["pca"].transform(bundle["scaler"].transform(X_full))
    cluster = int(bundle["kmeans"].predict(X_pca)[0])

    # --- Task 4: phân loại nhu cầu Cao / Không cao
    clf = bundle["clf_models"][clf_name]
    X_clf = X_full[bundle["clf_cols"]]
    label = str(clf.predict(X_clf)[0])
    proba = clf.predict_proba(X_clf)[0]
    p_high = float(proba[list(clf.classes_).index("Cao")])

    return {
        "revenue": max(revenue, 0.0), "rmse": rmse,
        "cluster": cluster, "pca_point": X_pca[0],
        "label": label, "p_high": p_high,
        "occupancy_input": float(X_full["OccupancyRate"].iloc[0]),
        "features": X_full.iloc[0],
    }
