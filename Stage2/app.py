from datetime import date, time, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from task5_model import (
    build_encoded_input,
    category_label,
    category_options,
    load_resources,
    transform_input,
)


st.set_page_config(
    page_title="CGV Movie Analytics | Task 5",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def get_resources() -> dict[str, object]:
    return load_resources()


st.title("🎬 CGV Movie Analytics")
st.caption("Task 5 · Giao diện dự đoán nhãn cụm phim và suất chiếu")

try:
    resources = get_resources()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

raw_data = resources["raw_data"]
encoded_columns = resources["encoded_columns"]
scaler = resources["scaler"]
pca = resources["pca"]
models = resources["models"]
metrics = resources["metrics"]
features = resources["features"]
labels = resources["labels"]
clustered_data = resources["clustered_data"]

with st.sidebar:
    st.header("Thông tin đầu vào")
    selected_model = st.selectbox(
        "Mô hình phân loại",
        options=list(models),
        help="Hai mô hình được huấn luyện theo cấu hình của Task 4.",
    )
    duration = st.slider(
        "Thời lượng phim (phút)",
        min_value=60,
        max_value=300,
        value=120,
        step=5,
    )
    show_date = st.date_input("Ngày chiếu", value=date.today())
    show_time = st.time_input("Giờ chiếu", value=time(19, 0))
    release_date = st.date_input(
        "Ngày phát hành",
        value=date.today() - timedelta(days=30),
    )

    categories: dict[str, str] = {}
    for column, label in (
        ("Cinema", "Rạp chiếu"),
        ("Format", "Định dạng chiếu"),
        ("Rating", "Nhãn phân loại phim"),
        ("Genre", "Thể loại"),
        ("Language", "Ngôn ngữ"),
    ):
        options = category_options(raw_data, column)
        default_value = str(raw_data[column].mode(dropna=True).iloc[0])
        categories[column] = st.selectbox(
            label,
            options=options,
            index=options.index(default_value),
            format_func=category_label,
        )

try:
    encoded_input = build_encoded_input(
        duration_minutes=duration,
        show_date=show_date,
        show_time=show_time,
        release_date=release_date,
        categories=categories,
        encoded_columns=encoded_columns,
    )
    pca_input = transform_input(encoded_input, scaler, pca)
except ValueError as error:
    st.error(str(error))
    st.stop()

model = models[selected_model]
predicted_cluster = int(model.predict(pca_input)[0])
cluster_probabilities = model.predict_proba(pca_input)[0]
class_labels = list(model.classes_)
probability_table = pd.DataFrame(
    {
        "Nhãn cụm": class_labels,
        "Xác suất dự đoán": cluster_probabilities,
    }
).sort_values("Xác suất dự đoán", ascending=False)
cluster_size = int((labels == predicted_cluster).sum())

prediction_tab, visualization_tab = st.tabs(
    ["🔎 Dự đoán", "📊 Trực quan hóa"]
)

with prediction_tab:
    st.subheader("Kết quả suy luận")
    prediction_col, confidence_col, cluster_col = st.columns(3)
    prediction_col.metric("Nhãn cụm dự đoán", f"Cluster {predicted_cluster}")
    confidence_col.metric(
        "Xác suất cao nhất",
        f"{probability_table.iloc[0]['Xác suất dự đoán']:.1%}",
    )
    cluster_col.metric("Số suất chiếu trong cụm", f"{cluster_size:,}")
    st.caption(
        "Cluster_Label là nhãn cụm do K-Means tạo ở Task 2; "
        "nhãn số không phải thứ hạng chất lượng phim."
    )

    st.markdown("#### Xác suất theo từng cụm")
    st.dataframe(
        probability_table.style.format({"Xác suất dự đoán": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Đánh giá trên tập kiểm tra Task 4")
    metrics_table = pd.DataFrame.from_dict(metrics, orient="index")
    st.dataframe(
        metrics_table.style.format("{:.3f}"),
        width="stretch",
    )
    st.caption(
        "Các chỉ số được tính trên cùng phép chia train/test phân tầng 80/20 "
        "và random_state=42 như Task 4."
    )

with visualization_tab:
    st.subheader("Vị trí dữ liệu trong không gian PCA")
    figure, axis = plt.subplots(figsize=(10, 6))
    for cluster_label in sorted(labels.unique()):
        mask = labels == cluster_label
        axis.scatter(
            features.loc[mask, "PC1"],
            features.loc[mask, "PC2"],
            label=f"Cụm {cluster_label}",
            alpha=0.55,
            s=24,
        )
    axis.scatter(
        pca_input.loc[0, "PC1"],
        pca_input.loc[0, "PC2"],
        marker="*",
        s=280,
        color="black",
        edgecolor="white",
        linewidth=0.8,
        label="Dữ liệu đang dự đoán",
        zorder=5,
    )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("Phân bố các cụm trên hai thành phần PCA đầu tiên")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.markdown("#### Một số phim/suất chiếu trong cụm được dự đoán")
    examples = clustered_data.loc[
        clustered_data["Cluster_Label"] == predicted_cluster,
        ["MovieName", "Cinema", "ShowDate", "Rating", "Genre"],
    ].head(8)
    st.dataframe(examples, hide_index=True, width="stretch")
    st.caption(
        "Biểu đồ và danh sách trên dùng các nhãn cụm từ Task 2; "
        "dữ liệu mới được chiếu lên cùng không gian PCA."
    )
