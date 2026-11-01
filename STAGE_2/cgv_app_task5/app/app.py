"""CGV – Ứng dụng web dự đoán & phân khúc suất chiếu (Streamlit).

Tích hợp 3 kết quả suy luận cho một suất chiếu mới:
  1) Giá trị dự đoán (hồi quy)      – Task 3
  2) Nhóm phân cụm (PCA + K-Means)  – Task 1, 2
  3) Nhãn phân lớp (Cao/Không cao)  – Task 4
"""
from datetime import date, time, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import db
from inference import predict_all
from train_models import load_or_train

st.set_page_config(page_title="CGV – Dự đoán & Phân khúc suất chiếu",
                   page_icon="🎬", layout="wide")

CLUSTER_COLORS = px.colors.qualitative.Set2


@st.cache_resource(show_spinner="Đang tải / huấn luyện mô hình...")
def get_bundle() -> dict:
    return load_or_train()


@st.cache_data(ttl=20, show_spinner=False)
def get_db_status() -> tuple[bool, str]:
    return db.check_connection()


bundle = get_bundle()
opt = bundle["options"]
movies = bundle["movies"]
profile = bundle["cluster_profile"]

# =====================================================================  SIDEBAR – INPUTS
with st.sidebar:
    st.header("🎬 Thông tin suất chiếu")

    movie_mode = st.radio("Phim", ["Chọn phim có sẵn", "Nhập phim mới"], horizontal=True)
    if movie_mode == "Chọn phim có sẵn":
        movie_name = st.selectbox("Tên phim", list(movies.index))
        m = movies.loc[movie_name]
        rating, genre_str = m["Rating"], m["Genre"]
        duration = int(m["DurationMinutes"])
        release_date = date.fromisoformat(str(m["ReleaseDate"]))
        st.caption(f"Nhãn tuổi **{rating}** · {duration} phút · {genre_str} · "
                   f"khởi chiếu {release_date:%d/%m/%Y}")
    else:
        movie_name = st.text_input("Tên phim", value="Phim mới")
        rating = st.selectbox("Nhãn độ tuổi", opt["ratings"])
        genres = st.multiselect("Thể loại", opt["genres"], default=opt["genres"][:1])
        genre_str = ", ".join(genres)
        duration = st.slider("Thời lượng (phút)", 60, 210, 110)
        release_date = st.date_input("Ngày khởi chiếu", value=date.fromisoformat(opt["date_min"]))
        if not genres:
            st.warning("Nên chọn ít nhất một thể loại.")

    st.divider()
    cinema = st.selectbox("Rạp", opt["cinemas"])
    format_tech = st.selectbox("Định dạng chiếu", opt["format_techs"])
    room_type = st.selectbox("Loại phòng", ["Phòng thường",
                                            "Phòng đặc biệt (IMAX / 4DX / GOLD CLASS / Lamour...)"])
    show_date = st.date_input("Ngày chiếu", value=date.fromisoformat(opt["date_max"]))
    show_time = st.slider("Giờ chiếu", min_value=time(8, 0), max_value=time(23, 55),
                          value=time(19, 0), step=timedelta(minutes=5), format="HH:mm")
    if show_date < release_date:
        st.warning("Ngày chiếu sớm hơn ngày khởi chiếu của phim.")

    st.divider()
    total_seats = st.selectbox("Tổng số ghế của phòng", opt["total_seats"],
                               index=opt["total_seats"].index(100) if 100 in opt["total_seats"] else 0)
    ticket_price = st.slider("Giá vé (VND)", min_value=50_000, max_value=150_000,
                             value=opt["price_median"], step=5_000, format="%d VND")
    available_seats = st.slider("Số ghế còn trống", min_value=0, max_value=int(total_seats),
                                value=int(round(total_seats * 0.8)), key=f"avail_{total_seats}",
                                help="Chỉ dùng cho bước phân cụm (K-Means). "
                                     "Mô hình hồi quy và phân loại không dùng cột này.")

    with st.expander("⚙️ Tuỳ chọn thuật toán"):
        reg_name = st.radio("Hồi quy (giá trị dự đoán)", ["Decision Tree", "Linear Regression"])
        clf_name = st.radio("Phân loại (nhãn)", ["Decision Tree", "Logistic Regression"])

    predict_clicked = st.button("🔮 Dự đoán", type="primary")

inputs = {
    "Cinema": cinema, "ShowDate": show_date.isoformat(), "ShowTime": f"{show_time:%H:%M}",
    "FormatTech": format_tech, "IsPremiumRoom": int(room_type != "Phòng thường"),
    "TicketPrice": int(ticket_price), "AvailableSeats": int(available_seats),
    "TotalSeats": int(total_seats), "Rating": rating, "Genre": genre_str,
    "ReleaseDate": release_date.isoformat(), "DurationMinutes": int(duration),
}

if predict_clicked:
    res = predict_all(bundle, inputs, reg_name, clf_name)
    ok, msg = db.log_prediction(movie_name, inputs, res)
    st.session_state["result"] = {
        "movie": movie_name, "inputs": inputs, "res": res,
        "reg_name": reg_name, "clf_name": clf_name, "db_ok": ok, "db_msg": msg,
    }

# =====================================================================  MAIN
st.title("🎬 CGV – Dự đoán & Phân khúc suất chiếu")
db_ok, db_msg = get_db_status()
st.caption(("🟢 " if db_ok else "🟡 ") + db_msg)

tab_pred, tab_viz, tab_perf, tab_hist = st.tabs(
    ["🔮 Dự đoán", "📊 Trực quan hóa PCA & K-Means", "📈 Hiệu năng mô hình", "🗄️ Lịch sử (Database)"])

# ---------------------------------------------------------------  TAB 1: kết quả dự đoán
with tab_pred:
    saved = st.session_state.get("result")
    if not saved:
        st.info("Nhập thông tin suất chiếu ở thanh bên rồi bấm **🔮 Dự đoán** để xem 3 kết quả.")
    else:
        res, inp = saved["res"], saved["inputs"]
        st.subheader(f"Kết quả cho: {saved['movie']}")
        st.caption(f"{inp['Cinema']} · {inp['ShowDate']} {inp['ShowTime']} · {inp['FormatTech']} · "
                   f"{inp['TotalSeats']} ghế · giá vé {inp['TicketPrice']:,} VND")

        c1, c2, c3 = st.columns(3)
        with c1, st.container(border=True):
            st.markdown("##### 1️⃣ Giá trị dự đoán")
            st.metric("Doanh thu suất chiếu", f"{res['revenue'] / 1e6:,.2f} triệu VND")
            st.caption(f"Mô hình: {saved['reg_name']} · sai số điển hình (RMSE) "
                       f"≈ ±{res['rmse'] / 1e6:,.2f} triệu VND")
            st.caption(f"≈ {res['revenue'] / inp['TicketPrice']:,.0f} vé với giá vé đã nhập")
        with c2, st.container(border=True):
            k = res["cluster"]
            st.markdown("##### 2️⃣ Nhóm phân cụm")
            st.metric("Cụm K-Means", f"Cụm {k}")
            st.caption(f"{profile.loc[k, 'Mô tả']}")
            st.caption(f"Cụm có {profile.loc[k, 'Số suất']} suất · giá vé TB "
                       f"{profile.loc[k, 'Giá vé TB']:,} VND")
        with c3, st.container(border=True):
            st.markdown("##### 3️⃣ Nhãn phân lớp")
            st.metric("Mức nhu cầu", res["label"])
            st.progress(min(max(res["p_high"], 0.0), 1.0),
                        text=f"Xác suất nhu cầu cao: {res['p_high']:.0%}")
            st.caption(f"Mô hình: {saved['clf_name']} · 'Cao' = tỉ lệ lấp đầy > "
                       f"{bundle['occupancy_threshold']:.0%}")

        (st.success if saved["db_ok"] else st.warning)(saved["db_msg"])

        with st.expander("Hồ sơ cụm được dự đoán"):
            st.dataframe(profile.loc[[res["cluster"]]], hide_index=False)
        with st.expander("Vector đặc trưng đã đưa vào mô hình"):
            st.dataframe(res["features"].rename("Giá trị").to_frame())

    r2 = bundle["reg_metrics"].loc["Decision Tree", "R² (CV 5-fold)"]
    st.caption(f"⚠️ Mô hình học từ {int(profile['Số suất'].sum()):,} suất chiếu trong khoảng "
               f"{opt['date_min']} → {opt['date_max']}. Dữ liệu lịch chiếu cho ít tín hiệu về nhu cầu "
               f"(R² kiểm tra chéo của hồi quy ≈ {r2:.2f}), nên hãy xem kết quả như ước lượng tham khảo.")

# ---------------------------------------------------------------  TAB 2: trực quan hóa
with tab_viz:
    saved = st.session_state.get("result")
    viz = bundle["viz"]
    centers = bundle["kmeans"].cluster_centers_
    explained, cumulative = bundle["explained"], bundle["cumulative"]
    n_comp = bundle["n_components"]

    st.subheader("Phương sai giải thích của PCA")
    x = list(range(1, len(explained) + 1))
    fig_var = go.Figure()
    fig_var.add_bar(x=x, y=explained, name="Từng thành phần", marker_color="#4C78A8")
    fig_var.add_scatter(x=x, y=cumulative, name="Tích lũy", mode="lines+markers",
                        line=dict(color="#E45756"))
    for thr in (0.85, 0.90, 0.95):
        fig_var.add_hline(y=thr, line_dash="dot", line_color="gray",
                          annotation_text=f"{thr:.0%}", annotation_position="bottom right")
    fig_var.add_vline(x=n_comp, line_dash="dash", line_color="green",
                      annotation_text=f"Chọn {n_comp} thành phần ({cumulative[n_comp - 1]:.1%})")
    fig_var.update_layout(xaxis_title="Thành phần chính", yaxis_title="Tỉ lệ phương sai",
                          height=380, margin=dict(t=30))
    st.plotly_chart(fig_var)

    st.subheader("Chọn K tối ưu cho K-Means")
    scan = bundle["kmeans_scan"]
    fig_k = make_subplots(rows=1, cols=2, subplot_titles=("Elbow (WCSS)", "Silhouette score"))
    fig_k.add_scatter(x=scan["K"], y=scan["WCSS"], mode="lines+markers", row=1, col=1, name="WCSS")
    fig_k.add_scatter(x=scan["K"], y=scan["Silhouette"], mode="lines+markers", row=1, col=2,
                      name="Silhouette", line=dict(color="#54A24B"))
    fig_k.add_vline(x=bundle["k"], line_dash="dash", line_color="red", row=1, col=2)
    fig_k.update_layout(height=340, showlegend=False, margin=dict(t=40))
    fig_k.update_xaxes(title_text="Số cụm K")
    st.plotly_chart(fig_k)

    st.subheader(f"Phân cụm K-Means (K={bundle['k']}) trong không gian PCA")
    plot_df = viz.copy()
    plot_df["Cụm"] = plot_df["Cluster"].astype(str)
    order = {"Cụm": [str(i) for i in range(bundle["k"])]}

    cA, cB = st.columns(2)
    with cA:
        fig2 = px.scatter(plot_df, x="PC1", y="PC2", color="Cụm", category_orders=order,
                          hover_data=["MovieName", "Cinema", "ShowTime"],
                          color_discrete_sequence=CLUSTER_COLORS, opacity=0.7, title="Scatter 2D")
        fig2.add_scatter(x=centers[:, 0], y=centers[:, 1], mode="markers", name="Tâm cụm",
                         marker=dict(symbol="x", size=14, color="black"))
        if saved:
            p = saved["res"]["pca_point"]
            fig2.add_scatter(x=[p[0]], y=[p[1]], mode="markers", name="Suất chiếu của bạn",
                             marker=dict(symbol="star", size=20, color="gold",
                                         line=dict(color="black", width=1.5)))
        fig2.update_layout(height=520)
        st.plotly_chart(fig2)
    with cB:
        fig3 = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="Cụm", category_orders=order,
                             hover_data=["MovieName", "Cinema", "ShowTime"],
                             color_discrete_sequence=CLUSTER_COLORS, opacity=0.7, title="Scatter 3D")
        fig3.update_traces(marker=dict(size=3))
        if saved:
            p = saved["res"]["pca_point"]
            fig3.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                        name="Suất chiếu của bạn",
                                        marker=dict(symbol="diamond", size=9, color="gold",
                                                    line=dict(color="black", width=2))))
        fig3.update_layout(height=520)
        st.plotly_chart(fig3)
    if not saved:
        st.caption("Bấm **Dự đoán** để đánh dấu vị trí suất chiếu của bạn (⭐) trên các biểu đồ.")

    st.subheader("Hồ sơ từng cụm")
    st.dataframe(profile)

# ---------------------------------------------------------------  TAB 3: hiệu năng
with tab_perf:
    st.subheader("Hồi quy – dự đoán doanh thu (Task 3)")
    st.dataframe(bundle["reg_metrics"].style.format({
        "MSE (test)": "{:,.0f}", "RMSE (test)": "{:,.0f}", "R² (test)": "{:.3f}",
        "R² (train)": "{:.3f}", "R² (CV 5-fold)": "{:.3f}"}))

    y_true, preds = bundle["reg_test"]
    sel = saved["reg_name"] if (saved := st.session_state.get("result")) else "Decision Tree"
    fig_av = px.scatter(x=y_true / 1e6, y=preds[sel] / 1e6, opacity=0.5,
                        labels={"x": "Thực tế (triệu VND)", "y": "Dự đoán (triệu VND)"},
                        title=f"Thực tế so với Dự đoán – {sel}")
    lim = float(max(y_true.max(), preds[sel].max()) / 1e6)
    fig_av.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="Dự đoán hoàn hảo",
                       line=dict(color="red", dash="dash"))
    fig_av.update_layout(height=420)
    st.plotly_chart(fig_av)

    st.subheader("Phân loại – mức nhu cầu Cao / Không cao (Task 4)")
    st.dataframe(bundle["clf_metrics"].style.format("{:.3f}"))
    cols = st.columns(len(bundle["confusion"]))
    for col, (name, cm) in zip(cols, bundle["confusion"].items()):
        fig_cm = px.imshow(cm, x=["Không cao", "Cao"], y=["Không cao", "Cao"], text_auto=True,
                           color_continuous_scale="Blues", aspect="auto",
                           labels=dict(x="Dự đoán", y="Thực tế"), title=name)
        fig_cm.update_layout(height=380, coloraxis_showscale=False)
        col.plotly_chart(fig_cm)

# ---------------------------------------------------------------  TAB 4: lịch sử database
with tab_hist:
    st.subheader("Lịch sử dự đoán đã lưu trong PostgreSQL")
    if st.button("🔄 Tải lại"):
        get_db_status.clear()
    hist, status = db.fetch_history()
    if hist is None:
        st.warning(status)
        st.caption("Khi chạy bằng `docker compose up --build`, container database sẽ tự khởi động "
                   "và dữ liệu được lưu bền vững trong volume `pgdata`.")
    elif hist.empty:
        st.info("Chưa có dự đoán nào được lưu. Hãy bấm **Dự đoán** ở thanh bên.")
    else:
        st.dataframe(hist, hide_index=True)
        st.download_button("⬇️ Tải CSV", hist.to_csv(index=False).encode("utf-8-sig"),
                           file_name="lich_su_du_doan.csv", mime="text/csv")
