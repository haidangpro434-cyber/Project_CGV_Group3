# CGV – Ứng dụng web dự đoán & phân khúc suất chiếu (Task 5)

Ứng dụng **Streamlit** tích hợp các mô hình của Task 1–4 và được đóng gói bằng **Docker** (nhiều container).

## Chạy bằng Docker (khuyến nghị) – chỉ một lệnh

```bash
docker compose up --build        # hoặc: docker-compose up --build
```

Mở trình duyệt: **http://localhost:8501**. Dừng hệ thống: `Ctrl + C`, sau đó `docker compose down`
(thêm `-v` nếu muốn xóa luôn dữ liệu database).

Hệ thống gồm 2 container nằm chung mạng `cgv-net`:

| Container | Vai trò | Ghi chú |
|---|---|---|
| `cgv_web` | Streamlit web app | Mô hình được huấn luyện ngay lúc `build` nên khởi động là dùng được |
| `cgv_db` | PostgreSQL 16 | Lưu lịch sử dự đoán; dữ liệu nằm trong volume `pgdata` |

`web` chỉ khởi động khi `db` đã qua healthcheck (`depends_on: service_healthy`). Web gọi database bằng tên
service `db` qua mạng nội bộ; cổng 5432 không mở ra ngoài. Đổi mật khẩu database: sao chép `.env.example`
thành `.env` rồi sửa giá trị.

## Chạy trực tiếp trên máy (không dùng Docker)

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python app/train_models.py                          # huấn luyện và lưu models/bundle.joblib
streamlit run app/app.py
```

Nếu không có database, ứng dụng vẫn chạy bình thường; chỉ tab **Lịch sử** báo chưa kết nối.
(Nếu bỏ qua bước `train_models.py`, ứng dụng tự huấn luyện ở lần mở đầu tiên.)

## Đối chiếu với yêu cầu

| Yêu cầu | Nơi thực hiện |
|---|---|
| Framework Streamlit | `app/app.py` |
| Đầu vào: sliders, text box, dropdown | Thanh bên: slider (giờ chiếu, giá vé, ghế trống, thời lượng), text box (tên phim mới), dropdown (phim, rạp, định dạng, loại phòng, tổng ghế), multiselect thể loại, date picker |
| Kết quả 1 – giá trị dự đoán | Doanh thu suất chiếu (hồi quy Task 3) |
| Kết quả 2 – nhóm phân cụm | Cụm K-Means trên PCA (Task 1, 2) |
| Kết quả 3 – nhãn phân lớp | Nhu cầu Cao / Không cao + xác suất (Task 4) |
| Tab trực quan hóa | Biểu đồ phương sai PCA, Elbow/Silhouette, scatter 2D/3D có đánh dấu suất chiếu vừa nhập |
| Dockerfile | `Dockerfile` |
| docker-compose đa container, network, volume | `docker-compose.yml` (web + db, mạng `cgv-net`, volume `pgdata`, mount `./data`) |

## Cấu trúc

```
app/features.py       tạo đặc trưng (dùng chung cho huấn luyện và suy luận)
app/train_models.py   huấn luyện PCA, K-Means, hồi quy, phân loại -> models/bundle.joblib
app/inference.py      suy luận 3 kết quả cho một suất chiếu mới
app/db.py             ghi/đọc lịch sử dự đoán (PostgreSQL)
app/app.py            giao diện Streamlit
data/                 cgv_eda_cleaned.csv
db/init.sql           tạo bảng predictions lần đầu
```

## Lưu ý khi diễn giải

Dữ liệu là lịch chiếu của 5 rạp trong ~10 ngày (1.172 suất) và gần như không có tín hiệu về nhu cầu khách,
nên các mô hình có hiệu năng khiêm tốn (xem tab **Hiệu năng mô hình**). Hãy xem kết quả là ước lượng tham khảo.
