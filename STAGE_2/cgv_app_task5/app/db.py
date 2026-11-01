"""Lớp truy cập PostgreSQL – lưu lịch sử dự đoán.

Nếu không kết nối được database, ứng dụng vẫn chạy bình thường
(các hàm trả về trạng thái lỗi thay vì ném ngoại lệ)."""
import json
import os

import pandas as pd

try:
    import psycopg2
except ImportError:  # chạy local mà chưa cài psycopg2
    psycopg2 = None

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    movie_name      TEXT,
    cinema          TEXT,
    show_date       DATE,
    show_time       TEXT,
    format_tech     TEXT,
    total_seats     INTEGER,
    ticket_price    INTEGER,
    available_seats INTEGER,
    predicted_revenue DOUBLE PRECISION,
    cluster_label   INTEGER,
    demand_label    TEXT,
    p_high_demand   DOUBLE PRECISION,
    raw_input       JSONB
);
"""


def _connect():
    if psycopg2 is None:
        raise RuntimeError("Chưa cài psycopg2")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "cgv"),
        user=os.getenv("POSTGRES_USER", "cgv_user"),
        password=os.getenv("POSTGRES_PASSWORD", "cgv_pass"),
        connect_timeout=3,
    )


def check_connection() -> tuple[bool, str]:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(DDL)
        return True, "Đã kết nối PostgreSQL"
    except Exception as exc:  # noqa: BLE001
        return False, f"Chưa kết nối được database ({type(exc).__name__}). Ứng dụng vẫn chạy bình thường."


def log_prediction(movie_name: str, inputs: dict, result: dict) -> tuple[bool, str]:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(
                """INSERT INTO predictions
                   (movie_name, cinema, show_date, show_time, format_tech, total_seats,
                    ticket_price, available_seats, predicted_revenue, cluster_label,
                    demand_label, p_high_demand, raw_input)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (movie_name, inputs["Cinema"], str(inputs["ShowDate"]), inputs["ShowTime"],
                 inputs["FormatTech"], int(inputs["TotalSeats"]), int(inputs["TicketPrice"]),
                 int(inputs["AvailableSeats"]), float(result["revenue"]),
                 int(result["cluster"]), result["label"], float(result["p_high"]),
                 json.dumps(inputs, default=str, ensure_ascii=False)),
            )
        return True, "Đã lưu kết quả vào database"
    except Exception as exc:  # noqa: BLE001
        return False, f"Không lưu được vào database ({type(exc).__name__})"


def fetch_history(limit: int = 100) -> tuple[pd.DataFrame | None, str]:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, movie_name, cinema, show_date, show_time,
                          format_tech, total_seats, ticket_price, available_seats,
                          ROUND(predicted_revenue::numeric, 0)::float8,
                          cluster_label, demand_label,
                          ROUND(p_high_demand::numeric, 3)::float8
                   FROM predictions ORDER BY id DESC LIMIT %s""", (limit,))
            rows = cur.fetchall()
        cols = ["id", "created_at", "Phim", "Rạp", "Ngày chiếu", "Giờ", "Định dạng",
                "Tổng ghế", "Giá vé", "Ghế trống", "Doanh thu dự đoán", "Cụm", "Nhu cầu", "P(Cao)"]
        return pd.DataFrame(rows, columns=cols), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"Không đọc được lịch sử ({type(exc).__name__})"
