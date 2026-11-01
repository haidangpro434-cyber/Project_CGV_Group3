"""Tạo đặc trưng – dùng chung cho HUẤN LUYỆN và SUY LUẬN.

Logic giống hệt các notebook Task 1-4 để kết quả khớp nhau.
"""
import re

import pandas as pd

BASE_COLS = [
    "TicketPrice", "AvailableSeats", "TotalSeats", "DurationMinutes",
    "OccupancyRate", "ShowHour", "DaysSinceRelease", "IsWeekend", "IsPremiumRoom",
]
# Task 3 (hồi quy): bỏ các cột cấu thành doanh thu để tránh rò rỉ dữ liệu
REG_EXCLUDE = ["TicketPrice", "AvailableSeats", "OccupancyRate"]
# Task 4 (phân loại): bỏ các cột cấu thành nhãn tỉ lệ lấp đầy
CLF_EXCLUDE = ["AvailableSeats", "OccupancyRate"]


def format_tech(fmt) -> str:
    """'2D Phụ Đề Việt | Rạp Lamour' -> '2D' ; '4DX2D Phụ Đề ...' -> '4DX2D'."""
    s = re.sub(r"\s+", " ", str(fmt)).strip()
    return re.split(r" Phụ| Lồng", s, maxsplit=1)[0].strip()


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm FormatTech và IsPremiumRoom từ cột Format gốc."""
    out = df.copy()
    out["FormatTech"] = out["Format"].map(format_tech)
    out["IsPremiumRoom"] = out["Format"].astype(str).str.contains("|", regex=False).astype(int)
    return out


def split_genres(value) -> list:
    return [g.strip() for g in str(value).split(",") if g.strip()]


def make_schema(df_in: pd.DataFrame) -> dict:
    genres = sorted({g for v in df_in["Genre"] for g in split_genres(v)})
    return {
        "Cinema": sorted(df_in["Cinema"].astype(str).unique()),
        "Rating": sorted(df_in["Rating"].astype(str).unique()),
        "FormatTech": sorted(df_in["FormatTech"].astype(str).unique()),
        "Genre": genres,
    }


def build_matrix(df_in: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Biến bảng thô (cột: Cinema, ShowDate, ShowTime, FormatTech, IsPremiumRoom,
    TicketPrice, AvailableSeats, TotalSeats, Rating, Genre, ReleaseDate,
    DurationMinutes) thành ma trận đặc trưng số, đúng thứ tự cột như lúc huấn luyện."""
    X = pd.DataFrame(index=df_in.index)
    X["TicketPrice"] = pd.to_numeric(df_in["TicketPrice"])
    X["AvailableSeats"] = pd.to_numeric(df_in["AvailableSeats"])
    X["TotalSeats"] = pd.to_numeric(df_in["TotalSeats"])
    X["DurationMinutes"] = pd.to_numeric(df_in["DurationMinutes"], errors="coerce")
    X["OccupancyRate"] = (X["TotalSeats"] - X["AvailableSeats"]) / X["TotalSeats"]

    t = pd.to_datetime(df_in["ShowTime"].astype(str), format="%H:%M")
    X["ShowHour"] = t.dt.hour + t.dt.minute / 60

    show_date = pd.to_datetime(df_in["ShowDate"])
    release_date = pd.to_datetime(df_in["ReleaseDate"])
    X["DaysSinceRelease"] = (show_date - release_date).dt.days
    X["IsWeekend"] = (show_date.dt.dayofweek >= 5).astype(int)
    X["IsPremiumRoom"] = pd.to_numeric(df_in["IsPremiumRoom"]).astype(int)

    for col in ["Cinema", "Rating", "FormatTech"]:
        values = df_in[col].astype(str)
        for v in schema[col]:
            X[f"{col}_{v}"] = (values == v).astype(int)

    genre_sets = df_in["Genre"].map(lambda s: set(split_genres(s)))
    for g in schema["Genre"]:
        X[f"Genre_{g}"] = genre_sets.map(lambda st, g=g: int(g in st))
    return X
