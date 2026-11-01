# ---- Web App container: Streamlit + mô hình đã huấn luyện sẵn ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Cài thư viện trước để tận dụng cache của Docker khi chỉ sửa code
COPY requirements.txt .
RUN pip install -r requirements.txt

# Mã nguồn + dữ liệu
COPY app/ ./app/
COPY data/ ./data/

# Huấn luyện mô hình ngay lúc build -> container khởi động là dùng được ngay
RUN python app/train_models.py

# Chạy bằng user thường (không dùng root)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
