-- Chạy tự động lần đầu khi container PostgreSQL khởi tạo volume.
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
