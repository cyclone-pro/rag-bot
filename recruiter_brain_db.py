"""
Create Cloud SQL (Postgres) tables for Gmail agent.

Requires:
    pip install psycopg2-binary

Make sure the instance is reachable (public IP or via Cloud SQL Proxy),
and set these environment variables:

    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
"""

import os
from urllib.parse import urlparse
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DDL_SQL = """
-- vendors
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id            TEXT PRIMARY KEY,
    vendor_code          TEXT NOT NULL,
    company_name         TEXT NOT NULL,
    company_name_search  TEXT NOT NULL,
    created_at_utc       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- recruiters
CREATE TABLE IF NOT EXISTS recruiters (
    recruiter_id   TEXT PRIMARY KEY,
    vendor_id      TEXT NOT NULL REFERENCES vendors(vendor_id) ON DELETE CASCADE,
    email_enc      TEXT NOT NULL,
    name_enc       TEXT NOT NULL,
    role_title     TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- jobs
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    vendor_id        TEXT NOT NULL REFERENCES vendors(vendor_id) ON DELETE CASCADE,
    title            TEXT NOT NULL,
    role_code        TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    location         TEXT,
    location_type    TEXT,
    created_at_utc   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status           TEXT NOT NULL DEFAULT 'open',
    closed_at_utc    TIMESTAMPTZ,
    closed_reason    TEXT
);

-- candidates
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id           TEXT PRIMARY KEY,
    email_enc              TEXT NOT NULL,
    phone_enc              TEXT,
    name_enc               TEXT,
    location_city          TEXT,
    location_state         TEXT,
    location_country       TEXT,
    consent_to_process     BOOLEAN DEFAULT FALSE,
    consent_timestamp_utc  TIMESTAMPTZ,
    created_at_utc         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- job_applications
CREATE TABLE IF NOT EXISTS job_applications (
    application_id     TEXT PRIMARY KEY,
    candidate_id       TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    job_id             TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    vendor_id          TEXT NOT NULL REFERENCES vendors(vendor_id) ON DELETE CASCADE,
    current_stage      TEXT NOT NULL,
    stage_history_json JSONB,
    created_at_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- conversation_state
CREATE TABLE IF NOT EXISTS conversation_state (
    thread_type            TEXT NOT NULL,
    thread_key             TEXT PRIMARY KEY,
    job_id                 TEXT,
    summary_json           JSONB,
    pending_fields_json    JSONB,
    last_stage_counts_json JSONB,
    last_action            TEXT,
    last_seen_at_utc       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at_utc         TIMESTAMPTZ
);

-- events
CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    event_type     TEXT NOT NULL,
    actor_type     TEXT,
    actor_id       TEXT,
    job_id         TEXT,
    ts_utc         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json  JSONB
);
"""


def get_conn():
    raw_host = os.getenv("PG_HOST", "127.0.0.1")
    parsed = urlparse(raw_host)
    host = parsed.hostname if parsed.scheme else raw_host
    port = str(parsed.port) if parsed.scheme and parsed.port else os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "recruiter_brain")
    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "")

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=pwd,
    )
    conn.autocommit = True
    return conn


def create_tables():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL_SQL)
            print("✅ Tables created / already existed.")
    finally:
        conn.close()


if __name__ == "__main__":
    create_tables()
