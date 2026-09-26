"""Tiny Postgres helpers shared by every Python service."""
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()
PG_DSN = os.getenv("PG_DSN", "dbname=stocks user=stocks password=stocks host=localhost port=5432")


def read_sql(sql: str, params: tuple = (), dsn: str = PG_DSN) -> pd.DataFrame:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()


def write_rows(sql: str, rows: list[tuple], dsn: str = PG_DSN, template: str | None = None) -> None:
    """sql must contain a single `VALUES %s` placeholder (psycopg2 execute_values).
    template lets you add casts per value, e.g. "(%s, %s::vector)"."""
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            execute_values(cur, sql, rows, template=template)
    finally:
        conn.close()


def execute(sql: str, params: tuple = (), dsn: str = PG_DSN) -> None:
    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, params)
    finally:
        conn.close()
