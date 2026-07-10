"""
Database connection module.
The DB connection is configured via DATABASE_URL env var.
"""
import os
import sqlite3
from typing import Optional
from urllib.parse import urlparse

import psycopg2
import pymysql

DEFAULT_DATABASE_URL = "sqlite:///local_dev.db"
CONNECTION_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
CONNECTION_TIMEOUT = int(os.getenv("DB_TIMEOUT", "30"))


def _parse_sqlite_url(url: str) -> dict:
    parsed = urlparse(url)
    db_path = parsed.path.lstrip("/")
    return {"database": db_path, "type": "sqlite"}


def _parse_postgres_url(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "type": "postgres",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/"),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
    }


def _parse_mysql_url(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "type": "mysql",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "database": parsed.path.lstrip("/"),
        "user": parsed.username or "root",
        "password": parsed.password or "",
    }


def parse_database_url(url: str) -> dict:
    if url.startswith("sqlite"):
        return _parse_sqlite_url(url)
    elif url.startswith("postgres"):
        return _parse_postgres_url(url)
    elif url.startswith("mysql"):
        return _parse_mysql_url(url)
    else:
        raise ValueError(f"Unsupported database scheme in URL: {url}")


def get_connection() -> object:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    config = parse_database_url(url)

    if config["type"] == "sqlite":
        conn = sqlite3.connect(
            config["database"],
            timeout=CONNECTION_TIMEOUT,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    elif config["type"] == "postgres":
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            dbname=config["database"],
            user=config["user"],
            password=config["password"],
            connect_timeout=CONNECTION_TIMEOUT,
        )
        conn.autocommit = False
        return conn

    elif config["type"] == "mysql":
        conn = pymysql.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
            connect_timeout=CONNECTION_TIMEOUT,
        )
        return conn

    raise RuntimeError(f"Unable to create connection for type: {config['type']}")


def close_connection(conn: Optional[object]) -> None:
    if conn is not None:
        conn.close()


def ping_database(conn: object) -> bool:
    try:
        if isinstance(conn, sqlite3.Connection):
            conn.execute("SELECT 1")
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        return True
    except Exception:
        return False
