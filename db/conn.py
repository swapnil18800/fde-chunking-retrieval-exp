"""psycopg3 connection pools (sync for scripts, async for the API). pgvector types registered on connect."""

from __future__ import annotations

from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from config import get_settings


def _configure(conn: psycopg.Connection) -> None:
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:  # extension not created yet (first setup_db run)
        pass


async def _configure_async(conn: psycopg.AsyncConnection) -> None:
    try:
        await register_vector_async(conn)
    except psycopg.ProgrammingError:
        pass


@lru_cache
def get_pool() -> ConnectionPool:
    s = get_settings()
    pool = ConnectionPool(
        s.database_url, min_size=s.db_pool_min, max_size=s.db_pool_max, open=True,
        configure=_configure, kwargs={"row_factory": dict_row, "connect_timeout": 15},
    )
    return pool


_async_pool: AsyncConnectionPool | None = None


async def get_async_pool() -> AsyncConnectionPool:
    global _async_pool
    if _async_pool is None:
        s = get_settings()
        _async_pool = AsyncConnectionPool(
            s.database_url, min_size=s.db_pool_min, max_size=s.db_pool_max, open=False,
            configure=_configure_async, kwargs={"row_factory": dict_row, "connect_timeout": 15},
        )
        await _async_pool.open()
    return _async_pool


def connect() -> psycopg.Connection:
    """One-off sync connection (autocommit off). Prefer get_pool() in long-lived code."""
    s = get_settings()
    conn = psycopg.connect(s.database_url, row_factory=dict_row, connect_timeout=15)
    _configure(conn)
    return conn
