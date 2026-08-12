"""Print public-safe FlyRank warehouse schemas without exposing credentials."""

from __future__ import annotations

import duckdb
from huggingface_hub import get_token


BASE = "hf://datasets/FlyRank/internship-warehouse/"
TABLES = [
    "dim_clients.parquet",
    "dim_content.parquet",
    "fact_content_query_90d.parquet",
    "fact_content_daily_performance/month=2026-03/data_0.parquet",
]


def main() -> None:
    token = get_token()
    if not token:
        raise RuntimeError("Run `hf auth login` before this script.")

    connection = duckdb.connect()
    escaped_token = token.replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{escaped_token}')"
    )

    for table in TABLES:
        relation = f"read_parquet('{BASE}{table}')"
        schema = connection.sql(f"DESCRIBE SELECT * FROM {relation}").df()
        print(f"\n### {table}")
        print(schema.to_string(index=False))


if __name__ == "__main__":
    main()
