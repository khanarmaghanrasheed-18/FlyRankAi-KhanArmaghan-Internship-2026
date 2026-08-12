"""Build a small, public-safe page-pair feature table from remote aggregates."""

from __future__ import annotations

from pathlib import Path

import duckdb
from huggingface_hub import get_token


BASE = "hf://datasets/FlyRank/internship-warehouse"
OUTPUT = Path("work/outputs/page_pair_features.parquet")


PAIR_SQL = f"""
WITH ranked_query_rows AS (
    SELECT
        client_hash_id,
        content_hash_id,
        query_hash_id,
        impressions_90d,
        impressions_last30,
        impressions_prev30,
        clicks_last30,
        clicks_prev30,
        avg_position_90d,
        content_total_impressions_90d,
        content_visible_query_count,
        rare_impressions_share,
        anonymized_impressions_share,
        COUNT(*) OVER (PARTITION BY client_hash_id, query_hash_id) AS pages_per_query,
        ROW_NUMBER() OVER (
            PARTITION BY client_hash_id, content_hash_id
            ORDER BY impressions_90d DESC, query_hash_id
        ) AS query_rank_for_page
    FROM read_parquet('{BASE}/fact_content_query_90d.parquet')
    WHERE content_total_impressions_90d >= 500
      AND impressions_90d >= 10
),
query_rows AS (
    SELECT * FROM ranked_query_rows
    WHERE query_rank_for_page <= 50
),
eligible AS (
    SELECT *
    FROM query_rows
    WHERE pages_per_query BETWEEN 2 AND 10
),
pair_shared AS (
    SELECT
        a.client_hash_id,
        a.content_hash_id AS content_a,
        b.content_hash_id AS content_b,
        COUNT(*) AS shared_query_count,
        SUM(LEAST(a.impressions_90d, b.impressions_90d)) AS shared_impression_intersection,
        SUM(a.impressions_90d + b.impressions_90d) AS shared_query_impressions,
        SUM(a.impressions_last30 + b.impressions_last30) AS shared_impressions_last30,
        SUM(a.impressions_prev30 + b.impressions_prev30) AS shared_impressions_prev30,
        AVG(ABS(a.avg_position_90d - b.avg_position_90d)) AS mean_shared_position_gap,
        ANY_VALUE(a.content_total_impressions_90d) AS total_impressions_a,
        ANY_VALUE(b.content_total_impressions_90d) AS total_impressions_b,
        ANY_VALUE(a.content_visible_query_count) AS visible_queries_a,
        ANY_VALUE(b.content_visible_query_count) AS visible_queries_b,
        ANY_VALUE(a.rare_impressions_share) AS rare_share_a,
        ANY_VALUE(b.rare_impressions_share) AS rare_share_b,
        ANY_VALUE(a.anonymized_impressions_share) AS anonymized_share_a,
        ANY_VALUE(b.anonymized_impressions_share) AS anonymized_share_b
    FROM eligible a
    JOIN eligible b
      ON a.client_hash_id = b.client_hash_id
     AND a.query_hash_id = b.query_hash_id
     AND a.content_hash_id < b.content_hash_id
    GROUP BY 1, 2, 3
    HAVING COUNT(*) >= 2
),
content_movement AS (
    SELECT
        client_hash_id,
        content_hash_id,
        SUM(impressions_last30) AS visible_impressions_last30,
        SUM(impressions_prev30) AS visible_impressions_prev30,
        SUM(clicks_last30) AS visible_clicks_last30,
        SUM(clicks_prev30) AS visible_clicks_prev30
    FROM query_rows
    GROUP BY 1, 2
),
content_meta AS (
    SELECT
        client_hash_id,
        content_hash_id,
        content_type,
        main_intent,
        search_volume,
        word_count,
        content_created_date,
        content_updated_date,
        is_published,
        is_deleted
    FROM read_parquet('{BASE}/dim_content.parquet')
),
joined AS (
    SELECT
        p.*,
        ma.visible_impressions_last30 AS impressions_last30_a,
        ma.visible_impressions_prev30 AS impressions_prev30_a,
        mb.visible_impressions_last30 AS impressions_last30_b,
        mb.visible_impressions_prev30 AS impressions_prev30_b,
        ma.visible_clicks_last30 AS clicks_last30_a,
        ma.visible_clicks_prev30 AS clicks_prev30_a,
        mb.visible_clicks_last30 AS clicks_last30_b,
        mb.visible_clicks_prev30 AS clicks_prev30_b,
        ca.content_type AS content_type_a,
        cb.content_type AS content_type_b,
        ca.main_intent AS main_intent_a,
        cb.main_intent AS main_intent_b,
        ca.word_count AS word_count_a,
        cb.word_count AS word_count_b,
        ca.content_created_date AS created_a,
        cb.content_created_date AS created_b,
        ca.content_updated_date AS updated_a,
        cb.content_updated_date AS updated_b
    FROM pair_shared p
    JOIN content_movement ma
      ON p.client_hash_id = ma.client_hash_id
     AND p.content_a = ma.content_hash_id
    JOIN content_movement mb
      ON p.client_hash_id = mb.client_hash_id
     AND p.content_b = mb.content_hash_id
    JOIN content_meta ca
      ON p.client_hash_id = ca.client_hash_id
     AND p.content_a = ca.content_hash_id
    JOIN content_meta cb
      ON p.client_hash_id = cb.client_hash_id
     AND p.content_b = cb.content_hash_id
    WHERE ca.is_published IS TRUE AND cb.is_published IS TRUE
      AND ca.is_deleted IS NOT TRUE AND cb.is_deleted IS NOT TRUE
)
SELECT
    *,
    2.0 * shared_impression_intersection /
        NULLIF(total_impressions_a + total_impressions_b, 0) AS weighted_query_overlap,
    shared_query_count * 1.0 /
        NULLIF(LEAST(visible_queries_a, visible_queries_b), 0) AS smaller_page_query_coverage,
    shared_impression_intersection * 1.0 /
        NULLIF(LEAST(total_impressions_a, total_impressions_b), 0) AS smaller_page_demand_overlap,
    (impressions_last30_a - impressions_prev30_a) * 1.0 /
        NULLIF(impressions_prev30_a, 0) AS growth_a,
    (impressions_last30_b - impressions_prev30_b) * 1.0 /
        NULLIF(impressions_prev30_b, 0) AS growth_b,
    LEAST(total_impressions_a, total_impressions_b) * 1.0 /
        NULLIF(GREATEST(total_impressions_a, total_impressions_b), 0) AS visibility_balance,
    CASE WHEN main_intent_a = main_intent_b THEN 1 ELSE 0 END AS same_intent,
    CASE WHEN content_type_a = content_type_b THEN 1 ELSE 0 END AS same_content_type
FROM joined
"""


def main() -> None:
    token = get_token()
    if not token:
        raise RuntimeError("Run `hf auth login` before this script.")

    connection = duckdb.connect()
    escaped_token = token.replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{escaped_token}')"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    escaped_output = str(OUTPUT).replace("'", "''")
    connection.execute(
        f"COPY ({PAIR_SQL}) TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    stats = connection.sql(
        f"SELECT COUNT(*) AS pairs, COUNT(DISTINCT client_hash_id) AS clients "
        f"FROM read_parquet('{escaped_output}')"
    ).df()
    print(stats.to_string(index=False))
    print(f"Wrote aggregate pair features to {OUTPUT}")


if __name__ == "__main__":
    main()
