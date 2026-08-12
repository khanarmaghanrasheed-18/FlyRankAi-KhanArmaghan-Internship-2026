from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"work"/"notebooks"
def m(x): return nbf.v4.new_markdown_cell(x.strip())
def c(x): return nbf.v4.new_code_cell(x.strip())
def nb(cells):
    x=nbf.v4.new_notebook(cells=cells)
    x.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    return x

setup='''from pathlib import Path
import duckdb, numpy as np, pandas as pd
def find_root(p=Path.cwd()):
    for x in [p,*p.parents]:
        if (x/"skills"/"README.md").exists(): return x
    raise FileNotFoundError("Run inside repository")
ROOT=find_root()
PAIR_PATH=ROOT/"work"/"outputs"/"page_pair_features.parquet"
assert PAIR_PATH.exists() and PAIR_PATH.stat().st_size>0, "Run work/scripts/build_pair_features.py"
con=duckdb.connect()
R=f"read_parquet('{PAIR_PATH.as_posix()}')"
'''

feature_code=setup+'''
raw=con.sql(f"""SELECT weighted_query_overlap,smaller_page_query_coverage,
smaller_page_demand_overlap,shared_query_count,shared_impression_intersection,
mean_shared_position_gap,visibility_balance,growth_a,growth_b,same_intent,
same_content_type,rare_share_a,rare_share_b,anonymized_share_a,anonymized_share_b
FROM {R}""").df()
X=pd.DataFrame(index=raw.index)
X["log_shared_queries"]=np.log1p(raw.shared_query_count)
X["log_shared_demand"]=np.log1p(raw.shared_impression_intersection)
X["weighted_query_overlap"]=raw.weighted_query_overlap.clip(0,1)
X["smaller_page_query_coverage"]=raw.smaller_page_query_coverage.clip(0,1)
X["smaller_page_demand_overlap"]=raw.smaller_page_demand_overlap.clip(0,1)
X["position_proximity"]=np.exp(-raw.mean_shared_position_gap.clip(lower=0)/10)
X["visibility_balance"]=raw.visibility_balance.clip(0,1)
X["growth_a_missing"]=raw.growth_a.isna().astype(int)
X["growth_b_missing"]=raw.growth_b.isna().astype(int)
X["growth_a"]=raw.growth_a.clip(-2,2).fillna(0)
X["growth_b"]=raw.growth_b.clip(-2,2).fillna(0)
X["movement_gap"]=(X.growth_a-X.growth_b).abs()
X["opposite_direction"]=(X.growth_a*X.growth_b<0).astype(int)
X["shared_growth"]=np.minimum(X.growth_a,X.growth_b).clip(lower=0)
X["shared_decline"]=(-np.maximum(X.growth_a,X.growth_b)).clip(lower=0)
X["same_intent"]=raw.same_intent.fillna(0).astype(int)
X["same_content_type"]=raw.same_content_type.fillna(0).astype(int)
X["hidden_query_share"]=raw[["rare_share_a","rare_share_b",
"anonymized_share_a","anonymized_share_b"]].mean(axis=1).clip(0,1)
assert not X.isna().any().any()
X.describe().T
'''

w01=nb([
m("""# ML-02 — Research Question and Freestyle Lane

## 1. Lane and motivation

I choose Freestyle: Sibling-Page Search Interaction and Consolidation Review.

Same-site pages can share demand in several ways: one gains while a sibling declines, both grow
while demand remains fragmented, both decline with a topic-level problem, or both legitimately
serve complementary intent. The system discovers recurring page-pair patterns and never treats
overlap alone as harmful."""),
m("""## 2. Decision, action, and error cost

Decision: Which same-client pairs should an editor review first for consolidation,
differentiation, protection, topic improvement, or monitoring?

The output is a ranked public-safe queue with an observed pattern and reason codes. A human checks
intent and business purpose first. False positives can destroy useful coverage; false negatives
can leave fragmented demand unresolved. Unsupervised learning fits because no verified
cannibalization label exists."""),
m("""## 3. Research question

What recurring search-interaction patterns appear among same-client pages sharing visible query
demand, and how can they prioritize consolidation and differentiation review without treating
overlap as proof of cannibalization?

Source grain is page-query, model grain is an unordered page pair, and output is a discovered
pair archetype plus review priority."""),
c(setup+'''con.sql(f"""SELECT COUNT(*) candidate_pairs,
COUNT(DISTINCT client_hash_id) represented_clients,
MEDIAN(shared_query_count) median_shared_queries,
MEDIAN(weighted_query_overlap) median_overlap FROM {R}""").df()'''),
c(setup+'''con.sql(f"""SELECT
SUM((growth_a>0 AND growth_b>0)::INT) shared_growth,
SUM((growth_a<0 AND growth_b<0)::INT) shared_decline,
SUM((growth_a*growth_b<0)::INT) opposite_movement,
SUM((growth_a IS NULL OR growth_b IS NULL)::INT) insufficient_history
FROM {R}""").df()'''),
m("""## 4. Claim boundary

We can report measured overlap, balance, and movement and say a pair is consistent with a review
opportunity. We cannot confirm semantic equivalence, causation, or that merging improves ranking.
Every output is decision support.""")
])

w02=nb([
m("""# ML-03 — Frame the Freestyle Lane as an ML Task

## 1. Task type

This is unsupervised page-pair clustering plus within-cluster ranking. Clustering discovers
interaction patterns; it does not learn an is-cannibalizing target. Action names are assigned
only after cluster profiles are inspected."""),
m("""## 2. Target and output

There is no supervised label. The learned output is cluster_id. Names such as possible
substitution, fragmented shared demand, and complementary overlap are cautious post-hoc
interpretations. A later review score ranks evidence-rich pairs; it is not a probability."""),
m("""## 3. Success criteria

Use stability across seeds and bootstrap samples, silhouette score, materially different
profiles, sensitivity checks, and structured Top-20 review. Later compare ranking with a
transparent overlap-only baseline. A mathematically neat but non-actionable solution fails."""),
c(setup+'''pd.DataFrame({"stage":["generation","features","learning","interpretation","ranking"],
"grain":["page-query","page pair","page pair","cluster","page pair"],
"output":["shared pairs","evidence vector","cluster_id","review action","priority/reasons"]})'''),
m("""## 4. Candidate population

Development uses published non-deleted pages with at least 500 impressions, their top 50 visible
queries, queries appearing on 2–10 pages per client, and at least two shared hashes. These are
tractability and evidence-floor policies, not universal SEO truths."""),
c(setup+'''con.sql(f"""SELECT COUNT(*) pairs,COUNT(DISTINCT client_hash_id) clients,
MIN(shared_query_count) min_shared_queries,
MEDIAN(total_impressions_a+total_impressions_b) median_pair_impressions FROM {R}""").df()'''),
m("""## 5. Why ML may help

One overlap rule cannot distinguish growth, substitution, joint decline, dominance, balanced
fragmentation, and unique demand without brittle thresholds. Clustering earns its place only
when these combinations form stable, interpretable groups.""")
])

w03c=nb([
m("""# ML-04 — Sibling-Page Interaction Data Contract

## 1. Grain and windows

The query table has client-content-query grain and the content table has content grain. The model
has one unordered same-client page pair per row. Evidence uses the fixed 90-day query snapshot;
movement compares previous 30 with last 30 days. This is a current diagnostic, not prediction."""),
c(setup+'''con.sql(f"""SELECT COUNT(*) row_count,
COUNT(*)-COUNT(DISTINCT client_hash_id||'|'||content_a||'|'||content_b) duplicate_pairs,
SUM((content_a>=content_b)::INT) unordered_key_violations,
SUM((shared_query_count<2)::INT) evidence_floor_violations FROM {R}""").df()'''),
m("""## 2. Field roles

Features: weighted overlap, smaller-page coverage, overlap breadth, shared demand, position
proximity, visibility balance, recent movement, metadata agreement, and evidence-quality flags.

Context only: client/content hashes and dates. Excluded: hash identities as predictors, raw text,
deletion status as discovery evidence, provider/model fields, product flags, and future data.
There is no label."""),
c('''pd.DataFrame([
("overlap and coverage","feature","relationship strength"),
("position gap and balance","feature","fragmentation structure"),
("growth_a and growth_b","feature","current movement"),
("intent and content type","feature","coarse agreement"),
("all hashes","context","join/group only"),
("is_deleted","excluded","selection only")],
columns=["fields","role","reason"])'''),
m("""## 3. Missingness and limits

Growth is undefined when previous visible-query impressions are zero, so explicit flags replace
silent zero imputation. Rare and anonymized query detail is unavailable; measured overlap is a
lower-bound diagnostic. Meaning, text, conversions, and business purpose are absent. Actions
always mean review, never automatic merge or deletion."""),
c(setup+'''con.sql(f"""SELECT AVG((growth_a IS NULL)::INT) growth_a_missing,
AVG((growth_b IS NULL)::INT) growth_b_missing,
AVG((main_intent_a IS NULL OR main_intent_b IS NULL)::INT) intent_missing,
AVG((word_count_a IS NULL OR word_count_b IS NULL)::INT) word_count_missing FROM {R}""").df()'''),
m("""## 4. Reproducibility and safety

work/scripts/build_pair_features.py aggregates remotely into a Git-ignored Parquet cache. It
prints no credentials, private queries, URLs, client names, or raw exports. Rerunners authenticate
with hf auth login. Public output contains aggregates only.""")
])

w03l=nb([
m("""# ML-05 — Pair Feature Vector and Leakage/Privacy Check

## 1. Build the feature vector

Counts are logged, growth is clipped to limit denominator explosions, missing growth gets flags,
and scaling will occur inside the later clustering pipeline."""),
c(feature_code),
m("""## 2. Availability notes

All features belong to the fixed diagnostic window and exist when the queue is built. This is not
a future-outcome model. If later reframed as prediction, features must end before the target.
Hidden-query share records evidence quality, not absence of overlap."""),
c('''pd.DataFrame([
("hashes","excluded from X","identity/grouping only"),
("last30 and prev30","allowed now","current diagnostic; no future label"),
("published/deleted","selection only","cannot teach actions"),
("product outputs","prohibited","circular decisions"),
("raw query/title/URL","not shipped","private origin fields"),
("cluster action","post-hoc only","never fed back")],
columns=["field","treatment","reason"])'''),
m("## 3. Automated leakage and privacy assertions"),
c(feature_code+'''forbidden=("client","content_id","hash","query_id","url","title","health",
"priority","action","label")
assert not any(t in col.lower() for col in X.columns for t in forbidden)
assert np.isfinite(X.to_numpy()).all()
print(f"Checks passed: {len(X):,} vectors and {X.shape[1]} safe features.")'''),
m("""## 4. Decision limits

IDs, product outputs, and deletion status cannot become features. Cluster action names are
interpretations, not facts. No cluster automates merging, redirecting, or deletion; intent and
business-value review remain mandatory.""")
])

w04=nb([
m("""# ML-06 — Signal Audit: Do Pair-Interaction Signals Hold?

Verdicts are CONFIRMED, OPPOSITE, MIXED, or FALSE for this development cohort only."""),
m("""## 1. Distributions

Overlap and demand are heavy-tailed; use quantiles, logs, and evidence floors rather than means."""),
c(setup+'''con.sql(f"""SELECT COUNT(*) pairs,
QUANTILE_CONT(shared_query_count,[.1,.5,.9,.99]) shared_query_q,
QUANTILE_CONT(weighted_query_overlap,[.1,.5,.9,.99]) overlap_q,
QUANTILE_CONT(smaller_page_demand_overlap,[.1,.5,.9,.99]) demand_q,
QUANTILE_CONT(mean_shared_position_gap,[.1,.5,.9,.99]) position_gap_q FROM {R}""").df()'''),
m("""## 2. Signal test 1 — overlap and position proximity

Hypothesis: stronger weighted overlap generally accompanies smaller shared-query position gaps.
This is supportive structure, not proof of competition."""),
c(setup+'''t1=con.sql(f"""WITH q AS (SELECT *,NTILE(5) OVER
(ORDER BY weighted_query_overlap) oq FROM {R})
SELECT oq,COUNT(*) pairs,MEDIAN(weighted_query_overlap) median_overlap,
MEDIAN(mean_shared_position_gap) median_position_gap FROM q GROUP BY 1 ORDER BY 1""").df()
print("Verdict:","CONFIRMED" if t1.iloc[-1].median_position_gap<t1.iloc[0].median_position_gap else "OPPOSITE")
t1'''),
m("""## 3. Signal test 2 — movement is not one-dimensional

Hypothesis: shared-growth, shared-decline, and opposite-movement regimes are all substantial.
Substitution is therefore a feature, not a candidate gate."""),
c(setup+'''t2=con.sql(f"""SELECT CASE
WHEN growth_a IS NULL OR growth_b IS NULL THEN 'insufficient_history'
WHEN growth_a>0 AND growth_b>0 THEN 'shared_growth'
WHEN growth_a<0 AND growth_b<0 THEN 'shared_decline'
WHEN growth_a*growth_b<0 THEN 'opposite_movement' ELSE 'flat_or_mixed' END pattern,
COUNT(*) pairs,MEDIAN(weighted_query_overlap) median_overlap,
MEDIAN(visibility_balance) median_balance FROM {R} GROUP BY 1 ORDER BY pairs DESC""").df()
needed={"shared_growth","shared_decline","opposite_movement"}
print("Verdict:","CONFIRMED" if needed<=set(t2.loc[t2.pairs>=1000,"pattern"]) else "FALSE")
t2'''),
m("""## 4. Signal test 3 — balanced fragmentation differs from dominance

Hypothesis: evidence-rich overlap contains both balanced and dominant relationships, supporting
different editorial reviews."""),
c(setup+'''t3=con.sql(f"""WITH h AS (SELECT * FROM {R}
WHERE weighted_query_overlap>=.10 AND smaller_page_demand_overlap>=.20)
SELECT CASE WHEN visibility_balance>=.67 THEN 'balanced_fragmentation'
WHEN visibility_balance<=.33 THEN 'dominant_sibling' ELSE 'moderately_imbalanced' END relationship,
COUNT(*) pairs,MEDIAN(mean_shared_position_gap) median_position_gap,
MEDIAN(shared_impression_intersection) median_shared_demand FROM h GROUP BY 1 ORDER BY pairs DESC""").df()
print("Verdict:","CONFIRMED" if {"balanced_fragmentation","dominant_sibling"}<=set(t3.relationship) else "FALSE")
t3'''),
m("""## 5. Underlying flag-assumption test

Product flags are absent. Test the underlying assumption: raw overlap is too broad, so an
actionable review needs material overlap and position evidence."""),
c(setup+'''ft=con.sql(f"""SELECT COUNT(*) all_pairs,
SUM((weighted_query_overlap>=.10 AND smaller_page_demand_overlap>=.20
AND mean_shared_position_gap<=20)::INT) evidence_rich_pairs FROM {R}""").df()
ft["evidence_rich_share"]=ft.evidence_rich_pairs/ft.all_pairs
print("Verdict:","CONFIRMED" if ft.evidence_rich_share.iloc[0]<.5 else "MIXED")
ft'''),
m("""## 6. Practical meaning

The audit supports a broad interaction model, not an opposite-movement detector. Overlap,
proximity, balance, and movement add different information. Shared growth is not automatically
bad; shared decline may be topic weakness; pruning remains human review because page purpose and
conversion value are unavailable.""")
])

for name,obj in {
"w01_research_question.ipynb":w01,
"w02_ml_task_framing.ipynb":w02,
"w03_data_contract.ipynb":w03c,
"w03_feature_leakage_check.ipynb":w03l,
"w04_signal_audit.ipynb":w04}.items():
    nbf.write(obj,OUT/name)
    print("Wrote",name)
