# Quality Scoring

`truthound.profiler.quality` evaluates bound validation rules against a Polars
`DataFrame`. A positive prediction means **the value passes the rule**. Ground
truth uses the same convention: a non-null Boolean series aligned to every input
row. Sampling applies the same selected indices to data and labels.

## RuleQualityScorer

```python
import polars as pl
from truthound.profiler.quality import RuleQualityScorer, RuleType, ValidationRule

frame = pl.DataFrame({"amount": [10, -1, 20]})
rule = ValidationRule("nonnegative", RuleType.RANGE, column="amount",
                      min_value=0, nullable=False)
scorer = RuleQualityScorer(
    estimator="sampling",
    estimator_options={"sample_size": 1000, "random_seed": 7,
                       "confidence_level": 0.95},
    min_confidence=0.5, quality_threshold=0.70, max_cache_entries=128,
)
score = scorer.score(rule, frame, ground_truth=pl.Series([True, False, True]))
print(score.metrics.f1_score)
print(score.metrics.quality_level.value)
```

`score()` returns `RuleQualityScore`: rule name/type/column, `metrics`,
`recommendation`, `should_use` and optional alternatives. `metrics` is
`QualityMetrics`, including precision, recall, F1, accuracy, specificity, MCC,
confidence intervals, sample/population sizes and evaluation time.
`ConfusionMatrix` fields are `true_positives`, `true_negatives`, `false_positives`
and `false_negatives`. Zero-denominator ratios return zero.

`ValidationRule.validate_column()` follows scalar `validate()` semantics,
including nullability, custom callbacks and pattern prefix matching. A custom
callback must return a Boolean. Custom rule code is trusted application code.

## QualityLevel

F1 thresholds are excellent ≥0.95, good ≥0.85, acceptable ≥0.70, poor ≥0.50,
and unacceptable below 0.50. The scorer's acceptance threshold and minimum
confidence affect recommendations; they do not redefine these levels.

## Quality Estimation Strategies

- `SamplingQualityEstimator(sample_size=1000, confidence_level=0.95,
  random_seed=None)` uses a local RNG, leaving process-global random state intact.
  Its configured confidence level controls Wilson precision/recall intervals.
  Labels must cover the entire input, even when only a sample is evaluated.
- `CrossValidationEstimator(n_folds=5, random_seed=None)` uses at most one fold per
  row. With labels it combines observed fold confusion counts; without labels it
  estimates quality from consistency. Empty input returns empty metrics.
- `HeuristicQualityEstimator(strictness=0.5)` estimates quality from data/rule
  characteristics. It does not establish supervised accuracy.

Without labels, sampling and cross-validation metrics are heuristic estimates,
not measured classification accuracy. `confidence` is not a guarantee that a rule
is correct. No estimator produces an F1 confidence interval; `f1_ci` retains its
empty default. `estimate_quality(rule, frame, estimator="sampling", **options)`
returns metrics without labels; use an estimator's `estimate()` or scorer's
`score()` for explicit ground truth.

## Rule Comparison and Cache

`scorer.score_all(rules, frame, ground_truth=None)` returns scores in input order.
`scorer.compare(rules, frame)` and `compare_rules(rules, frame)` return a list
sorted by descending F1. They do not return a report or a best-rule wrapper.

The bounded process-local cache includes complete data, labels, rule parameters,
estimator settings and recommendation thresholds. Custom estimators, custom rule
callbacks and unsupported object values bypass automatic caching. Cached scores
are copied so caller mutation cannot alter future results. `use_cache=False`
bypasses it; `max_cache_entries=0` disables it; `clear_cache()` removes entries.
Caches have no persistent restore or implicit TTL.

## Trend Analysis

`QualityTrendAnalyzer(storage_path=None)` records metrics with
`record(rule_name, metrics, timestamp=None, data_size=0, notes="")`.
`analyze_trend(rule_name, window_days=30)` returns a dictionary; `get_history` returns
measurement dictionaries. At most 100 points per rule are retained. This is a
measurement history, not an alerting service or a forecast model.

## Approved MCP Quality Operations

A private `ProfilerRuntime` factory binds `quality_scorers` to native
`RuleQualityScorer` objects and `quality_rules` to native `ValidationRule` objects.
`inspect` lists aliases. `quality_score(scorer_id, rule_ids=[one_alias], asset_id)`
and `quality_compare(scorer_id, rule_ids, asset_id)` use pinned Arrow data, with
optional `label_column` and `use_cache`. Results contain rule aliases, metrics and
acceptance decisions; private rule names/callbacks/configuration are omitted.
`quality_cache_clear(scorer_id)` clears the selected scorer. These commands require
exact approval and sensitive-result consent. Rule lists are unique and limited to
100 entries. Factory configuration owns sampling and estimator options.

## Next Steps

- [Threshold Tuning](threshold-tuning.md)
- [ML Inference](ml-inference.md)

Persistent trend snapshots use a cooperating native file lock and atomic sibling
publication with file/directory flush. Each writer reloads the latest snapshot
before appending. Reopen restores enums, timestamps, intervals and confusion
counts. Malformed or oversized snapshots fail rather than becoming empty history.
A failed write fences that owner from further recording; open a fresh owner to
inspect the observed file before deciding whether another measurement is needed.
A failure after publication can leave the new measurement present. This is not
an exactly-once receipt protocol. The bound is 4096 rules, 100 points per rule and
16 MiB per snapshot; directories must already exist. Timestamps are sorted in UTC.
`get_history(limit=0)` is empty, and negative limits are rejected.

Factories may bind native `QualityTrendAnalyzer` objects as `quality_trends`.
`quality_record` uses the same selected scorer/rule/input as `quality_score`, plus
`trend_id`, and records the native result. `quality_history(trend_id, rule_ids,
limit=50)` and `quality_trend(trend_id, rule_ids, window_days=30)` require exactly
one rule alias. History omits private notes; storage paths remain factory-owned.
