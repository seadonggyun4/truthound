# ML-based Type Inference

`truthound.profiler.ml_inference` combines column names, sampled values, statistics
and table context to suggest a native `DataType`. This local inference subsystem
does not call an external AI service. Model confidence is a score, not a calibrated
probability guarantee.

## Inference and Configuration

```python
import polars as pl
from truthound.profiler.ml_inference import MLTypeInferrer, InferrerConfig

inferrer = MLTypeInferrer(config=InferrerConfig(
    model="ensemble", confidence_threshold=0.5,
    use_caching=True, cache_size=1000, enable_learning=True,
    feedback_batch_size=100,
))
result = inferrer.infer(
    pl.Series("email", ["a@example.test", "b@example.test"]),
    {"table_name": "contacts"},
)
print(result.inferred_type.value, result.confidence)
```

An explicit `model=` object/name overrides `config.model`; otherwise the config
selects the model. `infer(column, context=None)` copies context before adding column
information. `infer_table(frame, table_name="")` supplies neighboring-column
context and returns a dictionary keyed by column. Scores below
`confidence_threshold` produce `DataType.UNKNOWN`, retaining the best candidates
in `alternatives`. Invalid or nonfinite prediction scores fail.

`InferenceResult` contains `column_name`, `inferred_type`, `confidence`,
`alternatives`, `reasoning`, `features_used`, `model_version` and
`inference_time_ms`. `to_dict()` converts enum values and alternative predictions.
Convenience functions are `infer_column_type_ml(column, context=None,
model="ensemble")` and `infer_table_types_ml(frame, table_name="", model="ensemble")`.

## Feature Extraction

`Feature` contains name, numeric value, `FeatureType`, importance and metadata.
Feature types are `NAME_BASED`, `VALUE_BASED`, `STATISTICAL`, `CONTEXTUAL` and
`PATTERN_BASED`. An extractor's `extract(series, context)` returns a list of
features; `FeatureExtractorRegistry.extract_all()` returns a `FeatureVector`
with `to_array()`, `to_dict()` and `get_feature(name)`.

The default registry includes `NameFeatureExtractor`, `ValueFeatureExtractor`,
`StatisticalFeatureExtractor` and `ContextFeatureExtractor`. Native feature errors
and invalid/nonfinite features fail the operation instead of returning a partial
success. Custom extractors and models are trusted application code.

## Models and Cache

`create_inference_model("rule_based" | "naive_bayes" | "ensemble")` returns a
native model. `RuleBasedModel` uses weighted feature rules and does not learn from
feedback. `NaiveBayesModel` learns Gaussian feature likelihoods and class priors;
before training it falls back to rule-based prediction. `EnsembleModel` combines
weighted predictions. Weights must be finite, nonnegative and have a positive sum.
Ensemble training prepares child copies before replacing the model list.

The bounded process-local cache includes complete column content, context, model
state and confidence threshold. It returns copies. Custom models/extractors and
unsupported object values bypass implicit caching. `cache_size=0` or
`use_caching=False` disables it; `clear_cache()` removes entries. It has no disk
cache or TTL. Inference and feedback on one inferrer serialize through its lock.

## Feedback and Persistence

```python
from truthound.profiler.base import DataType
inferrer.provide_feedback(pl.Series("count", [1, 2]), DataType.INTEGER)
```

When learning is enabled, feedback accumulates until `feedback_batch_size`
(default 100, range 1–10000), then trains the model and clears inference cache.
Disabled learning leaves the model unchanged. `model_path` selects a local JSON
snapshot: construction loads it when present, and completed feedback batches save
it. Unflushed feedback is process-local and is lost on restart.

Built-in models implement `save(path)` and `load(path)` with bounded JSON and no
pickle/import execution. Rule parameters, Naive Bayes state and nested ensemble
children are preserved. Loading validates the entire snapshot before changing the
model; legacy Naive Bayes JSON is accepted. Snapshots are limited to 16 MiB, nesting
to 8 levels and each ensemble to 32 children. Save uses a private sibling file,
atomic replacement and file/directory flush. Parent directories must exist.
Custom models must explicitly implement their own persistence.

Cooperating inferrers lock their selected model path and check the observed file
digest before learning. A stale owner refuses to overwrite a newer snapshot.
Training/save failures retain the pending feedback and fence further inference or
learning on that owner. Reopen and inspect the observed persisted model; a failure
after publication can leave new state on disk. This does not provide exactly-once
feedback receipts or coordination with arbitrary non-cooperating writers.

## Approved MCP Inference

A private `ProfilerRuntime.inferrers` map binds native `MLTypeInferrer` objects;
`inference_contexts` binds copied private contexts. `inspect` exposes aliases.
`inference_infer(inferrer_id, asset_id, column, inference_context_id?)` and
`inference_table(inferrer_id, asset_id)` use pinned Arrow input. Results contain
column/type/confidence/alternatives, omitting private context, feature diagnostics,
model paths and model versions.

`inference_feedback(inferrer_id, asset_id, column, true_type,
inference_context_id?)` delegates to native learning and reports whether learning
is enabled and the pending feedback count. It may persist a completed batch to the
factory-selected path. `inference_cache_clear(inferrer_id)` clears the selected
cache. These calls require exact approval and sensitive-result consent. Wire
requests cannot choose paths, import models or send callback code. Installed-worker
verification covers native local model learning and fresh-owner restore.

## Next Steps

- [Sampling](sampling.md)
- [Quality Scoring](quality-scoring.md)
- [Visualization](visualization.md)

## Native sklearn classifier models

The separate `truthound.profiler.ml` API accepts `TrainingData(features, labels,
feature_names, weights=...)` and `train_type_classifier(data, model_type, config)`.
It provides `random_forest`, `gradient_boosting`, `naive_bayes`,
`logistic_regression`, `svm`, `neural_network` and a soft-voting `ensemble`
(forest, logistic regression and Gaussian Naive Bayes). The `ml` extra requires
scikit-learn 1.7 or newer for weighted neural-network fitting. `rule_based`
predicts without sklearn; its `train` evaluates supplied examples rather than
learning rules or inventing accuracy. Unknown model names and untrained ML
predictions fail instead of selecting another model.

Features must be finite, with unique names and aligned labels, optional sample IDs
and finite nonnegative weights with positive total. At least two classes and at
least `cross_validation_folds` examples per class are required. Native stratified
holdout constraints also apply. Feature selection uses native `SelectKBest` up to
`max_features`, fitted independently inside each evaluation fold. Metrics come
from the holdout; cross-validation scores are measured separately; the published
candidate then fits all rows. Evaluation uses effective sample weights. A failed
fit or cross-validation leaves the previous estimator, labels and metrics intact.

Tree settings apply to forest/boosting; `n_jobs` applies to forest and ensemble;
`early_stopping` applies to boosting and neural networks. Class balancing uses
native class weights or weighted fitting. Other models do not acquire tree or
thread settings they cannot implement. Configuration is copied at construction;
selecting an explicit classifier class chooses its backend. Native sklearn
warnings remain visible, including degenerate-feature and convergence warnings.

Rule models save validated JSON, including feature names and measured metrics.
Sklearn models save native **trusted Python pickle** with backend identity,
configuration, metrics and feature schema. Load only operator-owned trusted files;
this format can execute Python during deserialization and is not a portable
cross-version model format. Legacy files without the current format marker fail.
Writes use atomic replacement, and invalid loads leave the live model intact.
These sklearn artifacts are distinct from the bounded JSON snapshots of
`MLTypeInferrer` above.

`PretrainedModelManager.load_default()` binds extracted feature names to native
rules without mutating the shared rule object. Its zero accuracy means unevaluated.
`train_and_cache(name, n_samples_per_type=500, cache=True)` trains from seeded
synthetic values, with at least 250 values per type. This is synthetic evaluation,
not a claim about real datasets. The generator owns its RNG, including UUIDs.
Names are bounded plain identifiers. `cache=False` neither reads nor writes a cache;
otherwise the operator-owned trusted pickle cache lives in `DEFAULT_CACHE_DIR`.
Corrupt or differently configured caches fail visibly; use a new name to train a
new configuration. Publication is atomic.

An approved profiler factory can put `NativeClassifierBinding(model,
confidence_threshold=0.0, sample_size=1000)` from
`truthound.mcp.profiler_classifier` in `ProfilerRuntime.inferrers`. Existing
`inference_infer` and `inference_table` then return native predictions with named
features. The binding owns a model copy and has no implicit cache; clearing it is
a no-op. Sklearn prediction requires the trained feature schema. Training and
trusted model-file loading are factory-owned; this binding rejects
`inference_feedback` with `unsupported_operation`. Wire requests cannot supply
pickle files, training callbacks or file paths. Installed MCP verification covers
the native rule binding; native sklearn lifecycle tests cover all seven models.
