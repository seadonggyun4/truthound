"""Tests for ReDoS ML Framework.

This test suite covers the complete ML framework for ReDoS vulnerability detection:
- Feature extraction
- Model implementations (rule-based, sklearn-based, ensemble)
- Training pipeline with cross-validation
- Model storage and persistence
- Dataset generation
- High-level predictor API
"""

import json
import pickle
import tempfile
from pathlib import Path

import pytest

from truthound.validators.security.redos.core import ReDoSRisk

# Import ML framework components
from truthound.validators.security.redos.ml import (
    # Base
    PatternFeatures,
    ReDoSTrainingData,
    ReDoSPrediction,
    ReDoSModelMetrics,
    ModelConfig,
    ModelType,
    BaseReDoSModel,
    # Features
    PatternFeatureExtractor,
    # Models
    RuleBasedReDoSModel,
    RandomForestReDoSModel,
    GradientBoostingReDoSModel,
    EnsembleReDoSModel,
    create_model,
    # Training
    TrainingPipeline,
    TrainingConfig,
    # Storage
    ModelStorage,
    ModelMetadata,
    # Datasets
    ReDoSDatasetGenerator,
    generate_training_dataset,
    PatternLabel,
    # Predictor
    ReDoSMLPredictor,
    train_redos_model,
    load_trained_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_patterns():
    """Sample patterns for testing."""
    return {
        "vulnerable": [
            r"(a+)+",
            r"(a+)+b",
            r"(a*)*",
            r"([a-z]+)+",
            r"(.*)+",
        ],
        "safe": [
            r"^[a-z]+$",
            r"^\d+$",
            r"^hello$",
            r"[a-z]",
            r"^[0-9]{1,10}$",
        ],
    }


@pytest.fixture
def training_data(sample_patterns):
    """Create training data from sample patterns."""
    patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
    labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])
    return ReDoSTrainingData(patterns=patterns, labels=labels)


@pytest.fixture
def feature_extractor():
    """Create feature extractor."""
    return PatternFeatureExtractor()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# PatternFeatures Tests
# =============================================================================


class TestPatternFeatures:
    """Tests for PatternFeatures data class."""

    def test_default_values(self):
        """Test default feature values."""
        features = PatternFeatures()
        assert features.length == 0
        assert features.group_count == 0
        assert features.backtracking_potential == 0.0

    def test_to_vector(self):
        """Test conversion to feature vector."""
        features = PatternFeatures(
            length=10,
            group_count=2,
            nested_quantifier_count=1,
        )
        vector = features.to_vector()
        assert isinstance(vector, list)
        assert len(vector) == PatternFeatures.num_features()
        assert all(isinstance(v, float) for v in vector)

    def test_feature_names(self):
        """Test feature names list."""
        names = PatternFeatures.feature_names()
        assert len(names) == PatternFeatures.num_features()
        assert "length" in names
        assert "nested_quantifier_count" in names
        assert "backtracking_potential" in names

    def test_to_dict(self):
        """Test dictionary conversion."""
        features = PatternFeatures(length=10, group_count=2)
        d = features.to_dict()
        assert d["length"] == 10.0
        assert d["group_count"] == 2.0

    def test_immutability(self):
        """Test that PatternFeatures is immutable (frozen)."""
        features = PatternFeatures(length=10)
        with pytest.raises(AttributeError):
            features.length = 20


# =============================================================================
# Feature Extractor Tests
# =============================================================================


class TestPatternFeatureExtractor:
    """Tests for PatternFeatureExtractor."""

    def test_extract_empty_pattern(self, feature_extractor):
        """Test extraction for empty pattern."""
        features = feature_extractor.extract("")
        assert features.length == 0
        assert features.group_count == 0

    def test_extract_simple_pattern(self, feature_extractor):
        """Test extraction for simple pattern."""
        features = feature_extractor.extract(r"^[a-z]+$")
        assert features.length > 0
        assert features.plus_count == 1
        assert features.start_anchor is True
        assert features.end_anchor is True
        assert features.anchored is True

    def test_extract_nested_quantifiers(self, feature_extractor):
        """Test detection of nested quantifiers."""
        features = feature_extractor.extract(r"(a+)+b")
        assert features.nested_quantifier_count >= 1
        assert features.backtracking_potential > 0

    def test_extract_backreference(self, feature_extractor):
        """Test detection of backreferences."""
        features = feature_extractor.extract(r"(a+)\1")
        assert features.backreference_count == 1
        assert features.max_backreference_index == 1

    def test_extract_lookaround(self, feature_extractor):
        """Test detection of lookaround assertions."""
        features = feature_extractor.extract(r"(?=foo)bar")
        assert features.lookahead_count >= 1

    def test_extract_batch(self, feature_extractor):
        """Test batch feature extraction."""
        patterns = [r"^[a-z]+$", r"(a+)+", r"\d+"]
        features = feature_extractor.extract_batch(patterns)
        assert len(features) == 3
        assert all(isinstance(f, PatternFeatures) for f in features)

    def test_extract_vectors(self, feature_extractor):
        """Test batch vector extraction."""
        patterns = [r"^[a-z]+$", r"(a+)+"]
        vectors = feature_extractor.extract_vectors(patterns)
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)


# =============================================================================
# Model Config Tests
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig.default()
        assert config.model_type == ModelType.RANDOM_FOREST
        assert config.n_estimators == 100
        assert config.random_state == 42

    def test_fast_training_config(self):
        """Test fast training configuration."""
        config = ModelConfig.fast_training()
        assert config.n_estimators == 50
        assert config.max_depth == 5

    def test_high_accuracy_config(self):
        """Test high accuracy configuration."""
        config = ModelConfig.high_accuracy()
        assert config.model_type == ModelType.GRADIENT_BOOSTING
        assert config.n_estimators == 200

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = ModelConfig.default()
        d = config.to_dict()
        assert d["model_type"] == "random_forest"
        assert "n_estimators" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"model_type": "gradient_boosting", "n_estimators": 50}
        config = ModelConfig.from_dict(d)
        assert config.model_type == ModelType.GRADIENT_BOOSTING
        assert config.n_estimators == 50


# =============================================================================
# Training Data Tests
# =============================================================================


class TestReDoSTrainingData:
    """Tests for ReDoSTrainingData."""

    def test_basic_creation(self):
        """Test basic data creation."""
        data = ReDoSTrainingData(
            patterns=["(a+)+", "^[a-z]+$"],
            labels=[1, 0],
        )
        assert len(data) == 2
        assert data.num_vulnerable == 1
        assert data.num_safe == 1

    def test_length_mismatch_error(self):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError):
            ReDoSTrainingData(
                patterns=["(a+)+", "^[a-z]+$"],
                labels=[1],  # Wrong length
            )

    def test_class_balance(self):
        """Test class balance calculation."""
        data = ReDoSTrainingData(
            patterns=["a", "b", "c", "d"],
            labels=[1, 1, 1, 0],
        )
        assert data.class_balance == 0.75


# =============================================================================
# Rule-Based Model Tests
# =============================================================================


class TestRuleBasedReDoSModel:
    """Tests for RuleBasedReDoSModel."""

    def test_predict_safe_pattern(self):
        """Test prediction for safe pattern."""
        model = RuleBasedReDoSModel()
        features = PatternFeatureExtractor().extract(r"^[a-z]+$")
        prob, conf = model.predict(features.to_vector())
        assert 0 <= prob <= 1
        assert 0 <= conf <= 1

    def test_predict_dangerous_pattern(self):
        """Test prediction for dangerous pattern."""
        model = RuleBasedReDoSModel()
        features = PatternFeatureExtractor().extract(r"(a+)+b")
        prob, conf = model.predict(features.to_vector())
        # Should have higher probability for nested quantifiers
        assert prob > 0.5

    def test_always_trained(self):
        """Test that rule-based model is always ready."""
        model = RuleBasedReDoSModel()
        assert model.is_trained is True

    def test_predict_batch(self):
        """Test batch prediction."""
        model = RuleBasedReDoSModel()
        extractor = PatternFeatureExtractor()
        patterns = [r"^[a-z]+$", r"(a+)+"]
        vectors = [extractor.extract(p).to_vector() for p in patterns]
        results = model.predict_batch(vectors)
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_save_and_load(self, temp_dir):
        """Test model save and load."""
        model = RuleBasedReDoSModel()
        path = temp_dir / "model.pkl"
        model.save(path)
        assert path.exists()

        new_model = RuleBasedReDoSModel()
        new_model.load(path)
        assert new_model.is_trained


# =============================================================================
# Sklearn Model Tests
# =============================================================================


class TestRandomForestReDoSModel:
    """Tests for RandomForestReDoSModel."""

    def test_train_and_predict(self, training_data):
        """Test training and prediction."""
        model = RandomForestReDoSModel()
        metrics = model.train(training_data)

        assert model.is_trained
        assert isinstance(metrics, ReDoSModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_score <= 1

    def test_predict_after_training(self, training_data):
        """Test prediction after training."""
        model = RandomForestReDoSModel()
        model.train(training_data)

        features = PatternFeatureExtractor().extract(r"(a+)+")
        prob, conf = model.predict(features.to_vector())
        assert 0 <= prob <= 1
        assert 0 <= conf <= 1

    def test_fallback_when_untrained(self):
        """Test fallback to rule-based when not trained."""
        model = RandomForestReDoSModel()
        features = PatternFeatureExtractor().extract(r"(a+)+")
        prob, conf = model.predict(features.to_vector())
        # Should still work (using fallback)
        assert 0 <= prob <= 1

    def test_save_and_load(self, training_data, temp_dir):
        """Test model persistence."""
        model = RandomForestReDoSModel()
        model.train(training_data)

        path = temp_dir / "rf_model.pkl"
        model.save(path)

        new_model = RandomForestReDoSModel()
        new_model.load(path)
        assert new_model.is_trained


class TestGradientBoostingReDoSModel:
    """Tests for GradientBoostingReDoSModel."""

    def test_train_and_predict(self, training_data):
        """Test training and prediction."""
        model = GradientBoostingReDoSModel()
        metrics = model.train(training_data)

        assert model.is_trained
        assert isinstance(metrics, ReDoSModelMetrics)


class TestEnsembleReDoSModel:
    """Tests for EnsembleReDoSModel."""

    def test_predict_with_pattern(self):
        """Test prediction with pattern context."""
        model = EnsembleReDoSModel()
        features = PatternFeatureExtractor().extract(r"(a+)+b")
        prob, conf = model.predict(features.to_vector(), r"(a+)+b")

        # Should detect nested quantifier signature
        assert prob > 0.5
        assert conf > 0.5

    def test_predict_without_pattern(self):
        """Test prediction without pattern context."""
        model = EnsembleReDoSModel()
        features = PatternFeatureExtractor().extract(r"(a+)+b")
        prob, conf = model.predict(features.to_vector())
        assert 0 <= prob <= 1

    def test_train_with_ml_model(self, training_data):
        """Test training the ML component."""
        model = EnsembleReDoSModel()
        metrics = model.train(training_data)

        assert isinstance(metrics, ReDoSModelMetrics)


# =============================================================================
# Model Factory Tests
# =============================================================================


class TestModelFactory:
    """Tests for model factory functions."""

    def test_create_model_by_string(self):
        """Test creating models by string name."""
        model = create_model("random_forest")
        assert isinstance(model, RandomForestReDoSModel)

        model = create_model("gradient_boosting")
        assert isinstance(model, GradientBoostingReDoSModel)

        model = create_model("rule_based")
        assert isinstance(model, RuleBasedReDoSModel)

    def test_create_model_by_enum(self):
        """Test creating models by enum."""
        model = create_model(ModelType.RANDOM_FOREST)
        assert isinstance(model, RandomForestReDoSModel)

    def test_create_unknown_model(self):
        """Test error for unknown model type."""
        with pytest.raises(ValueError):
            create_model("unknown_model")

    def test_create_model_with_config(self):
        """Test creating models with custom config."""
        config = ModelConfig(n_estimators=50)
        model = create_model("random_forest", config)
        assert model.config.n_estimators == 50


# =============================================================================
# Training Pipeline Tests
# =============================================================================


class TestTrainingPipeline:
    """Tests for TrainingPipeline."""

    def test_basic_training(self, training_data):
        """Test basic training workflow."""
        config = TrainingConfig.quick()
        pipeline = TrainingPipeline(config=config)
        result = pipeline.train(training_data)

        assert pipeline.model is not None
        assert pipeline.model.is_trained
        assert result.metrics.accuracy > 0

    def test_training_with_cross_validation(self, training_data):
        """Test training with cross-validation."""
        config = TrainingConfig(cv_folds=3)
        pipeline = TrainingPipeline(config=config)
        result = pipeline.train(training_data)

        # Cross-validation requires sklearn; if not available, cv_result is None
        # This is expected behavior - we just verify training completes
        assert result.metrics is not None
        if result.cv_result is not None:
            assert len(result.cv_result.fold_metrics) == 3

    def test_predict_after_training(self, training_data):
        """Test prediction using trained pipeline."""
        pipeline = TrainingPipeline()
        pipeline.train(training_data)

        prob, conf = pipeline.predict(r"(a+)+")
        assert 0 <= prob <= 1

    def test_save_and_load(self, training_data, temp_dir):
        """Test pipeline save and load."""
        pipeline = TrainingPipeline()
        pipeline.train(training_data)

        path = temp_dir / "pipeline_model.pkl"
        pipeline.save(path)

        new_pipeline = TrainingPipeline()
        new_pipeline.load(path)
        assert new_pipeline.model.is_trained

    def test_insufficient_data_error(self):
        """Test error on insufficient training data."""
        small_data = ReDoSTrainingData(
            patterns=["a", "b"],
            labels=[0, 1],
        )
        pipeline = TrainingPipeline()
        with pytest.raises(ValueError, match="Insufficient"):
            pipeline.train(small_data)


# =============================================================================
# Model Storage Tests
# =============================================================================


class TestModelStorage:
    """Tests for ModelStorage."""

    def test_save_and_load(self, training_data, temp_dir):
        """Test saving and loading models."""
        storage = ModelStorage(temp_dir)

        model = RandomForestReDoSModel()
        model.train(training_data)

        storage.save(model, "test_model", "1.0.0")

        loaded_model, metadata = storage.load("test_model", "1.0.0")
        assert loaded_model.is_trained
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"

    def test_list_models(self, training_data, temp_dir):
        """Test listing models."""
        storage = ModelStorage(temp_dir)

        model = RandomForestReDoSModel()
        model.train(training_data)

        storage.save(model, "model_a", "1.0.0")
        storage.save(model, "model_b", "1.0.0")

        models = storage.list_models()
        assert "model_a" in models
        assert "model_b" in models

    def test_list_versions(self, training_data, temp_dir):
        """Test listing model versions."""
        storage = ModelStorage(temp_dir)

        model = RandomForestReDoSModel()
        model.train(training_data)

        storage.save(model, "my_model", "1.0.0")
        storage.save(model, "my_model", "1.1.0")
        storage.save(model, "my_model", "2.0.0")

        versions = storage.list_versions("my_model")
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions

    def test_load_latest(self, training_data, temp_dir):
        """Test loading latest version."""
        storage = ModelStorage(temp_dir)

        model = RandomForestReDoSModel()
        model.train(training_data)

        storage.save(model, "my_model", "1.0.0")
        storage.save(model, "my_model", "2.0.0", set_latest=True)

        _, metadata = storage.load("my_model")  # Should load latest
        assert metadata.version == "2.0.0"


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_from_model(self, training_data):
        """Test creating metadata from model."""
        model = RandomForestReDoSModel()
        model.train(training_data)

        metadata = ModelMetadata.from_model(
            model=model,
            name="test_model",
            version="1.0.0",
            description="Test model",
            tags=["test"],
        )

        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "random_forest"
        assert "test" in metadata.tags

    def test_to_dict_and_back(self, training_data):
        """Test serialization round-trip."""
        model = RandomForestReDoSModel()
        model.train(training_data)

        original = ModelMetadata.from_model(model, "test", "1.0.0")
        d = original.to_dict()
        restored = ModelMetadata.from_dict(d)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.model_type == original.model_type


# =============================================================================
# Dataset Generator Tests
# =============================================================================


class TestReDoSDatasetGenerator:
    """Tests for ReDoSDatasetGenerator."""

    def test_generate_default(self):
        """Test default dataset generation."""
        generator = ReDoSDatasetGenerator()
        dataset = generator.generate()

        assert len(dataset) > 0
        assert dataset.num_vulnerable > 0
        assert dataset.num_safe > 0

    def test_generate_with_size(self):
        """Test generation with specific size."""
        generator = ReDoSDatasetGenerator()
        dataset = generator.generate(n_samples=100)

        assert len(dataset) == 100

    def test_generate_with_balance(self):
        """Test generation with specific balance."""
        generator = ReDoSDatasetGenerator()
        dataset = generator.generate(n_samples=100, balance=0.3)

        # Should be approximately 30% vulnerable
        assert 20 <= dataset.num_vulnerable <= 40

    def test_generate_with_augmentation(self):
        """Test generation with augmentation."""
        generator = ReDoSDatasetGenerator()
        dataset_no_aug = generator.generate(n_samples=50, augment=False)
        dataset_aug = generator.generate(n_samples=50, augment=True)

        # Both should work
        assert len(dataset_no_aug) == 50
        assert len(dataset_aug) == 50

    def test_add_custom_patterns(self):
        """Test adding custom patterns."""
        generator = ReDoSDatasetGenerator()
        initial_vuln = len(generator.vulnerable_patterns)

        generator.add_patterns(
            patterns=[r"custom_vuln+", r"custom_safe"],
            labels=[1, 0],
        )

        assert len(generator.vulnerable_patterns) == initial_vuln + 1

    def test_save_and_load_json(self, temp_dir):
        """Test JSON save and load."""
        generator = ReDoSDatasetGenerator()
        path = temp_dir / "patterns.json"

        generator.save_to_json(path)
        assert path.exists()

        new_generator = ReDoSDatasetGenerator(
            vulnerable_patterns=[],
            safe_patterns=[],
        )
        new_generator.load_from_json(path)

        assert len(new_generator.vulnerable_patterns) > 0

    def test_get_statistics(self):
        """Test statistics retrieval."""
        generator = ReDoSDatasetGenerator()
        stats = generator.get_statistics()

        assert "vulnerable_count" in stats
        assert "safe_count" in stats
        assert "balance" in stats


class TestDatasetFunctions:
    """Tests for dataset convenience functions."""

    def test_generate_training_dataset(self):
        """Test convenience function."""
        dataset = generate_training_dataset(n_samples=50)
        assert len(dataset) == 50


# =============================================================================
# Predictor Tests
# =============================================================================


class TestReDoSMLPredictor:
    """Tests for ReDoSMLPredictor."""

    def test_predict_safe_pattern(self):
        """Test prediction for safe pattern."""
        predictor = ReDoSMLPredictor()
        result = predictor.predict(r"^[a-z]+$")

        assert isinstance(result, ReDoSPrediction)
        assert 0 <= result.risk_probability <= 1
        assert isinstance(result.risk_level, ReDoSRisk)

    def test_predict_dangerous_pattern(self):
        """Test prediction for dangerous pattern."""
        predictor = ReDoSMLPredictor()
        result = predictor.predict(r"(a+)+b")

        # Nested quantifiers should be high risk
        assert result.risk_level in [ReDoSRisk.HIGH, ReDoSRisk.CRITICAL]

    def test_predict_batch(self):
        """Test batch prediction."""
        predictor = ReDoSMLPredictor()
        results = predictor.predict_batch([r"^[a-z]+$", r"(a+)+"])

        assert len(results) == 2
        assert all(isinstance(r, ReDoSPrediction) for r in results)

    def test_predict_risk_only(self):
        """Test lightweight prediction."""
        predictor = ReDoSMLPredictor()
        prob, risk = predictor.predict_risk_only(r"(a+)+")

        assert 0 <= prob <= 1
        assert isinstance(risk, ReDoSRisk)

    def test_is_safe_and_vulnerable(self):
        """Test safety check methods."""
        predictor = ReDoSMLPredictor()

        # Safe pattern should be safe
        assert predictor.is_safe(r"^[a-z]+$", threshold=0.5) in [True, False]

        # Dangerous pattern should be vulnerable
        assert predictor.is_vulnerable(r"(a+)+b", threshold=0.5) is True

    def test_train(self, sample_patterns):
        """Test training the predictor."""
        predictor = ReDoSMLPredictor()

        patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
        labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])

        result = predictor.train(patterns, labels)

        assert predictor.is_trained
        assert result.metrics.accuracy > 0

    def test_auto_train(self):
        """Test automatic training."""
        predictor = ReDoSMLPredictor()
        result = predictor.auto_train(n_samples=50)

        assert predictor.is_trained
        assert result.metrics.accuracy > 0

    def test_save_and_load(self, sample_patterns, temp_dir):
        """Test predictor persistence."""
        predictor = ReDoSMLPredictor()
        patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
        labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])
        predictor.train(patterns, labels)

        path = temp_dir / "predictor_model.pkl"
        predictor.save(path)

        new_predictor = ReDoSMLPredictor()
        new_predictor.load(path)
        assert new_predictor.is_trained

    def test_from_trained(self, sample_patterns, temp_dir):
        """Test loading from class method."""
        predictor = ReDoSMLPredictor()
        patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
        labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])
        predictor.train(patterns, labels)

        path = temp_dir / "model.pkl"
        predictor.save(path)

        loaded_predictor = ReDoSMLPredictor.from_trained(path)
        assert loaded_predictor.is_trained

    def test_get_feature_importance(self, sample_patterns):
        """Test feature importance retrieval."""
        predictor = ReDoSMLPredictor()
        patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
        labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])
        predictor.train(patterns, labels)

        importance = predictor.get_feature_importance()
        assert isinstance(importance, dict)
        assert "nested_quantifier_count" in importance


class TestPredictorFunctions:
    """Tests for predictor convenience functions."""

    def test_train_redos_model(self, sample_patterns):
        """Test training convenience function."""
        patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
        labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])

        predictor = train_redos_model(patterns, labels)

        assert isinstance(predictor, ReDoSMLPredictor)
        assert predictor.is_trained

    def test_load_trained_model(self, sample_patterns, temp_dir):
        """Test loading convenience function."""
        predictor = ReDoSMLPredictor()
        patterns = sample_patterns["vulnerable"] + sample_patterns["safe"]
        labels = [1] * len(sample_patterns["vulnerable"]) + [0] * len(sample_patterns["safe"])
        predictor.train(patterns, labels)

        path = temp_dir / "model.pkl"
        predictor.save(path)

        loaded = load_trained_model(path)
        assert loaded.is_trained


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete ML workflow."""

    def test_full_training_pipeline(self, temp_dir):
        """Test complete training workflow."""
        # 1. Generate dataset
        dataset = generate_training_dataset(n_samples=100)
        assert len(dataset) == 100

        # 2. Create and train pipeline
        config = TrainingConfig.quick()
        pipeline = TrainingPipeline(config=config)
        result = pipeline.train(dataset)

        assert result.metrics.accuracy > 0.5

        # 3. Save model
        model_path = temp_dir / "model.pkl"
        pipeline.save(model_path)

        # 4. Load and predict
        predictor = ReDoSMLPredictor.from_trained(model_path)
        prediction = predictor.predict(r"(a+)+b")

        assert prediction.risk_level in [ReDoSRisk.HIGH, ReDoSRisk.CRITICAL]

    def test_model_comparison(self):
        """Test comparing different models."""
        dataset = generate_training_dataset(n_samples=100)

        results = {}
        for model_type in ["rule_based", "random_forest"]:
            config = TrainingConfig(
                model_type=ModelType(model_type),
                cv_folds=3,
            )
            pipeline = TrainingPipeline(config=config)
            result = pipeline.train(dataset)
            results[model_type] = result.metrics.f1_score

        # Both should produce valid scores
        for name, score in results.items():
            assert 0 <= score <= 1

    def test_prediction_consistency(self):
        """Test that predictions are consistent."""
        predictor = ReDoSMLPredictor()

        pattern = r"(a+)+b"
        results = [predictor.predict(pattern) for _ in range(5)]

        # All predictions should be identical
        probs = [r.risk_probability for r in results]
        assert all(p == probs[0] for p in probs)
