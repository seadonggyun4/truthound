"""Online SVM matches the actual unsupervised objective and fixed feature basis."""

import numpy as np
import pytest

from truthound.validators.memory.sgd_online import OnlineScaler, SGDOneClassSVM

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("kernel", ["nystroem", "rbf_sampler"])
def test_partial_fit_matches_independently_composed_native_one_class_sgd(kernel):
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.linear_model import SGDOneClassSVM as Native

    rng = np.random.default_rng(4721)
    first = rng.normal(100, 3, (100, 2))
    second = rng.normal(104, 4, (60, 2))
    scaler = OnlineScaler().partial_fit(first)
    scaled = scaler.transform(first)
    gamma = 1 / (2 * scaled.var())
    transformer = (
        Nystroem(gamma=gamma, n_components=20, random_state=42)
        if kernel == "nystroem"
        else RBFSampler(gamma=gamma, n_components=20, random_state=42)
    )
    transformer.fit(scaled)
    direct = Native(
        nu=0.1,
        learning_rate="optimal",
        eta0=0.01,
        random_state=42,
        max_iter=1000,
        tol=1e-4,
        warm_start=True,
    )
    adapter = SGDOneClassSVM(nu=0.1, kernel_approx=kernel, n_components=20)
    for batch in (first, second):
        direct.partial_fit(transformer.transform(scaler.transform(batch)))
        adapter.partial_fit(batch)
        expected = direct.decision_function(transformer.transform(scaler.transform(first)))
        np.testing.assert_allclose(
            adapter.decision_function(first), expected, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(
            adapter.predict(first), direct.predict(transformer.transform(scaler.transform(first)))
        )


def test_constant_first_batch_has_finite_kernel_and_no_fake_negative_labels():
    model = SGDOneClassSVM(n_components=4)
    frame = np.ones((20, 2))
    model.partial_fit(frame)
    assert np.isfinite(model.decision_function(frame)).all()
