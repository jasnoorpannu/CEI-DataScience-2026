from __future__ import annotations

import numpy as np

from src.calibration import (
    COMPONENT_NAMES,
    fit_logistic_weights,
    grid_search_weights,
    load_calibrated_weights,
    ndcg_at_k,
    pairwise_accuracy,
    save_calibrated_weights,
)


def _data():
    rng = np.random.default_rng(42)
    n = 200
    components = np.column_stack(
        [rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)]
    )
    score = 0.5 * components[:, 0] + 0.3 * components[:, 1] + 0.2 * components[:, 2] + rng.normal(0, 0.2, n)
    outcomes = (score > np.median(score)).astype(int)
    return components, outcomes


def test_fit_logistic_weights():
    components, outcomes = _data()
    cal = fit_logistic_weights(components, outcomes)
    assert set(cal.weights.keys()) == set(COMPONENT_NAMES)
    assert abs(sum(cal.weights.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in cal.weights.values())
    assert cal.metrics["roc_auc"] > 0.5


def test_grid_search_weights():
    components, outcomes = _data()
    cal = grid_search_weights(components, outcomes)
    assert cal.metrics["roc_auc"] >= 0.5


def test_ndcg():
    assert ndcg_at_k([3, 2, 1], [1, 1, 0], k=3) > 0
    assert ndcg_at_k([1, 2, 3], [1, 0, 0], k=3) <= 1.0


def test_pairwise_accuracy():
    assert pairwise_accuracy([1.0, 0.0, 0.5], [1, 0, 1]) == 1.0


def test_save_load_roundtrip(tmp_path):
    components, outcomes = _data()
    cal = fit_logistic_weights(components, outcomes)
    path = save_calibrated_weights(cal, tmp_path / "weights.json")
    loaded = load_calibrated_weights(path)
    assert loaded is not None
    assert loaded.weights == cal.weights
    assert loaded.method == cal.method
