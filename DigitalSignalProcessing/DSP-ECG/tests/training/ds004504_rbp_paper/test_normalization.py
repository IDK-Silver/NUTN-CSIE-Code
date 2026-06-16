import numpy as np
import pytest

from ecg.training.ds004504_rbp_paper import fit_paper_min_max, transform_min_max


def test_paper_min_max_uses_full_feature_columns() -> None:
    features = np.asarray([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]], dtype=np.float32)

    params = fit_paper_min_max(features)
    normalized = transform_min_max(features, params)

    np.testing.assert_array_equal(params.feature_min, np.asarray([2.0, 10.0], dtype=np.float32))
    np.testing.assert_array_equal(params.feature_max, np.asarray([6.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(
        normalized,
        np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
    )


def test_paper_min_max_rejects_constant_feature() -> None:
    with pytest.raises(ValueError, match="constant feature"):
        fit_paper_min_max(np.asarray([[1.0, 2.0], [1.0, 3.0]], dtype=np.float32))
