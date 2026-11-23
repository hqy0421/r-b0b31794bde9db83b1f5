"""Tests for equivalence-group handling in NARX term selection."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fastcan import check_group_equivalence, fill_group_missing
from fastcan.narx import make_narx


def _make_duplicate_channel_data(n_samples=400, max_delay=3, seed=12345):
    """A NARX system whose input u0 is recorded by two duplicate channels."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.05, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    # Second channel is an affine duplicate of the first.
    u0_dup = 2.0 * u0 + 0.5
    u1 = rng.normal(0, 0.1, n_samples)
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = 0.5 * y[i - 1] + 0.7 * u0[i - 1] + 1.0
    y = y[max_delay:] + e
    X = np.c_[u0[max_delay:], u0_dup[max_delay:], u1]
    groups = ["u0", "u0", "u1"]
    return X, y, groups, max_delay


@pytest.mark.parametrize("lazy", [False, True])
def test_make_narx_groups_dedup_term_budget(lazy):
    X, y, groups, max_delay = _make_duplicate_channel_data()
    narx = make_narx(
        X=X,
        y=y,
        n_terms_to_select=3,
        max_delay=max_delay,
        poly_degree=2,
        groups=groups,
        lazy=lazy,
        batch_size=8,
        verbose=0,
        refine_verbose=0,
    )
    # Raw-channel bookkeeping is attached to the returned model.
    assert_array_equal(narx.group_ids_, [0, 0, 1])
    assert narx.group_labels_.tolist() == ["u0", "u1"]
    assert_array_equal(narx.group_representatives_, [0, 2])
    assert [m.tolist() for m in narx.group_members_] == [[0, 1], [2]]
    assert narx.X_train_.shape == X.shape

    # No selected term may use the duplicate (non-representative) channel 1.
    input_feat_ids = narx.feat_ids[narx.feat_ids < X.shape[1]]
    assert not np.isin(input_feat_ids, [1]).any()

    # The model remains trainable and predictive.
    narx.fit(X, y)
    pred = narx.predict(X)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("lazy", [False, True])
def test_make_narx_groups_reduce_candidates(lazy):
    # Grouping the duplicate channels yields the same physical structure as
    # removing the duplicate column before building the term library.
    X, y, groups, max_delay = _make_duplicate_channel_data()
    narx_grouped = make_narx(
        X=X,
        y=y,
        n_terms_to_select=3,
        max_delay=max_delay,
        poly_degree=2,
        groups=groups,
        lazy=lazy,
        batch_size=8,
        verbose=0,
        refine_verbose=0,
    )
    narx_reduced = make_narx(
        X=X[:, [0, 2]],
        y=y,
        n_terms_to_select=3,
        max_delay=max_delay,
        poly_degree=2,
        lazy=lazy,
        batch_size=8,
        verbose=0,
        refine_verbose=0,
    )
    # Same number of terms, same output assignment, same delay pattern.
    assert narx_grouped.feat_ids.shape == narx_reduced.feat_ids.shape
    assert_array_equal(narx_grouped.delay_ids, narx_reduced.delay_ids)
    assert_array_equal(narx_grouped.output_ids, narx_reduced.output_ids)
    # The grouped model never selects the duplicate (non-representative) input.
    n_features_in = X.shape[1]
    grouped_input_ids = narx_grouped.feat_ids[narx_grouped.feat_ids < n_features_in]
    assert not np.isin(grouped_input_ids, [1]).any()

    # Predictive equivalence on the representative channels.
    narx_grouped.fit(X, y)
    narx_reduced.fit(X[:, [0, 2]], y)
    assert_allclose(
        narx_grouped.predict(X),
        narx_reduced.predict(X[:, [0, 2]]),
        atol=1e-8,
    )


def test_make_narx_groups_length_validation():
    X, y, _, max_delay = _make_duplicate_channel_data()
    with pytest.raises(ValueError, match="length of `groups`"):
        make_narx(
            X=X,
            y=y,
            n_terms_to_select=2,
            max_delay=max_delay,
            poly_degree=1,
            groups=["u0", "u0"],  # one label short
            verbose=0,
        )


def test_narx_fill_group_missing_roundtrip():
    X, y, groups, max_delay = _make_duplicate_channel_data()
    narx = make_narx(
        X=X,
        y=y,
        n_terms_to_select=3,
        max_delay=max_delay,
        poly_degree=2,
        groups=groups,
        verbose=0,
        refine_verbose=0,
    )
    narx.fit(X, y)

    # The representative channel drops out at prediction time.
    X_missing = X.copy()
    X_missing[:, 0] = np.nan

    report = check_group_equivalence(narx, narx.X_train_, r2_tol=0.9)
    assert report["valid"][1]
    assert report["r2"][1] > 0.99

    X_filled = fill_group_missing(narx, X_missing, r2_tol=0.9)
    assert np.isfinite(X_filled[:, 0]).all()
    assert_allclose(X_filled[:, 0], X[:, 0], atol=1e-6)

    pred_repaired = narx.predict(X_filled)
    pred_original = narx.predict(X)
    # The substitution must reproduce the original-channel predictions.
    assert_allclose(pred_repaired, pred_original, atol=1e-6)


def test_make_narx_groups_multi_output():
    rng = np.random.default_rng(7)
    n_samples, max_delay = 300, 2
    u = rng.normal(size=n_samples)
    X = np.column_stack([u, 1.5 * u - 0.5])
    groups = ["u", "u"]
    y = np.column_stack(
        [
            u + rng.normal(scale=0.05, size=n_samples),
            -0.5 * u + rng.normal(scale=0.05, size=n_samples),
        ]
    )
    narx = make_narx(
        X=X,
        y=y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=1,
        groups=groups,
        verbose=0,
        refine_verbose=0,
    )
    # One term per output and both outputs represented.
    assert_array_equal(narx.output_ids, [0, 0, 1, 1])
    input_ids = narx.feat_ids[narx.feat_ids < X.shape[1]]
    assert not np.isin(input_ids, [1]).any()
    narx.fit(X, y)
    pred = narx.predict(X)
    assert pred.shape == (n_samples, 2)
    assert np.isfinite(pred).all()


def test_narx_without_groups_has_no_group_attrs():
    X, y, _, max_delay = _make_duplicate_channel_data()
    narx = make_narx(
        X=X,
        y=y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=1,
        verbose=0,
        refine_verbose=0,
    )
    assert not hasattr(narx, "group_ids_")
    assert not hasattr(narx, "X_train_")
