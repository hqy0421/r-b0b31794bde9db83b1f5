"""Tests for equivalence-group (duplicate channel) representative selection."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fastcan import (
    FastCan,
    LazyFastCan,
    check_group_equivalence,
    fill_group_missing,
)
from fastcan._group import alternate_mask, normalize_groups


def _make_rig(n_samples=200, seed=0):
    """Two physical signals, each recorded by two (nearly) collinear channels."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n_samples)
    v = rng.normal(size=n_samples)
    noise = rng.normal(scale=1e-8, size=(n_samples, 4))
    # Channels: u, u' (duplicate), v, v' (duplicate)
    X = np.column_stack([u, 2.0 * u + 1.0, v, -v + 3.0]) + noise
    y = u + 0.5 * v + rng.normal(scale=0.05, size=n_samples)
    groups = ["sig_u", "sig_u", "sig_v", "sig_v"]
    return X, y, groups


def test_normalize_groups_deterministic_representative():
    group_ids, labels, reps, members = normalize_groups(
        ["b", "b", "a", "a", "a"], n_features=5
    )
    assert_array_equal(group_ids, [0, 0, 1, 1, 1])
    assert labels.tolist() == ["b", "a"]
    assert_array_equal(reps, [0, 2])
    assert [m.tolist() for m in members] == [[0, 1], [2, 3, 4]]
    assert_array_equal(
        alternate_mask(group_ids, reps), [False, True, False, True, True]
    )


def test_normalize_groups_validations():
    with pytest.raises(ValueError, match="length of `groups`"):
        normalize_groups([0, 0], n_features=3)
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_groups([[0, 1]], n_features=2)
    # Excluding a representative leaves the group unselectable.
    with pytest.raises(ValueError, match="representative"):
        normalize_groups([0, 0, 1], n_features=3, indices_exclude=np.array([0]))
    # Two inclusions from one group.
    with pytest.raises(ValueError, match="same equivalence group"):
        normalize_groups([0, 0, 1], n_features=3, indices_include=np.array([0, 1]))
    # Inclusion of an alternate (non-representative) member.
    with pytest.raises(ValueError, match="representatives"):
        normalize_groups([0, 0, 1], n_features=3, indices_include=np.array([1]))


@pytest.mark.parametrize("beam_width", [1, 2])
def test_fastcan_duplicates_consume_one_slot(beam_width):
    X, y, groups = _make_rig()
    selector = FastCan(
        n_features_to_select=2,
        groups=groups,
        beam_width=beam_width,
        verbose=0,
    ).fit(X, y)
    # Exactly two distinct groups, representatives are lowest-index members.
    assert_array_equal(np.sort(selector.indices_), [0, 2])
    assert selector.indices_.shape[0] == 2
    selected_groups = selector.group_ids_[selector.indices_]
    assert np.unique(selected_groups).size == 2
    assert_array_equal(selector.support_.nonzero()[0], [0, 2])


def test_fastcan_group_budget_error():
    X, y, groups = _make_rig()
    with pytest.raises(ValueError, match="available equivalence groups"):
        FastCan(n_features_to_select=3, groups=groups, verbose=0).fit(X, y)


def test_fastcan_group_beam_width_error():
    X, y, groups = _make_rig()
    with pytest.raises(ValueError, match="beam_width should be <= the number"):
        FastCan(
            n_features_to_select=2,
            groups=groups,
            beam_width=3,
            verbose=0,
        ).fit(X, y)


def test_fastcan_groups_match_ungrouped_on_representatives():
    # Selection over representatives must match an explicitly reduced matrix.
    X, y, groups = _make_rig()
    grouped = FastCan(2, groups=groups, verbose=0).fit(X, y)
    rep_positions = np.array([0, 2])
    reduced = FastCan(2, verbose=0).fit(X[:, rep_positions], y)
    assert_array_equal(grouped.indices_, rep_positions[reduced.indices_])
    np.testing.assert_allclose(grouped.scores_, reduced.scores_, rtol=1e-6)


def test_fastcan_group_include_representative():
    X, y, groups = _make_rig()
    # Pre-include the representative of the first group.
    selector = FastCan(
        n_features_to_select=2,
        groups=groups,
        indices_include=[0],
        verbose=0,
    ).fit(X, y)
    assert selector.indices_[0] == 0
    assert_array_equal(np.sort(selector.indices_), [0, 2])


def test_fastcan_without_groups_unchanged():
    X, y, _ = _make_rig()
    selector = FastCan(2, verbose=0).fit(X, y)
    assert selector.group_ids_ is None
    assert not hasattr(selector, "X_train_")


def test_lazy_fastcan_groups_skip_alternates():
    X, y, groups = _make_rig()
    selector = LazyFastCan(2, groups=groups).fit(X, y)
    assert_array_equal(np.sort(selector.indices_), [0, 2])
    # Alternates must never appear in a selection.
    assert not np.isin(selector.indices_, [1, 3]).any()
    np.testing.assert_array_equal(selector.group_representatives_, np.array([0, 2]))


def test_lazy_fastcan_group_budget_error():
    X, y, groups = _make_rig()
    with pytest.raises(ValueError, match="equivalence groups"):
        LazyFastCan(3, groups=groups).fit(X, y)


def test_check_group_equivalence_report():
    X, y, groups = _make_rig()
    selector = FastCan(2, groups=groups, verbose=0).fit(X, y)
    report = check_group_equivalence(selector, X, r2_tol=0.9)
    assert report["r2"].shape == (4,)
    # Representatives are trivially valid; alternates fit affine mappings.
    assert report["valid"].all()
    assert report["r2"][1] > 0.99
    assert report["r2"][3] > 0.99
    np.testing.assert_allclose(report["coef"][1], 2.0, rtol=1e-4)
    np.testing.assert_allclose(report["intercept"][1], 1.0, atol=1e-4)


def test_check_group_equivalence_failure():
    X, y, groups = _make_rig()
    selector = FastCan(2, groups=groups, verbose=0).fit(X, y)
    X_bad = X.copy()
    # Break the duplicate relationship of channel 1 with random content.
    rng = np.random.default_rng(1)
    X_bad[:, 1] = rng.normal(size=X.shape[0])
    with pytest.raises(ValueError, match="not an admissible substitute"):
        check_group_equivalence(selector, X_bad, r2_tol=0.95)


def test_fill_group_missing_reconstructs_representative():
    X, y, groups = _make_rig()
    selector = FastCan(2, groups=groups, verbose=0).fit(X, y)
    X_pred = X.copy()
    X_pred[:, 0] = np.nan  # representative of sig_u drops out
    X_filled, _report = fill_group_missing(
        selector, X_pred, r2_tol=0.9, return_report=True
    )
    assert np.isfinite(X_filled[:, 0]).all()
    # Reconstruction of u from u' = 2u + 1 should recover u.
    np.testing.assert_allclose(X_filled[:, 0], X[:, 0], atol=1e-5)
    # Other columns untouched.
    np.testing.assert_array_equal(X_filled[:, 1:], X_pred[:, 1:])


def test_fill_group_missing_no_alternate_error():
    X, y, _ = _make_rig()
    singleton = ["a", "b", "c", "d"]
    selector = FastCan(2, groups=singleton, verbose=0).fit(X, y)
    X_pred = X.copy()
    X_pred[:, 0] = np.nan
    with pytest.raises(ValueError, match="no alternate channel"):
        fill_group_missing(selector, X_pred)


def test_fill_group_missing_unrelated_broken_group_does_not_block():
    # On-demand verification: a broken alternate in a group whose
    # representative is present must not block another group's substitution.
    X, y, groups = _make_rig()
    selector = FastCan(2, groups=groups, verbose=0).fit(X, y)
    selector.X_train_ = selector.X_train_.copy()
    rng = np.random.default_rng(2)
    # Break the alternate of sig_v (feature 3) in the stored training data.
    selector.X_train_[:, 3] = rng.normal(size=X.shape[0])
    X_pred = X.copy()
    X_pred[:, 0] = np.nan  # only sig_u's representative is missing
    X_filled = fill_group_missing(selector, X_pred, r2_tol=0.9)
    np.testing.assert_allclose(X_filled[:, 0], X[:, 0], atol=1e-5)
    # The untouched group is returned as-is.
    np.testing.assert_array_equal(X_filled[:, 2:], X_pred[:, 2:])


def test_fill_group_missing_falls_back_to_second_alternate():
    rng = np.random.default_rng(3)
    u = rng.normal(size=200)
    w = rng.normal(size=200)
    # Group "u" has one representative and two alternates.
    X = np.column_stack([u, 2.0 * u + 1.0, -u + 4.0, w])
    y = u + rng.normal(scale=0.05, size=200)
    groups = ["u", "u", "u", "w"]
    selector = FastCan(1, groups=groups, verbose=0).fit(X, y)
    X_pred = X.copy()
    X_pred[:, 0] = np.nan  # representative missing
    X_pred[:, 1] = np.nan  # first alternate also unusable
    X_filled, report = fill_group_missing(
        selector, X_pred, r2_tol=0.9, return_report=True
    )
    np.testing.assert_allclose(X_filled[:, 0], X[:, 0], atol=1e-5)
    # The unusable first alternate is recorded as invalid in the report.
    assert not bool(report["valid"][1])


def test_fill_group_missing_all_alternates_invalid():
    X, y, groups = _make_rig()
    selector = FastCan(2, groups=groups, verbose=0).fit(X, y)
    selector.X_train_ = selector.X_train_.copy()
    rng = np.random.default_rng(4)
    # Break the only alternate of sig_u in the stored training data.
    selector.X_train_[:, 1] = rng.normal(size=X.shape[0])
    X_pred = X.copy()
    X_pred[:, 0] = np.nan
    with pytest.raises(ValueError, match="none of its"):
        fill_group_missing(selector, X_pred, r2_tol=0.9)


def test_fill_group_missing_requires_groups():
    X, y, _ = _make_rig()
    selector = FastCan(2, verbose=0).fit(X, y)
    with pytest.raises(AttributeError, match="without equivalence groups"):
        fill_group_missing(selector, X)
