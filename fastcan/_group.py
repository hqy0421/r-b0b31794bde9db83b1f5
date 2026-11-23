"""
Equivalence-group (duplicate sampling channel) handling.

Experimental test rigs often record the same physical quantity through
several redundant channels.  The corresponding columns are (almost)
collinear, so a stepwise selector can pick different members of the same
group in different runs without changing the model.  The helpers in this
module make such groups explicit and trackable:

* :func:`check_group_equivalence` verifies that the members of a group can
  actually stand in for each other (an affine relationship with an
  :math:`R^2` above a tolerance), and
* :func:`fill_group_missing` substitutes a frozen representative with a
  verified group member whenever the representative column is missing.

The selectors (:class:`fastcan.FastCan` and
:class:`fastcan.LazyFastCan`) consume the normalized output of
:func:`normalize_groups`, which maps arbitrary group labels to compact
integer group ids used by the search routines.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Real

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "alternate_mask",
    "check_group_equivalence",
    "fill_group_missing",
    "normalize_groups",
]


def _group_label_dtype(groups):
    """Return a dtype able to hold arbitrary hashable group labels."""
    labels = np.asarray(groups).ravel()
    if labels.dtype.kind in "OU":
        # Keep string/unicode labels as (unicode) objects/strings.
        return labels.dtype
    return object


@validate_params(
    {
        "groups": ["array-like"],
        "indices_exclude": ["array-like", None],
        "indices_include": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def normalize_groups(
    groups, n_features=None, indices_exclude=None, indices_include=None
):
    """Normalize arbitrary group labels to compact integer group ids.

    Features sharing the same label belong to the same equivalence group,
    i.e. they are duplicate (redundant) sampling channels of the same
    physical quantity.  Each group is represented by its lowest-indexed
    member; the other members are called *alternates*.  A stepwise selector
    only spends one selection slot per group and freezes the representative
    once the group is selected.

    Parameters
    ----------
    groups : array-like of shape (n_features,), default=None
        Group label of every feature.  Features with the same label are
        treated as equivalent channels.  ``None`` means that every feature
        forms its own group.

    n_features : int, default=None
        The number of features.  When given, the length of `groups` is
        checked against it.

    indices_exclude : array-like of shape (n_exclusions,), default=None
        The indices of the excluded features.  An excluded representative
        would leave a group without a selectable member, so groups whose
        representative is excluded raise an error.

    indices_include : array-like of shape (n_inclusions,), default=None
        The indices of the prerequisite features.  Two included features
        from the same group would spend two slots on one physical signal,
        so this configuration raises an error.

    Returns
    -------
    group_ids : ndarray of shape (n_features,), dtype=np.int64
        Compact integer id of the group each feature belongs to.  Ids are
        assigned in the order of first appearance of the labels.  When
        `groups` is None, every feature gets a distinct id.

    group_labels_ : ndarray of shape (n_groups,), dtype=object
        The unique labels corresponding to the integer ids, i.e.
        ``group_labels_[group_ids[i]]`` reconstructs the label of feature
        ``i``.

    representative_ids_ : ndarray of shape (n_groups,), dtype=np.int64
        The feature index of the (lowest-indexed) representative of each
        group.

    group_members_ : list of ndarray
        ``group_members_[g]`` holds the feature indices belonging to group
        ``g``, ordered by feature index.

    Examples
    --------
    >>> import numpy as np
    >>> from fastcan._group import normalize_groups
    >>> group_ids, labels, reps, members = normalize_groups(
    ...     ["u", "u", "v"], n_features=3
    ... )
    >>> group_ids
    array([0, 0, 1])
    >>> labels.tolist()
    ['u', 'v']
    >>> reps
    array([0, 2])
    >>> [m.tolist() for m in members]
    [[0, 1], [2]]
    """
    if groups is None:
        if n_features is None:
            raise ValueError("`n_features` should be provided when `groups` is None.")
        group_ids = np.arange(n_features, dtype=np.int64)
        group_labels = np.arange(n_features, dtype=object)
        representatives = np.arange(n_features, dtype=np.int64)
        members = [np.array([i], dtype=np.int64) for i in range(n_features)]
        return group_ids, group_labels, representatives, members

    raw = check_array(
        np.asarray(groups, dtype=_group_label_dtype(groups)),
        ensure_2d=False,
        dtype=None,
        ensure_min_samples=1,
    )
    if raw.ndim != 1:
        raise ValueError(
            f"`groups` should be one-dimensional, but got {raw.ndim} dimensions."
        )
    if n_features is not None and raw.shape[0] != n_features:
        raise ValueError(
            f"The length of `groups` ({raw.shape[0]}) should be equal to "
            f"the number of features ({n_features})."
        )

    # np.unique on object arrays keeps the first-appearance order only via
    # return_index, so build the mapping explicitly for determinism.
    label_to_id = {}
    ordered_labels = []
    group_ids = np.empty(raw.shape[0], dtype=np.int64)
    for i, label in enumerate(raw.tolist()):
        if label is None:
            raise ValueError(
                "Group labels should not contain None; use a sentinel value "
                "such as -1 for ungrouped features instead."
            )
        if label not in label_to_id:
            label_to_id[label] = len(ordered_labels)
            ordered_labels.append(label)
        group_ids[i] = label_to_id[label]

    n_groups = len(ordered_labels)
    group_labels = np.asarray(ordered_labels, dtype=object)
    members = [np.flatnonzero(group_ids == g).astype(np.int64) for g in range(n_groups)]
    representatives = np.asarray([member[0] for member in members], dtype=np.int64)

    if indices_exclude is not None:
        excluded = np.asarray(indices_exclude, dtype=np.int64)
        blocked = np.intersect1d(representatives, excluded)
        if blocked.size:
            raise ValueError(
                "The representative feature(s) "
                f"{blocked.tolist()} of equivalence group(s) "
                f"{group_labels[group_ids[blocked]].tolist()} are excluded. "
                "Excluding a representative leaves its group without a "
                "selectable member; exclude the whole group instead."
            )

    if indices_include is not None:
        included = np.asarray(indices_include, dtype=np.int64)
        included_groups = group_ids[included]
        unique_groups, counts = np.unique(included_groups, return_counts=True)
        if unique_groups.size != included_groups.size:
            duplicated = unique_groups[counts > 1]
            raise ValueError(
                "`indices_include` contains several features from the same "
                f"equivalence group (group labels "
                f"{group_labels[duplicated].tolist()}). A group "
                "occupies a single selection slot."
            )
        non_representative = included[included != representatives[included_groups]]
        if non_representative.size:
            raise ValueError(
                "`indices_include` should only contain group representatives "
                "(the lowest-indexed member of each group), but got alternate "
                f"feature indices {non_representative.tolist()}."
            )

    return group_ids, group_labels, representatives, members


def alternate_mask(group_ids, representatives):
    """Boolean mask marking every non-representative group member.

    Parameters
    ----------
    group_ids : ndarray of shape (n_features,), dtype=np.int64
        Compact integer id of the group each feature belongs to.

    representatives : ndarray of shape (n_groups,), dtype=np.int64
        The feature index of the representative of each group.

    Returns
    -------
    mask : ndarray of shape (n_features,), dtype=bool
        True for features that are not the representative of their group.
        Such features are excluded from the candidate pool up front so that
        the frozen representative is always the lowest-indexed member.
    """
    group_ids = np.asarray(group_ids, dtype=np.int64)
    representatives = np.asarray(representatives, dtype=np.int64)
    is_representative = np.zeros(group_ids.shape[0], dtype=bool)
    is_representative[representatives] = True
    return ~is_representative


def _validate_fitted_groups(estimator):
    """Return the group bookkeeping stored on a fitted selector/NARX."""
    check_is_fitted(estimator)
    group_ids = getattr(estimator, "group_ids_", None)
    if group_ids is None:
        raise AttributeError(
            f"{type(estimator).__name__} was fitted without equivalence "
            "groups (`groups=None`); there is no representative to substitute."
        )
    return (
        group_ids,
        estimator.group_labels_,
        estimator.group_representatives_,
        estimator.group_members_,
    )


@validate_params(
    {
        "estimator": [object],
        "X": ["array-like"],
        "r2_tol": [Interval(Real, 0.0, 1.0, closed="both")],
        "ensure_all_finite": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def _fit_alternate_affine(X, rep, alt):
    """Fit x_alt = a * x_rep + b on pairwise-finite samples.

    Returns ``(r2, coef, intercept, n_pair)``.
    """
    finite = np.isfinite(X[:, rep]) & np.isfinite(X[:, alt])
    n_pair = int(finite.sum())
    if n_pair < 2:
        raise ValueError(
            f"Feature {alt} has only {n_pair} finite samples shared with its "
            f"representative feature {rep}; cannot verify the substitution "
            "condition."
        )
    x_rep = X[finite, rep].reshape(-1, 1)
    x_alt = X[finite, alt]
    reg = LinearRegression().fit(x_rep, x_alt)
    return (
        float(reg.score(x_rep, x_alt)),
        float(reg.coef_[0]),
        float(reg.intercept_),
        n_pair,
    )


def check_group_equivalence(estimator, X, r2_tol=0.95, ensure_all_finite=False):
    r"""Verify that duplicate channels remain equivalent to their representatives.

    For every equivalence group with more than one member, an affine
    relationship :math:`x_j = a\\,x_r + b` between each alternate channel
    :math:`x_j` and the frozen representative :math:`x_r` is fitted by
    ordinary least squares.  The relationship is accepted as a valid
    substitution condition when its coefficient of determination
    :math:`R^2` is at least `r2_tol`.  Samples containing non-finite values
    in either channel are ignored pairwise.

    Parameters
    ----------
    estimator : FastCan, LazyFastCan or NARX
        An estimator fitted with non-None `groups`.

    X : array-like of shape (n_samples, n_features)
        The data on which the equivalence is checked.  For a
        :class:`~fastcan.narx.NARX` model, pass the raw input matrix used
        at prediction time (output channels are not grouped).

    r2_tol : float, default=0.95
        Minimum acceptable :math:`R^2` of the affine fit between an
        alternate channel and its group representative.

    ensure_all_finite : bool, default=False
        If True, raise when `X` contains any non-finite value.  If False
        (default), non-finite samples are ignored pairwise.

    Returns
    -------
    report : dict
        A dictionary with keys:

        - ``"r2"``: ndarray of shape (n_features,), the :math:`R^2` of the
          affine fit for each feature against its group representative
          (representatives and singleton groups get 1.0);
        - ``"coef"``: ndarray of shape (n_features,), slope :math:`a`;
        - ``"intercept"``: ndarray of shape (n_features,), offset :math:`b`;
        - ``"valid"``: ndarray of shape (n_features,), dtype=bool, whether
          the feature is an admissible substitute for its representative;
        - ``"n_samples"``: ndarray of shape (n_features,), number of finite
          samples used for each fit.

    Raises
    ------
    ValueError
        If an alternate channel fails the `r2_tol` condition or does not
        have enough finite samples to fit the affine relationship.

    Examples
    --------
    >>> import numpy as np
    >>> from fastcan import FastCan, check_group_equivalence
    >>> rng = np.random.default_rng(0)
    >>> u = rng.normal(size=50)
    >>> X = np.column_stack([u, 2.0 * u + 1.0])
    >>> y = u + rng.normal(scale=0.01, size=50)
    >>> selector = FastCan(1, groups=["u", "u"], verbose=0).fit(X, y)
    >>> report = check_group_equivalence(selector, X, r2_tol=0.9)
    >>> report["valid"].tolist()
    [True, True]
    >>> round(float(report["r2"][1]), 3)
    1.0
    """
    group_ids, group_labels, _, members = _validate_fitted_groups(estimator)
    X = check_array(
        X,
        ensure_2d=True,
        dtype=float,
        ensure_all_finite=ensure_all_finite,
        ensure_min_features=group_ids.shape[0],
    )
    if X.shape[1] != group_ids.shape[0]:
        raise ValueError(
            f"`X` has {X.shape[1]} features but the estimator was fitted "
            f"with {group_ids.shape[0]} grouped features."
        )

    n_features = X.shape[1]
    r2 = np.ones(n_features, dtype=float)
    coef = np.ones(n_features, dtype=float)
    intercept = np.zeros(n_features, dtype=float)
    n_used = np.full(n_features, X.shape[0], dtype=np.int64)
    valid = np.ones(n_features, dtype=bool)

    for members_g in members:
        if members_g.size < 2:
            continue
        rep = int(members_g[0])
        for alt in members_g[1:]:
            alt = int(alt)
            try:
                score, a, b, n_pair = _fit_alternate_affine(X, rep, alt)
            except ValueError as exc:
                raise ValueError(
                    f"Feature {alt} (group label "
                    f"{group_labels[group_ids[alt]]!r}): {exc}"
                ) from exc
            n_used[alt] = n_pair
            r2[alt] = score
            coef[alt] = a
            intercept[alt] = b
            if score < r2_tol:
                valid[alt] = False
                raise ValueError(
                    f"Feature {alt} is not an admissible substitute for its "
                    f"frozen representative feature {rep} (group label "
                    f"{group_labels[group_ids[alt]]!r}): affine R^2="
                    f"{score:.6f} < r2_tol={r2_tol}."
                )
    return {
        "r2": r2,
        "coef": coef,
        "intercept": intercept,
        "valid": valid,
        "n_samples": n_used,
    }


@validate_params(
    {
        "estimator": [object],
        "X": ["array-like"],
        "r2_tol": [Interval(Real, 0.0, 1.0, closed="both")],
        "return_report": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def fill_group_missing(estimator, X, r2_tol=0.95, return_report=False):
    r"""Fill missing representative channels with verified group alternates.

    A column is considered missing when all of its samples are non-finite
    (e.g. a sensor dropout recorded as ``np.nan``).  In that case an
    alternate member of the same equivalence group is selected:

    #. the affine relationship :math:`x_j = a\\,x_r + b` between the
       alternate and the frozen representative is re-estimated on the
       training data stored by the estimator, and accepted only when its
       :math:`R^2` reaches `r2_tol`;
    #. the missing column is reconstructed as
       :math:`\\hat{x}_r = (x_j - b)/a` (or :math:`x_j` when the fitted
       slope is numerically zero), so the model keeps consuming its
       original, frozen feature position.

    Parameters
    ----------
    estimator : FastCan, LazyFastCan or NARX
        An estimator fitted with non-None `groups`.

    X : array-like of shape (n_samples, n_features)
        Prediction-time data.  Must have the same number of columns as the
        data used for fitting.

    r2_tol : float, default=0.95
        Minimum acceptable :math:`R^2` of the affine substitution mapping.

    return_report : bool, default=False
        If True, also return the verification report of the substitution
        mappings fitted on the stored training data.  Only groups whose
        representative is missing are verified, so entries for unchecked
        alternates keep their trivial defaults (see
        :func:`check_group_equivalence` for the report format).

    Returns
    -------
    X_filled : ndarray of shape (n_samples, n_features)
        A copy of `X` with every fully-missing representative column
        reconstructed from a verified alternate.  Columns that are not
        missing are returned unchanged (including their individual NaNs).

    report : dict, only when `return_report` is True
        The equivalence report computed on the stored training data.

    Examples
    --------
    >>> import numpy as np
    >>> from fastcan import FastCan, fill_group_missing
    >>> rng = np.random.default_rng(0)
    >>> u = rng.normal(size=50)
    >>> X = np.column_stack([u, 2.0 * u + 1.0])
    >>> y = u + rng.normal(scale=0.01, size=50)
    >>> selector = FastCan(1, groups=["u", "u"], verbose=0).fit(X, y)
    >>> X_missing = X.copy()
    >>> X_missing[:, 0] = np.nan
    >>> X_filled = fill_group_missing(selector, X_missing, r2_tol=0.9)
    >>> np.allclose(X_filled[:, 0], X[:, 0])
    True
    """
    group_ids, group_labels, representatives, members = _validate_fitted_groups(
        estimator
    )
    X = check_array(
        X,
        ensure_2d=True,
        dtype=float,
        ensure_all_finite=False,
        ensure_min_features=group_ids.shape[0],
    )
    if X.shape[1] != group_ids.shape[0]:
        raise ValueError(
            f"`X` has {X.shape[1]} features but the estimator was fitted "
            f"with {group_ids.shape[0]} grouped features."
        )

    X_train = getattr(estimator, "X_train_", None)
    if X_train is None:
        raise AttributeError(
            f"{type(estimator).__name__} does not store training data; "
            "cannot estimate the affine substitution mapping."
        )
    X_train = check_array(
        X_train,
        ensure_2d=True,
        dtype=float,
        ensure_all_finite=False,
        ensure_min_features=group_ids.shape[0],
    )

    # On-demand verification: only the groups whose representative is
    # actually missing are checked, so an unrelated broken group never
    # blocks an otherwise valid substitution.  The report arrays keep the
    # same contract as check_group_equivalence; entries of unchecked
    # alternates stay at their trivial defaults.
    n_features = X.shape[1]
    report = {
        "r2": np.ones(n_features, dtype=float),
        "coef": np.ones(n_features, dtype=float),
        "intercept": np.zeros(n_features, dtype=float),
        "valid": np.ones(n_features, dtype=bool),
        "n_samples": np.full(n_features, X_train.shape[0], dtype=np.int64),
    }

    X_filled = np.array(X, copy=True, dtype=float)
    for rep in representatives:
        rep = int(rep)
        if np.isfinite(X_filled[:, rep]).any():
            continue  # representative present, nothing to substitute
        group = int(group_ids[rep])
        alternates = [int(j) for j in members[group][1:]]
        if not alternates:
            raise ValueError(
                f"Representative feature {rep} (group label "
                f"{group_labels[group]!r}) is missing and the group has no "
                "alternate channel to substitute."
            )
        substituted = False
        for alt in alternates:
            if not np.isfinite(X_filled[:, alt]).all():
                # Need a complete alternate signal at prediction time.
                report["valid"][alt] = False
                continue
            try:
                score, a, b, n_pair = _fit_alternate_affine(X_train, rep, alt)
            except ValueError:
                report["valid"][alt] = False
                continue
            report["r2"][alt] = score
            report["coef"][alt] = a
            report["intercept"][alt] = b
            report["n_samples"][alt] = n_pair
            if score < r2_tol:
                report["valid"][alt] = False
                continue  # try the next alternate instead of failing hard
            if abs(a) < np.finfo(float).eps:
                X_filled[:, rep] = X_filled[:, alt]
            else:
                X_filled[:, rep] = (X_filled[:, alt] - b) / a
            substituted = True
            break
        if not substituted:
            raise ValueError(
                f"Representative feature {rep} (group label "
                f"{group_labels[group]!r}) is missing and none of its "
                f"alternate channels {alternates} is complete and verified "
                f"(r2_tol={r2_tol})."
            )

    if return_report:
        return X_filled, report
    return X_filled
