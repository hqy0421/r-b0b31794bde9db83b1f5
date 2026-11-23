.. _equivalence_groups:

============================================
Equivalence groups for duplicate channels
============================================

.. currentmodule:: fastcan

Experimental test rigs frequently record the same physical quantity through
several sampling channels, e.g. two sensors mounted on the same shaft or a
signal logged by two synchronized acquisition units. Such columns are
(nearly) collinear. Without further information, a stepwise selector may
therefore pick different members of the same set of duplicates in different
runs, or spend two selection slots on what is physically one signal.

Equivalence groups make this redundancy explicit. Pass a ``groups`` array to
:class:`FastCan`, :class:`LazyFastCan`, or :func:`fastcan.narx.make_narx`
with one label per feature; features sharing a label are treated as
equivalent channels of the same quantity.

Representative selection
=========================

Within each group the lowest-indexed member is the *representative* and the
other members are *alternates*. The search only considers representatives,
so:

* a group occupies at most one selection slot (duplicate signals never take
  two budgets);
* the chosen feature of a group is deterministic and independent of tiny
  numerical differences between collinear columns;
* the group bookkeeping is exposed through the ``group_ids_``,
  ``group_labels_``, ``group_representatives_`` and ``group_members_``
  attributes.

.. code-block:: python

    >>> import numpy as np
    >>> from fastcan import FastCan
    >>> rng = np.random.default_rng(0)
    >>> u, v = rng.normal(size=(2, 50))
    >>> # Channels 0/1 measure u, channels 2/3 measure v.
    >>> X = np.column_stack([u, 2.0 * u + 1.0, v, -v])
    >>> y = u + 0.5 * v
    >>> selector = FastCan(
    ...     n_features_to_select=2,
    ...     groups=["u", "u", "v", "v"],
    ...     verbose=0,
    ... ).fit(X, y)
    >>> selector.indices_
    array([0, 2]...)

The same mechanism applies to the NARX term library built by
:func:`fastcan.narx.make_narx`: the raw-channel labels are propagated to the
time-shift and polynomial term candidates, so two terms that differ only by
the duplicate channel they use share a term group and compete for a single
term budget. The returned :class:`fastcan.narx.NARX` keeps the raw-channel
bookkeeping.

Verifiable substitution when a representative is missing
==========================================================

Grouping also defines what may replace a frozen representative. Two helpers
make the substitution condition explicit and verifiable:

* :func:`check_group_equivalence` fits an affine relationship
  :math:`x_j = a x_r + b` between every alternate :math:`x_j` and its
  representative :math:`x_r`, and accepts the alternate as a substitute only
  when the coefficient of determination :math:`R^2` reaches ``r2_tol``;
* :func:`fill_group_missing` reconstructs a representative column that is
  entirely missing at prediction time from the first complete, verified
  alternate, inverting the fitted affine mapping, and writes it back into
  the original feature position so the frozen model can run unchanged.

.. code-block:: python

    >>> from fastcan import fill_group_missing
    >>> X_missing = X.copy()
    >>> X_missing[:, 0] = np.nan  # representative of u drops out
    >>> X_filled = fill_group_missing(selector, X_missing, r2_tol=0.9)
    >>> np.allclose(X_filled[:, 0], X[:, 0])
    True

For a NARX model, call :func:`fill_group_missing` on the prediction data
before :meth:`fastcan.narx.NARX.predict`.

.. rubric:: Related API

* :class:`FastCan` and :class:`LazyFastCan` accept the ``groups`` parameter.
* :func:`fastcan.narx.make_narx` propagates raw-channel groups to its term
  candidates.
* :func:`check_group_equivalence` and :func:`fill_group_missing` verify and
  apply representative substitution.
