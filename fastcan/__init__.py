"""
Fast canonical correlation analysis based search algorithm.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from . import narx, utils
from ._fastcan import FastCan
from ._group import check_group_equivalence, fill_group_missing
from ._lazyfastcan import LazyFastCan
from ._minibatch import minibatch
from ._refine import refine

__all__ = [
    "FastCan",
    "LazyFastCan",
    "check_group_equivalence",
    "fill_group_missing",
    "minibatch",
    "narx",
    "refine",
    "utils",
]
