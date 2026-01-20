# SPDX-License-Identifier: MPL-2.0
"""Pytest plugin for legacy_api_wrap."""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["_doctest_env", "_modify_env"]


@pytest.fixture
def _doctest_env() -> Generator[None, None, None]:
    """Pytest fixture to make doctests not error on expected warnings."""
    sys.stderr, stderr_orig = sys.stdout, sys.stderr
    with warnings.catch_warnings():
        warnings.resetwarnings()  # reset -Werror
        warnings.filterwarnings("ignore", r"The specified parameters", DeprecationWarning)
        yield
    sys.stderr = stderr_orig


@pytest.fixture(autouse=True)
def _modify_env(request: pytest.FixtureRequest) -> None:
    """Autorun fixture that conditionally modifies the test environment."""
    if isinstance(request.node, pytest.DoctestItem):
        request.getfixturevalue("_doctest_env")
