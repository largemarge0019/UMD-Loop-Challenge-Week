"""Shared pytest configuration and geometry fixtures.

Makes ``autotype_sim`` importable when running pytest from the repo root
without an installed package (colcon installs it properly in the VM), and
provides ONE source of board geometry for the whole suite:

* ``geometry_dict`` -- the site ``board_geometry.yaml`` when it can be found
  (``$AUTOTYPE_PRIVATE_DIR`` if set, else ``<repo>/private``), otherwise
  ``DEFAULT_GEOMETRY`` from ``autotype_sim.testing``.
* ``geometry`` / ``keymap`` -- ``BoardGeometry`` and the 87-key ``KeyMap``
  built from ``geometry_dict``.
* ``geometry_is_private`` -- which of the two you got.
* ``private_dir`` -- the configuration directory, skipping the test when
  absent. ``@pytest.mark.private`` does the same for a whole test.

Tests must not assert the *values* of the placement fields; assert consistency
(grid inside plate inside panel) so they pass on either geometry.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from autotype_sim.core.board import BoardGeometry  # noqa: E402
from autotype_sim.core.keymap import KeyMap, generate_tkl  # noqa: E402
from autotype_sim.testing import (  # noqa: E402  (re-exported for tests)
    DEFAULT_GEOMETRY,
    GEOMETRY_FILENAME,
    LAYOUT_FILENAME,
    PRIVATE_DIR_ENV,
    default_geometry,
    load_test_geometry,
    private_geometry_path,
    resolve_private_dir,
)

__all__ = [
    "DEFAULT_GEOMETRY",
    "GEOMETRY_FILENAME",
    "LAYOUT_FILENAME",
    "PRIVATE_DIR_ENV",
    "default_geometry",
    "load_test_geometry",
    "private_geometry_path",
    "resolve_private_dir",
]

_PRIVATE_SKIP_REASON = (
    f"private {GEOMETRY_FILENAME} not available (looked in {{where}}; set {PRIVATE_DIR_ENV})"
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running checks (e.g. the 50-seed solvability sweep)"
    )
    config.addinivalue_line(
        "markers",
        "private: needs a site board_geometry.yaml in private/; skipped when "
        "only the default geometry is available",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("private") is not None and private_geometry_path() is None:
        pytest.skip(_PRIVATE_SKIP_REASON.format(where=resolve_private_dir()))


@pytest.fixture(scope="session")
def private_dir() -> Path:
    """The directory holding the private YAML files; skips when it is absent."""
    if private_geometry_path() is None:
        pytest.skip(_PRIVATE_SKIP_REASON.format(where=resolve_private_dir()))
    return resolve_private_dir()


@pytest.fixture(scope="session")
def _test_geometry() -> tuple[dict, bool]:
    return load_test_geometry()


@pytest.fixture(scope="session")
def geometry_is_private(_test_geometry) -> bool:
    """True when ``geometry_dict`` came from the private YAML."""
    return _test_geometry[1]


@pytest.fixture(scope="session")
def geometry_dict(_test_geometry) -> dict:
    """board_geometry.yaml as a dict (from private/ if present, else DEFAULT_GEOMETRY).

    Session-shared: copy (``{**geometry_dict, ...}``) instead of mutating.
    """
    return _test_geometry[0]


@pytest.fixture(scope="session")
def geometry(geometry_dict) -> BoardGeometry:
    return BoardGeometry.from_dict(geometry_dict, source="geometry_dict fixture")


@pytest.fixture(scope="session")
def keymap(geometry_dict) -> KeyMap:
    """The 87-key TKL map generated from ``geometry_dict``."""
    return generate_tkl(geometry_dict)
