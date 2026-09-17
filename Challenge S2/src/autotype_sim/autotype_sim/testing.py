"""Helpers shared by the test suite and the offline tools.

Two things live here, in one place, so tests and tools cannot drift:

* :func:`resolve_private_dir` -- the directory in which the site-specific
  configuration files are looked for. ``$AUTOTYPE_PRIVATE_DIR`` is
  authoritative when set (a set-but-missing directory means "no site
  configuration"); otherwise ``<repo>/private``.
* :data:`DEFAULT_GEOMETRY` -- a complete, self-consistent board geometry used
  when no site configuration is present, so the tests and the offline tools
  run from a bare checkout.

Nothing in this module reads the configuration files itself except
:func:`load_test_geometry`, which is what ``conftest.py`` builds its fixtures
from.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

PRIVATE_DIR_ENV = "AUTOTYPE_PRIVATE_DIR"
GEOMETRY_FILENAME = "board_geometry.yaml"
LAYOUT_FILENAME = "keyboard_layout.yaml"

# Default rig description. The panel and marker fields are the published
# URC values: 0.400 x 0.175 m panel, 20 mm DICT_4X4_50 markers
# centred 16 mm in from each panel corner, Cherry MX pitch. The placement
# fields form one internally consistent set: the 18.25u x 6.25u key grid lies
# inside the keyboard plate and the plate inside the panel (BoardGeometry
# enforces both containments; test_board asserts them explicitly).
DEFAULT_GEOMETRY: dict[str, Any] = {
    "panel_w": 0.400,
    "panel_h": 0.175,
    "marker_size": 0.020,
    "marker_dict": "DICT_4X4_50",
    "marker_centers": {
        0: [0.016, 0.016],
        1: [0.384, 0.016],
        2: [0.384, 0.159],
        3: [0.016, 0.159],
    },
    "kb_origin": [0.024, 0.029],
    "kb_w": 0.374,
    "kb_h": 0.144,
    "key_pitch": 0.01905,
    "key_area_origin": [0.025, 0.0298],   # bezel 1.00 / 0.80 mm
    "key_inset": 0.0009,
    "pose_salt": 611209473388165509,
}

# Keys of DEFAULT_GEOMETRY that describe where the keyboard sits on the panel;
# everything else is URC-public. Tests use this list to check that they assert
# consistency rather than placement values.
PLACEMENT_FIELDS = ("kb_origin", "kb_w", "kb_h", "key_area_origin", "key_inset", "key_pitch")


def repo_root() -> Path:
    """Repository root when running from a source checkout.

    ``<repo>/src/autotype_sim/autotype_sim/testing.py`` -> ``<repo>``. When the
    package is installed elsewhere this is some unrelated directory; that is
    harmless because ``<root>/private`` then simply does not exist and the
    Docker image sets ``$AUTOTYPE_PRIVATE_DIR`` anyway.
    """
    return Path(__file__).resolve().parents[3]


def resolve_private_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Directory in which ``board_geometry.yaml`` is expected.

    Precedence: ``explicit`` (a CLI flag), then ``$AUTOTYPE_PRIVATE_DIR`` when
    set -- even if it does not exist, so pointing it at a missing directory
    reliably means "run without a site configuration" -- then
    ``<repo>/private``. The returned path may not exist; see
    :func:`private_geometry_path`.
    """
    if explicit is not None and str(explicit) != "":
        return Path(explicit).expanduser()
    env = os.environ.get(PRIVATE_DIR_ENV)
    if env is not None and env != "":
        return Path(env).expanduser()
    return repo_root() / "private"


def private_geometry_path(private_dir: str | os.PathLike[str] | None = None) -> Path | None:
    """Path of the configured geometry YAML if it exists, else ``None``."""
    path = resolve_private_dir(private_dir) / GEOMETRY_FILENAME
    return path if path.is_file() else None


def default_geometry() -> dict[str, Any]:
    """A fresh, mutable copy of :data:`DEFAULT_GEOMETRY`."""
    return copy.deepcopy(DEFAULT_GEOMETRY)


def load_test_geometry(
    private_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """``(geometry_dict, is_private)``: the configured YAML when available, else
    a copy of :data:`DEFAULT_GEOMETRY`.

    The dict has the ``board_geometry.yaml`` keys with ``marker_centers`` ids
    normalised to ``int`` (``keymap.load_geometry`` semantics), so it feeds
    both ``BoardGeometry.from_dict`` and ``keymap.generate_tkl``.
    """
    path = private_geometry_path(private_dir)
    if path is None:
        return default_geometry(), False
    from autotype_sim.core.keymap import load_geometry  # local: keep this module dependency-free

    return load_geometry(path), True


__all__ = [
    "DEFAULT_GEOMETRY",
    "GEOMETRY_FILENAME",
    "LAYOUT_FILENAME",
    "PLACEMENT_FIELDS",
    "PRIVATE_DIR_ENV",
    "default_geometry",
    "load_test_geometry",
    "private_geometry_path",
    "repo_root",
    "resolve_private_dir",
]
