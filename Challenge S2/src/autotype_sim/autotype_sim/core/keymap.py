"""ANSI 87-key TKL layout as registration rectangles in the board frame.

Board frame: origin at the mounting panel's top-left corner as seen from the
arm, X right, Y down, metres. Every key is an axis-aligned rectangle
``[x0, x1] x [y0, y1]``: its cell on the 1u grid shrunk by ``key_inset`` per
side. A point in the gap between two caps maps to no key (``NO_KEY``).

Key units: 1u = ``key_pitch`` (0.01905 m, Cherry MX). The layout table is
The design spec section 5 verbatim -- rows top->bottom, function row at y = 0, a
0.25u gap, then rows at y = 1.25 ... 5.25; every key is 1u tall; the whole grid
spans 18.25u x 6.25u from ``key_area_origin``.

Lookup convention: rectangles are CLOSED (a point exactly on a cap edge is a
hit). Because neighbouring caps are separated by ``2 * key_inset`` this is
unambiguous -- no point can lie in two rectangles. That invariant is enforced,
not assumed: ``generate_tkl`` requires ``key_inset > 0`` and ``KeyMap`` rejects
any two rectangles that share even a single point.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

KINDS = ("char", "backspace", "other")

# The 36 launch-key characters, the design spec section 5 (uppercase letters only).
ALNUM_NAMES = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Total extent of the 1u grid in key units.
TKL_WIDTH_U = 18.25
TKL_HEIGHT_U = 6.25


def _run(names: Iterable[str], x: float) -> tuple[tuple[str, float, float], ...]:
    """Consecutive 1u keys starting at ``x`` (a str iterates its characters)."""
    return tuple((name, x + i, 1.0) for i, name in enumerate(names))


# Per row: (y, ((name, x, width), ...)) in key units from key_area_origin.
_ROWS: tuple[tuple[float, tuple[tuple[str, float, float], ...]], ...] = (
    (
        0.0,
        (
            ("ESC", 0.0, 1.0),
            *_run(("F1", "F2", "F3", "F4"), 2.0),
            *_run(("F5", "F6", "F7", "F8"), 6.5),
            *_run(("F9", "F10", "F11", "F12"), 11.0),
            *_run(("PRTSC", "SCRLK", "PAUSE"), 15.25),
        ),
    ),
    (
        1.25,
        (
            *_run("`1234567890-=", 0.0),
            ("BACKSPACE", 13.0, 2.0),
            *_run(("INS", "HOME", "PGUP"), 15.25),
        ),
    ),
    (
        2.25,
        (
            ("TAB", 0.0, 1.5),
            *_run("QWERTYUIOP[]", 1.5),
            ("\\", 13.5, 1.5),
            *_run(("DEL", "END", "PGDN"), 15.25),
        ),
    ),
    (
        3.25,
        (
            ("CAPS", 0.0, 1.75),
            *_run("ASDFGHJKL;'", 1.75),
            ("ENTER", 12.75, 2.25),
        ),
    ),
    (
        4.25,
        (
            ("LSHIFT", 0.0, 2.25),
            *_run("ZXCVBNM,./", 2.25),
            ("RSHIFT", 12.25, 2.75),
            ("UP", 16.25, 1.0),
        ),
    ),
    (
        5.25,
        (
            ("LCTRL", 0.0, 1.25),
            ("LWIN", 1.25, 1.25),
            ("LALT", 2.5, 1.25),
            ("SPACE", 3.75, 6.25),
            ("RALT", 10.0, 1.25),
            ("RWIN", 11.25, 1.25),
            ("MENU", 12.5, 1.25),
            ("RCTRL", 13.75, 1.25),
            *_run(("LEFT", "DOWN", "RIGHT"), 15.25),
        ),
    ),
)

# Flattened layout: (name, x, y, width) in key units; height is always 1u.
TKL_LAYOUT: tuple[tuple[str, float, float, float], ...] = tuple(
    (name, x, y, w) for y, row in _ROWS for name, x, w in row
)


def classify(name: str) -> tuple[str | None, str]:
    """``(char, kind)`` for a key name, the design spec section 5.

    Single-character names (letters, digits, punctuation) type themselves;
    SPACE types ``' '``; BACKSPACE is ``backspace``; everything else is
    ``other`` (logged, ignored by the typed string).
    """
    if len(name) == 1:
        return name, "char"
    if name == "SPACE":
        return " ", "char"
    if name == "BACKSPACE":
        return None, "backspace"
    return None, "other"


@dataclass(frozen=True)
class Key:
    """One key cap's registration rectangle, board frame, metres.

    ``[x0, x1] x [y0, y1]`` is the 1u-grid cell shrunk by ``key_inset`` on
    every side. ``char`` is what a press types (``None`` for non-typing keys);
    ``kind`` is one of ``KINDS``.
    """

    name: str
    char: str | None
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        # DESIGN s5/s8: a ``char`` key types exactly one character; ``backspace``
        # and ``other`` keys type nothing. Checked here so a malformed key fails
        # when it is built, not on its first accepted press inside Episode.
        if self.kind == "char":
            if not (isinstance(self.char, str) and len(self.char) == 1):
                raise ValueError(
                    f"char key {self.name!r} must carry exactly one character, "
                    f"got {self.char!r}"
                )
        elif self.char is not None:
            raise ValueError(
                f"{self.kind} key {self.name!r} must not carry a character, "
                f"got {self.char!r}"
            )
        for attr in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        if not all(math.isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError(f"non-finite rectangle for key {self.name!r}")
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(f"degenerate rectangle for key {self.name!r}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class KeyMap:
    """A set of key rectangles with point lookup.

    Construction validates the invariants ``lookup`` and ``alnum`` rely on:
    names are unique, letter keys are uppercase (DESIGN s5), and no two CLOSED
    rectangles share a point -- overlapping *or merely touching* caps are
    rejected, so a lookup hit is never ambiguous.
    """

    keys: tuple[Key, ...]
    by_name: dict[str, Key] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        keys = tuple(self.keys)
        object.__setattr__(self, "keys", keys)
        by_name = {k.name: k for k in keys}
        if len(by_name) != len(keys):
            raise ValueError("key names must be unique")
        for k in keys:
            for label, text in (("name", k.name), ("char", k.char)):
                if text is not None and _is_lowercase_letter(text):
                    raise ValueError(
                        f"letters are stored uppercase (DESIGN s5): key "
                        f"{k.name!r} has lowercase {label} {text!r}"
                    )
        object.__setattr__(self, "by_name", by_name)
        rects = np.array(
            [(k.x0, k.y0, k.x1, k.y1) for k in keys], dtype=np.float64
        ).reshape(-1, 4)
        clash = _first_shared_point(rects)
        if clash is not None:
            a, b = (keys[i] for i in clash)
            raise ValueError(
                f"key rectangles {a.name!r} and {b.name!r} share a point; "
                "caps must be separated by a gap"
            )
        object.__setattr__(self, "_rects", rects)

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self):
        return iter(self.keys)

    def lookup(self, x: float, y: float) -> Key | None:
        """Key whose CLOSED rectangle contains board point ``(x, y)``, else None.

        Points in the inset gap between caps (and anywhere off the grid)
        return None; that is the press ladder's ``NO_KEY``.
        """
        r = self._rects
        hit = np.flatnonzero(
            (r[:, 0] <= x) & (x <= r[:, 2]) & (r[:, 1] <= y) & (y <= r[:, 3])
        )
        if hit.size == 0:
            return None
        return self.keys[int(hit[0])]

    def alnum(self) -> list[Key]:
        """The 36 launch-key keys A-Z 0-9, sorted by name (digits first)."""
        return sorted(
            (k for k in self.keys if k.name in ALNUM_NAMES), key=lambda k: k.name
        )

    @staticmethod
    def center(key: Key) -> tuple[float, float]:
        """Rectangle centre ``(x, y)`` in board metres."""
        return (0.5 * (key.x0 + key.x1), 0.5 * (key.y0 + key.y1))


def _is_lowercase_letter(text: str) -> bool:
    """True for a single alphabetic character that is not already uppercase."""
    return len(text) == 1 and text.isalpha() and text != text.upper()


def _first_shared_point(rects: np.ndarray) -> tuple[int, int] | None:
    """Indices of the first pair of CLOSED rectangles that share a point.

    ``rects`` is ``(n, 4)`` as ``x0, y0, x1, y1``. Two closed rectangles are
    disjoint iff they are strictly separated along some axis; anything else
    (overlap, a shared edge, a shared corner) would make ``lookup`` ambiguous.
    """
    n = rects.shape[0]
    if n < 2:
        return None
    x0, y0, x1, y1 = (rects[:, i] for i in range(4))
    separated = (
        (x1[:, None] < x0[None, :])
        | (x0[:, None] > x1[None, :])
        | (y1[:, None] < y0[None, :])
        | (y0[:, None] > y1[None, :])
    )
    clash = np.triu(~separated, k=1)
    i, j = np.nonzero(clash)
    if i.size == 0:
        return None
    return int(i[0]), int(j[0])


def _geom(geometry: Any, name: str) -> Any:
    """Read a geometry field from a mapping or an attribute-style object."""
    if isinstance(geometry, Mapping):
        return geometry[name]
    return getattr(geometry, name)


def generate_tkl(geometry: Mapping[str, Any]) -> KeyMap:
    """Build the 87-key TKL KeyMap from ``board_geometry.yaml`` fields.

    Uses ``key_pitch`` (1u in metres), ``key_area_origin`` (grid top-left in
    the board frame) and ``key_inset`` (cell -> cap shrink per side).
    """
    pitch = float(_geom(geometry, "key_pitch"))
    ox, oy = (float(v) for v in _geom(geometry, "key_area_origin"))
    inset = float(_geom(geometry, "key_inset"))
    if not (pitch > 0.0 and math.isfinite(pitch)):
        raise ValueError(f"key_pitch must be positive and finite, got {pitch}")
    # Strictly positive: at inset 0 neighbouring caps share an edge and the
    # CLOSED lookup would attribute that edge by table order instead of NO_KEY.
    if not 0.0 < inset < 0.5 * pitch:
        raise ValueError(
            f"key_inset must satisfy 0 < inset < key_pitch / 2, got {inset}"
        )

    keys = []
    for name, x, y, w in TKL_LAYOUT:
        char, kind = classify(name)
        keys.append(
            Key(
                name=name,
                char=char,
                kind=kind,
                x0=ox + x * pitch + inset,
                y0=oy + y * pitch + inset,
                x1=ox + (x + w) * pitch - inset,
                y1=oy + (y + 1.0) * pitch - inset,
            )
        )
    return KeyMap(tuple(keys))


def load_geometry(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read ``board_geometry.yaml`` into a dict (the design spec section 4).

    ``marker_centers`` keys are normalised to ``int`` whether the file wrote
    them as ints or strings.
    """
    with open(path, "r", encoding="utf-8") as fh:
        geom = yaml.safe_load(fh)
    if not isinstance(geom, Mapping):
        raise ValueError(f"{path}: expected a mapping at top level")
    geom = dict(geom)
    centers = geom.get("marker_centers")
    if isinstance(centers, Mapping):
        geom["marker_centers"] = {
            int(k): [float(v) for v in xy] for k, xy in centers.items()
        }
    return geom


_SCHEMA = "autotype_sim/keymap-1"
_KEY_FIELDS = ("name", "kind", "x0", "y0", "x1", "y1")
_HEADER = (
    "# Key registration rectangles, board frame, metres: origin at the panel\n"
    "# top-left as seen from the arm, X right, Y down. Each key is its 1u cell\n"
    "# shrunk by key_inset per side; the gap between caps maps to no key.\n"
    "# Generated from board_geometry.yaml -- do not hand-edit.\n"
)


def save_keymap(km: KeyMap, path: str | os.PathLike[str]) -> None:
    """Write a KeyMap as YAML, one flow-style mapping per key, in key order."""
    doc = {
        "schema": _SCHEMA,
        "frame": "board",
        "units": "m",
        "keys": [
            {
                "name": k.name,
                "char": k.char,
                "kind": k.kind,
                "x0": k.x0,
                "y0": k.y0,
                "x1": k.x1,
                "y1": k.y1,
            }
            for k in km.keys
        ],
    }
    body = yaml.safe_dump(
        doc, sort_keys=False, default_flow_style=None, width=1000
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_HEADER)
        fh.write(body)


def load_keymap(path: str | os.PathLike[str]) -> KeyMap:
    """Read a YAML file written by ``save_keymap`` back into a KeyMap."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, Mapping) or "keys" not in doc:
        raise ValueError(f"{path}: not a keymap file (missing 'keys')")
    schema = doc.get("schema", _SCHEMA)
    if schema != _SCHEMA:
        raise ValueError(f"{path}: unsupported keymap schema {schema!r}")
    entries = doc["keys"]
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'keys' must be a list")
    keys = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: key entry {i} is not a mapping")
        missing = [f for f in _KEY_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"{path}: key entry {i} is missing {missing}")
        char = entry.get("char")
        try:
            keys.append(
                Key(
                    name=str(entry["name"]),
                    char=None if char is None else str(char),
                    kind=str(entry["kind"]),
                    x0=float(entry["x0"]),
                    y0=float(entry["y0"]),
                    x1=float(entry["x1"]),
                    y1=float(entry["y1"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: key entry {i}: {exc}") from exc
    try:
        return KeyMap(tuple(keys))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
