"""Tests for core/keymap.py against the design spec sections 4-5.

The pure-geometry tests run on ``DEFAULT_GEOMETRY`` so they never depend on a
site configuration. The loader tests at the end use the session
``geometry_dict`` fixture (the configured YAML when available) and assert only
consistency (grid inside plate inside panel), never specific coordinates.
"""

import itertools

import numpy as np
import pytest
import yaml

from autotype_sim.core.keymap import (
    ALNUM_NAMES,
    TKL_HEIGHT_U,
    TKL_LAYOUT,
    TKL_WIDTH_U,
    Key,
    KeyMap,
    classify,
    generate_tkl,
    load_geometry,
    load_keymap,
    save_keymap,
)
from autotype_sim.testing import DEFAULT_GEOMETRY

# generate_tkl only reads these three fields; taken from the default rig.
GEOM = {k: DEFAULT_GEOMETRY[k] for k in ("key_pitch", "key_area_origin", "key_inset")}
PITCH = GEOM["key_pitch"]
OX, OY = GEOM["key_area_origin"]
INSET = GEOM["key_inset"]


@pytest.fixture(scope="module")
def km() -> KeyMap:
    return generate_tkl(GEOM)


def _cell(key: Key) -> tuple[float, float, float, float]:
    """Undo the inset: the key's 1u-grid cell."""
    return (key.x0 - INSET, key.y0 - INSET, key.x1 + INSET, key.y1 + INSET)


def _separation(a: Key, b: Key) -> float:
    """Largest axis gap between two rectangles (negative iff they overlap)."""
    return max(b.x0 - a.x1, a.x0 - b.x1, b.y0 - a.y1, a.y0 - b.y1)


# 1. counts and uniqueness ---------------------------------------------------


def test_counts_and_unique_names(km):
    assert len(km) == 87
    assert len(km.keys) == 87
    assert len(TKL_LAYOUT) == 87
    names = [k.name for k in km.keys]
    assert len(set(names)) == 87
    assert set(km.by_name) == set(names)
    alnum = km.alnum()
    assert len(alnum) == 36
    assert {k.name for k in alnum} == set(ALNUM_NAMES)
    assert [k.name for k in alnum] == sorted(k.name for k in alnum)
    assert all(k.kind == "char" and k.char == k.name for k in alnum)


def test_all_names_present(km):
    expected = set("`1234567890-=QWERTYUIOP[]\\ASDFGHJKL;'ZXCVBNM,./") | {
        "ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
        "F11", "F12", "PRTSC", "SCRLK", "PAUSE", "BACKSPACE", "INS", "HOME",
        "PGUP", "TAB", "DEL", "END", "PGDN", "CAPS", "ENTER", "LSHIFT",
        "RSHIFT", "UP", "LCTRL", "LWIN", "LALT", "SPACE", "RALT", "RWIN",
        "MENU", "RCTRL", "LEFT", "DOWN", "RIGHT",
    }
    assert len(expected) == 87
    assert set(km.by_name) == expected


# 2. bounding box ------------------------------------------------------------


def test_bounding_box_before_inset(km):
    cells = np.array([_cell(k) for k in km.keys])
    assert cells[:, 0].min() == pytest.approx(OX, abs=1e-9)
    assert cells[:, 1].min() == pytest.approx(OY, abs=1e-9)
    assert cells[:, 2].max() == pytest.approx(OX + 18.25 * PITCH, abs=1e-9)
    assert cells[:, 3].max() == pytest.approx(OY + 6.25 * PITCH, abs=1e-9)
    assert TKL_WIDTH_U == 18.25 and TKL_HEIGHT_U == 6.25
    # Same statement in key units, straight from the layout table.
    xs = [(x, x + w) for _, x, _, w in TKL_LAYOUT]
    ys = [(y, y + 1.0) for _, _, y, _ in TKL_LAYOUT]
    assert min(x for x, _ in xs) == 0.0 and max(x for _, x in xs) == 18.25
    assert min(y for y, _ in ys) == 0.0 and max(y for _, y in ys) == 6.25


# 3. tiling / no overlaps ----------------------------------------------------


def test_rows_tile_without_overlap(km):
    rows: dict[float, list[Key]] = {}
    for k in km.keys:
        rows.setdefault(round(_cell(k)[1], 9), []).append(k)
    assert len(rows) == 6
    row_ys = sorted(rows)
    expected_ys = [OY + u * PITCH for u in (0.0, 1.25, 2.25, 3.25, 4.25, 5.25)]
    assert np.allclose(row_ys, expected_ys, atol=1e-9)
    for y in row_ys:
        cells = sorted(_cell(k) for k in rows[y])
        assert cells[0][0] == pytest.approx(OX, abs=1e-9)  # every row starts at x=0
        for (_, _, x1_prev, _), (x0_next, _, _, _) in zip(cells, cells[1:]):
            assert x0_next >= x1_prev - 1e-12
        for x0, y0, x1, y1 in cells:
            assert y1 - y0 == pytest.approx(PITCH, abs=1e-12)


def test_pairwise_rectangles_disjoint_with_gap(km):
    for a, b in itertools.combinations(km.keys, 2):
        overlap = a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1
        assert not overlap, (a.name, b.name)
        assert _separation(a, b) >= 2 * INSET - 1e-12, (a.name, b.name)


def test_every_key_has_correct_inset(km):
    for k in km.keys:
        name, x, y, w = next(t for t in TKL_LAYOUT if t[0] == k.name)
        assert k.x0 == pytest.approx(OX + x * PITCH + INSET, abs=1e-12)
        assert k.y0 == pytest.approx(OY + y * PITCH + INSET, abs=1e-12)
        assert k.x1 == pytest.approx(OX + (x + w) * PITCH - INSET, abs=1e-12)
        assert k.y1 == pytest.approx(OY + (y + 1.0) * PITCH - INSET, abs=1e-12)


# 4. lookup ------------------------------------------------------------------


def test_lookup_at_centers(km):
    for k in km.keys:
        cx, cy = km.center(k)
        assert km.lookup(cx, cy) is k
        assert cx == pytest.approx(0.5 * (k.x0 + k.x1))
        assert cy == pytest.approx(0.5 * (k.y0 + k.y1))


def test_lookup_in_gap_between_f_and_g(km):
    f, g = km.by_name["F"], km.by_name["G"]
    # Shared cell edge is at x = 5.75u; the inset gap straddles it.
    edge_x = OX + 5.75 * PITCH
    assert f.x1 == pytest.approx(edge_x - INSET, abs=1e-12)
    assert g.x0 == pytest.approx(edge_x + INSET, abs=1e-12)
    mid_x = 0.5 * (f.x1 + g.x0)
    mid_y = km.center(f)[1]
    assert mid_x == pytest.approx(edge_x, abs=1e-12)
    assert km.lookup(mid_x, mid_y) is None
    # Just inside either cap it resolves again.
    assert km.lookup(f.x1 - 1e-6, mid_y) is f
    assert km.lookup(g.x0 + 1e-6, mid_y) is g
    # Vertical gap between rows too (between '5' and 'T').
    five, t = km.by_name["5"], km.by_name["T"]
    assert km.lookup(km.center(t)[0], 0.5 * (five.y1 + t.y0)) is None


def test_lookup_is_closed_on_edges(km):
    k = km.by_name["H"]
    for x, y in ((k.x0, k.y0), (k.x1, k.y1), (k.x0, k.y1), (k.x1, k.y0)):
        assert km.lookup(x, y) is k
    tiny = 1e-9
    assert km.lookup(k.x0 - tiny, k.y0) is None
    assert km.lookup(k.x1 + tiny, k.y0) is None
    assert km.lookup(k.x0, k.y0 - tiny) is None
    assert km.lookup(k.x0, k.y1 + tiny) is None


def test_lookup_far_outside(km):
    assert km.lookup(-1.0, -1.0) is None
    assert km.lookup(10.0, 0.05) is None
    assert km.lookup(0.1, 10.0) is None
    assert km.lookup(OX - 1e-3, OY - 1e-3) is None
    assert km.lookup(float("nan"), 0.05) is None


def test_lookup_random_points_agree_with_brute_force(km):
    rng = np.random.default_rng(1234)
    xs = rng.uniform(OX - 0.01, OX + 18.25 * PITCH + 0.01, size=2000)
    ys = rng.uniform(OY - 0.01, OY + 6.25 * PITCH + 0.01, size=2000)
    for x, y in zip(xs, ys):
        brute = [
            k for k in km.keys if k.x0 <= x <= k.x1 and k.y0 <= y <= k.y1
        ]
        assert len(brute) <= 1
        assert km.lookup(x, y) is (brute[0] if brute else None)


# 5. row sanity and widths ---------------------------------------------------


def test_row_stagger(km):
    cx = lambda n: km.center(km.by_name[n])[0]  # noqa: E731
    assert cx("1") < cx("Q") < cx("2")
    assert cx("Q") < cx("A") < cx("W")
    assert cx("A") < cx("Z") < cx("S")
    cy = lambda n: km.center(km.by_name[n])[1]  # noqa: E731
    assert cy("F1") < cy("1") < cy("Q") < cy("A") < cy("Z") < cy("SPACE")
    assert cy("1") - cy("F1") == pytest.approx(1.25 * PITCH, abs=1e-12)
    assert cy("Q") - cy("1") == pytest.approx(PITCH, abs=1e-12)


@pytest.mark.parametrize(
    "name,width_u",
    [
        ("SPACE", 6.25),
        ("BACKSPACE", 2.0),
        ("ENTER", 2.25),
        ("RSHIFT", 2.75),
        ("LSHIFT", 2.25),
        ("CAPS", 1.75),
        ("TAB", 1.5),
        ("\\", 1.5),
        ("LCTRL", 1.25),
        ("A", 1.0),
    ],
)
def test_key_widths(km, name, width_u):
    k = km.by_name[name]
    assert k.width == pytest.approx(width_u * PITCH - 2 * INSET, abs=1e-12)
    assert k.height == pytest.approx(PITCH - 2 * INSET, abs=1e-12)


# 6. save / load -------------------------------------------------------------


def test_save_load_round_trip(km, tmp_path):
    path = tmp_path / "keyboard_layout.yaml"
    save_keymap(km, path)
    loaded = load_keymap(path)
    assert loaded == km
    assert loaded.keys == km.keys
    assert [k.name for k in loaded.keys] == [k.name for k in km.keys]
    for a, b in zip(loaded.keys, km.keys):
        assert (a.x0, a.y0, a.x1, a.y1) == (b.x0, b.y0, b.x1, b.y1)
        assert a.char == b.char and a.kind == b.kind
    # Deterministic bytes.
    path2 = tmp_path / "again.yaml"
    save_keymap(loaded, path2)
    assert path.read_bytes() == path2.read_bytes()
    # Awkward names survive YAML quoting.
    for name in ("'", "`", "\\", ",", ";", "-", "=", "[", "]", ".", "/", "7"):
        assert loaded.by_name[name].name == name


def test_load_rejects_wrong_schema(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema: something-else\nkeys: []\n")
    with pytest.raises(ValueError):
        load_keymap(p)
    p.write_text("nothing: here\n")
    with pytest.raises(ValueError):
        load_keymap(p)


# 7. char / kind mapping -----------------------------------------------------


def test_char_kind_spot_checks(km):
    bn = km.by_name
    assert bn["A"].char == "A" and bn["A"].kind == "char"
    assert bn["7"].char == "7" and bn["7"].kind == "char"
    assert bn["SPACE"].char == " " and bn["SPACE"].kind == "char"
    assert bn["BACKSPACE"].char is None and bn["BACKSPACE"].kind == "backspace"
    assert bn["LSHIFT"].char is None and bn["LSHIFT"].kind == "other"
    assert bn[","].char == "," and bn[","].kind == "char"
    assert bn["ENTER"].kind == "other" and bn["ENTER"].char is None
    assert bn["F1"].kind == "other" and bn["F1"].char is None
    kinds = {k.kind for k in km.keys}
    assert kinds == {"char", "backspace", "other"}
    assert sum(k.kind == "backspace" for k in km.keys) == 1
    # char kind <=> char is not None. Typing keys: 26 letters + 10 digits +
    # 11 punctuation + SPACE = 48.
    assert all((k.kind == "char") == (k.char is not None) for k in km.keys)
    assert sum(k.kind == "char" for k in km.keys) == 48


def test_classify():
    assert classify("Q") == ("Q", "char")
    assert classify("0") == ("0", "char")
    assert classify("`") == ("`", "char")
    assert classify("SPACE") == (" ", "char")
    assert classify("BACKSPACE") == (None, "backspace")
    assert classify("PGUP") == (None, "other")


def test_key_validation():
    with pytest.raises(ValueError):
        Key("X", "X", "weird", 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        Key("X", "X", "char", 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        KeyMap((Key("X", "X", "char", 0, 0, 1, 1), Key("X", "X", "char", 2, 0, 3, 1)))
    with pytest.raises(ValueError):
        generate_tkl({**GEOM, "key_inset": 0.02})


# 8. validation regressions (adversarial review) -----------------------------
#
# The CLOSED lookup is only unambiguous while no two rectangles share a point,
# and Episode (DESIGN s8) only works while a ``char`` key carries a character
# and letters are uppercase (DESIGN s5). Those invariants are enforced where
# a KeyMap is built or loaded, so a bad file fails at load, not mid-episode.


def test_generate_tkl_requires_strictly_positive_inset():
    # At inset 0 neighbouring caps share an edge exactly (F.x1 == G.x0) and the
    # CLOSED lookup would attribute that edge by table order instead of NO_KEY.
    for bad in (0.0, -0.0, -1e-9, float("nan"), 0.5 * PITCH, PITCH):
        with pytest.raises(ValueError, match="key_inset"):
            generate_tkl({**GEOM, "key_inset": bad})
    # Any positive inset keeps the shared cell edge inside the gap.
    tiny = generate_tkl({**GEOM, "key_inset": 1e-9})
    f, g = tiny.by_name["F"], tiny.by_name["G"]
    edge_x = OX + 5.75 * PITCH
    assert f.x1 < edge_x < g.x0
    assert tiny.lookup(edge_x, tiny.center(f)[1]) is None
    assert tiny.lookup(f.x1, tiny.center(f)[1]) is f
    assert tiny.lookup(g.x0, tiny.center(g)[1]) is g


def test_generate_tkl_rejects_non_finite_or_non_positive_pitch():
    for bad in (float("nan"), float("inf"), 0.0, -0.01905):
        with pytest.raises(ValueError, match="key_pitch"):
            generate_tkl({**GEOM, "key_pitch": bad})


def test_keymap_rejects_rectangles_that_share_a_point():
    a = Key("A", "A", "char", 0.0, 0.0, 1.0, 1.0)
    overlapping = Key("B", "B", "char", 0.5, 0.0, 1.5, 1.0)
    shared_edge = Key("B", "B", "char", 1.0, 0.0, 2.0, 1.0)
    shared_corner = Key("B", "B", "char", 1.0, 1.0, 2.0, 2.0)
    for other in (overlapping, shared_edge, shared_corner):
        with pytest.raises(ValueError, match="'A' and 'B'"):
            KeyMap((a, other))
        with pytest.raises(ValueError, match="share a point"):
            KeyMap((other, a))
    # Strict separation by any positive gap is accepted and lookup stays unique.
    gap = Key("B", "B", "char", 1.0 + 1e-9, 0.0, 2.0, 1.0)
    km2 = KeyMap((a, gap))
    assert km2.lookup(1.0, 0.5) is a
    assert km2.lookup(1.0 + 1e-9, 0.5) is gap
    assert km2.lookup(1.0 + 5e-10, 0.5) is None
    # Degenerate sizes still construct.
    assert len(KeyMap(())) == 0
    assert len(KeyMap((a,))) == 1


def test_key_requires_char_consistent_with_kind():
    with pytest.raises(ValueError, match="exactly one character"):
        Key("Q", None, "char", 0, 0, 1, 1)
    with pytest.raises(ValueError, match="exactly one character"):
        Key("Q", "QQ", "char", 0, 0, 1, 1)
    with pytest.raises(ValueError, match="must not carry"):
        Key("ESC", "e", "other", 0, 0, 1, 1)
    with pytest.raises(ValueError, match="must not carry"):
        Key("BACKSPACE", "\b", "backspace", 0, 0, 1, 1)
    # Well-formed keys of every kind still construct.
    assert Key("Q", "Q", "char", 0, 0, 1, 1).char == "Q"
    assert Key("SPACE", " ", "char", 0, 0, 1, 1).char == " "
    assert Key(";", ";", "char", 0, 0, 1, 1).kind == "char"
    assert Key("BACKSPACE", None, "backspace", 0, 0, 1, 1).char is None
    assert Key("ESC", None, "other", 0, 0, 1, 1).char is None
    # Episode normalises case on its own, so a lowercase Key is still a valid
    # press carrier (test_episode relies on this); only KeyMap insists on
    # uppercase storage.
    assert Key("a", "a", "char", 0, 0, 1, 1).name == "a"


def test_key_rejects_non_finite_coordinates():
    inf, nan = float("inf"), float("nan")
    for rect in (
        (-inf, 0.0, inf, 1.0),
        (0.0, 0.0, inf, 1.0),
        (0.0, -inf, 1.0, 1.0),
        (nan, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, nan),
    ):
        with pytest.raises(ValueError):
            Key("X", "X", "char", *rect)


def test_keymap_stores_letters_uppercase():
    with pytest.raises(ValueError, match="uppercase"):
        KeyMap((Key("q", "q", "char", 0, 0, 1, 1),))
    with pytest.raises(ValueError, match="uppercase"):
        KeyMap((Key("Q", "q", "char", 0, 0, 1, 1),))
    # Digits and punctuation have no case and are unaffected.
    km2 = KeyMap((Key("7", "7", "char", 0, 0, 1, 1), Key(";", ";", "char", 2, 0, 3, 1)))
    assert [k.name for k in km2.alnum()] == ["7"]


def _entry(name, char, kind, x0, y0, x1, y1):
    return {"name": name, "char": char, "kind": kind, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _write_keymap_doc(path, keys):
    doc = {"schema": "autotype_sim/keymap-1", "frame": "board", "units": "m", "keys": keys}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_load_keymap_rejects_malformed_files_at_load(tmp_path):
    p = tmp_path / "hand_edited.yaml"
    good = _entry("A", "A", "char", 0.0, 0.0, 1.0, 1.0)
    cases = {
        "overlap": [good, _entry("B", "B", "char", 0.5, 0.0, 1.5, 1.0)],
        "touching": [good, _entry("B", "B", "char", 1.0, 0.0, 2.0, 1.0)],
        "duplicate names": [good, _entry("A", "A", "char", 2.0, 0.0, 3.0, 1.0)],
        "missing kind": [{k: v for k, v in good.items() if k != "kind"}],
        "missing x1": [{k: v for k, v in good.items() if k != "x1"}],
        "missing name": [{k: v for k, v in good.items() if k != "name"}],
        "char key without char": [_entry("Q", None, "char", 0.0, 0.0, 1.0, 1.0)],
        "other key with char": [_entry("ESC", "e", "other", 0.0, 0.0, 1.0, 1.0)],
        "lowercase letter": [_entry("q", "q", "char", 0.0, 0.0, 1.0, 1.0)],
        "infinite edge": [_entry("A", "A", "char", 0.0, 0.0, float("inf"), 1.0)],
        "degenerate": [_entry("A", "A", "char", 1.0, 0.0, 0.0, 1.0)],
        "bad kind": [_entry("A", "A", "weird", 0.0, 0.0, 1.0, 1.0)],
        "non-numeric coordinate": [_entry("A", "A", "char", "x", 0.0, 1.0, 1.0)],
        "null coordinate": [_entry("A", "A", "char", None, 0.0, 1.0, 1.0)],
        "entry not a mapping": ["A"],
        "keys not a list": {"A": good},
    }
    for label, keys in cases.items():
        _write_keymap_doc(p, keys)
        with pytest.raises(ValueError) as excinfo:
            load_keymap(p)
        assert str(p) in str(excinfo.value), label
    # And a well-formed hand-written file still loads.
    _write_keymap_doc(p, [good, _entry("B", "B", "char", 1.5, 0.0, 2.5, 1.0)])
    km2 = load_keymap(p)
    assert [k.name for k in km2.keys] == ["A", "B"]
    assert km2.lookup(1.25, 0.5) is None


def test_generated_map_passes_its_own_validation(km):
    # The 87-key map is strictly separated (by 2 * inset), all uppercase, and
    # consistent -- reconstructing it from its own keys must not raise.
    assert KeyMap(km.keys) == km
    assert all(k.kind != "char" or len(k.char) == 1 for k in km.keys)


# geometry loader and the CLI --------------------------------------------------


def _write_geometry_yaml(path, geom: dict, *, string_ids: bool = False) -> None:
    doc = dict(geom)
    if string_ids:
        doc["marker_centers"] = {str(k): list(v) for k, v in geom["marker_centers"].items()}
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def test_load_geometry_grid_lies_inside_plate_and_panel(tmp_path, geometry_dict):
    """Whatever geometry is in use: load_geometry gives int marker ids, and the
    generated grid sits inside the plate inside the panel. No coordinate is
    pinned."""
    p = tmp_path / "board_geometry.yaml"
    _write_geometry_yaml(p, geometry_dict)
    geom = load_geometry(p)
    assert set(geom["marker_centers"]) == {0, 1, 2, 3}
    assert all(isinstance(k, int) for k in geom["marker_centers"])
    assert geom["key_pitch"] == pytest.approx(0.01905)  # Cherry MX, the design spec s.4 (public)
    km_real = generate_tkl(geom)
    assert len(km_real) == 87
    assert km_real == generate_tkl(geometry_dict)
    # The key grid lies inside the keyboard plate, which lies inside the panel.
    kx, ky = geom["kb_origin"]
    gx, gy = geom["key_area_origin"]
    assert kx <= gx and ky <= gy
    for k in km_real.keys:
        assert kx <= k.x0 < k.x1 <= kx + geom["kb_w"]
        assert ky <= k.y0 < k.y1 <= ky + geom["kb_h"]
    assert gx + TKL_WIDTH_U * geom["key_pitch"] <= kx + geom["kb_w"] + 1e-12
    assert gy + TKL_HEIGHT_U * geom["key_pitch"] <= ky + geom["kb_h"] + 1e-12
    assert 0.0 <= kx and 0.0 <= ky
    assert kx + geom["kb_w"] <= geom["panel_w"] and ky + geom["kb_h"] <= geom["panel_h"]


def test_load_geometry_tolerates_string_marker_ids(tmp_path, geometry_dict):
    p = tmp_path / "geom.yaml"
    partial = {
        k: geometry_dict[k]
        for k in ("panel_w", "panel_h", "marker_size", "key_pitch", "key_area_origin", "key_inset")
    }
    partial["marker_centers"] = {m: geometry_dict["marker_centers"][m] for m in (0, 1)}
    _write_geometry_yaml(p, partial, string_ids=True)
    geom = load_geometry(p)
    assert geom["marker_centers"] == {
        0: [float(v) for v in geometry_dict["marker_centers"][0]],
        1: [float(v) for v in geometry_dict["marker_centers"][1]],
    }
    assert all(isinstance(k, int) for k in geom["marker_centers"])
    assert generate_tkl(geom) == generate_tkl(geometry_dict)
    # A file that is not a mapping is rejected by name.
    p.write_text("- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_geometry(p)


@pytest.mark.private
def test_private_layout_file_matches_generate_tkl(private_dir, geometry_dict):
    """The site keyboard_layout.yaml must be in step with board_geometry.yaml."""
    layout = private_dir / "keyboard_layout.yaml"
    if not layout.is_file():
        pytest.skip(f"{layout} not present")
    assert load_keymap(layout) == generate_tkl(geometry_dict)
