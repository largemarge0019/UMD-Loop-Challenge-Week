"""Tests for core/episode.py against the design spec section 8.

No wall clock anywhere: episode times are injected constants, and the only
randomness is a seeded ``numpy.random.default_rng`` used to cross-check
``levenshtein`` against an independent full-matrix oracle.
"""

from __future__ import annotations

import dataclasses
import enum
import math

import numpy as np
import pytest
import yaml

from autotype_sim.core.episode import (
    LAUNCH_KEY_PATTERN,
    Episode,
    EpisodeResult,
    PressResultLike,
    levenshtein,
    normalise_launch_key,
)
from autotype_sim.core.keymap import Key, classify

# ---------------------------------------------------------------- helpers


class Reason(enum.Enum):
    """Stand-in for press.Reason: only ``.name`` matters to the episode."""

    ACCEPTED = 0
    NO_INTERSECT = 1
    OUT_OF_REACH = 2
    TOO_CLOSE = 3
    GLANCING = 4
    NO_KEY = 5
    MOVING = 6
    DEBOUNCE = 7


class StrReason(str, enum.Enum):
    """Stand-in with press.Reason's exact shape: a ``str`` mixin whose value
    equals its name and whose ``__str__`` returns the value."""

    NO_KEY = "NO_KEY"
    MOVING = "MOVING"

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class FakeResult:
    """Duck-typed press result: exactly the PressResultLike surface."""

    accepted: bool
    reason: object
    key: Key | None


def key(name: str) -> Key:
    """A Key with the keymap's char/kind rule and a dummy unit rectangle."""
    char, kind = classify(name)
    return Key(name, char, kind, 0.0, 0.0, 1.0, 1.0)


def accepted(name: str) -> FakeResult:
    return FakeResult(True, Reason.ACCEPTED, key(name))


def rejected(reason: object, name: str | None = None) -> FakeResult:
    return FakeResult(False, reason, None if name is None else key(name))


def type_names(ep: Episode, names) -> None:
    """Record an accepted press per entry (a str iterates its characters)."""
    for n in names:
        ep.record_attempt(accepted(n))


def _oracle(a: str, b: str) -> int:
    """Full-matrix Wagner-Fischer, independent of the module's two-row code."""
    m, n = len(a), len(b)
    D = np.zeros((m + 1, n + 1), dtype=np.int64)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + (a[i - 1] != b[j - 1]),
            )
    return int(D[m, n])


def _holds_key(obj) -> bool:
    if isinstance(obj, Key):
        return True
    if isinstance(obj, dict):
        return any(_holds_key(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_holds_key(v) for v in obj)
    return False


# ------------------------------------------------------------ levenshtein


@pytest.mark.parametrize(
    "a, b, d",
    [
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("ab", "ba", 2),
        ("", "", 0),
        ("flaw", "lawn", 2),
        ("ABC", "abc", 3),  # case-sensitive: callers uppercase first
        ("A1B2", "A1B2C3", 2),
        ("HELLO", "HELO", 1),
    ],
)
def test_levenshtein_known_pairs(a, b, d):
    assert levenshtein(a, b) == d
    assert levenshtein(b, a) == d  # symmetric
    assert type(levenshtein(a, b)) is int


def test_levenshtein_matches_oracle_on_random_strings():
    rng = np.random.default_rng(8)
    alphabet = np.array(list("AB1 "))
    for _ in range(300):
        la, lb = (int(v) for v in rng.integers(0, 8, size=2))
        a = "".join(rng.choice(alphabet, size=la))
        b = "".join(rng.choice(alphabet, size=lb))
        d = levenshtein(a, b)
        assert d == _oracle(a, b)
        assert abs(la - lb) <= d <= max(la, lb)


def test_levenshtein_triangle_inequality():
    rng = np.random.default_rng(21)
    alphabet = np.array(list("XYZ0"))
    for _ in range(200):
        a, b, c = (
            "".join(rng.choice(alphabet, size=int(n)))
            for n in rng.integers(0, 7, size=3)
        )
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)


# ------------------------------------------------------- launch key checks


@pytest.mark.parametrize(
    "raw, target",
    [
        ("ABC", "ABC"),
        ("abc", "ABC"),
        ("aBc123", "ABC123"),
        ("000", "000"),
        ("ZZZZZZ", "ZZZZZZ"),
    ],
)
def test_launch_key_accepted_and_uppercased(raw, target):
    ep = Episode(raw, 0.0)
    assert ep.target == target
    assert ep.launch_key == target
    assert normalise_launch_key(raw) == target
    assert LAUNCH_KEY_PATTERN.fullmatch(target)


@pytest.mark.parametrize(
    "bad",
    ["ab", "abcdefg", "a-b", "", "AB C", "ABC\n", " ABC", "A_B", "ßab", "abc!"],
)
def test_launch_key_rejected(bad):
    with pytest.raises(ValueError):
        Episode(bad, 0.0)
    with pytest.raises(ValueError):
        normalise_launch_key(bad)


def test_launch_key_must_be_str():
    with pytest.raises(TypeError):
        Episode(12345, 0.0)  # type: ignore[arg-type]


def test_t_start_is_injected_and_must_be_finite():
    assert Episode("ABC", 7).t_start == 7.0
    assert Episode("ABC").t_start == 0.0  # the design spec section 10 form
    with pytest.raises(ValueError):
        Episode("ABC", float("nan"))
    with pytest.raises(ValueError):
        Episode("ABC", float("inf"))


# ------------------------------------------------------ typed string rules


def test_char_keys_append_in_order():
    ep = Episode("QWERTY", 0.0)
    assert ep.typed == ""
    type_names(ep, "QWE")
    assert ep.typed == "QWE"
    ep.record_attempt(accepted("SPACE"))
    assert ep.typed == "QWE "
    ep.record_attempt(accepted("7"))
    assert ep.typed == "QWE 7"
    assert ep.presses_attempted == 5
    assert ep.presses_accepted == 5


def test_backspace_pops_last_char():
    ep = Episode("ABC", 0.0)
    type_names(ep, "ABX")
    ep.record_attempt(accepted("BACKSPACE"))
    assert ep.typed == "AB"
    type_names(ep, "C")
    assert ep.typed == "ABC"


def test_backspace_on_empty_is_noop_but_counted():
    ep = Episode("ABC", 0.0)
    ep.record_attempt(accepted("BACKSPACE"))
    assert ep.typed == ""
    assert ep.presses_attempted == 1
    assert ep.presses_accepted == 1
    type_names(ep, "A")
    ep.record_attempt(accepted("BACKSPACE"))
    ep.record_attempt(accepted("BACKSPACE"))
    assert ep.typed == ""
    assert ep.presses_accepted == 4


@pytest.mark.parametrize(
    "name", ["ENTER", "LSHIFT", "F5", "TAB", "ESC", "UP", "PRTSC", "CAPS"]
)
def test_other_keys_counted_but_do_not_type(name):
    ep = Episode("ABC", 0.0)
    type_names(ep, "A")
    ep.record_attempt(accepted(name))
    assert key(name).kind == "other"
    assert ep.typed == "A"
    assert ep.presses_attempted == 2
    assert ep.presses_accepted == 2


def test_lowercase_char_is_stored_uppercase():
    ep = Episode("ABC", 0.0)
    ep.record_attempt(
        FakeResult(True, Reason.ACCEPTED, Key("a", "a", "char", 0, 0, 1, 1))
    )
    assert ep.typed == "A"


@pytest.mark.parametrize("char", ["", "AB", "abc", None])
def test_char_key_must_carry_exactly_one_character(char):
    """The design spec s8: ``char`` -> append one character. A multi-character
    ``char`` would be typed as one element and popped by one backspace."""
    ep = Episode("ABC", 0.0)
    type_names(ep, "A")
    with pytest.raises(ValueError):
        ep.apply(Key("MACRO", char, "char", 0, 0, 1, 1))
    with pytest.raises(ValueError):
        ep.record_attempt(
            FakeResult(True, Reason.ACCEPTED, Key("MACRO", char, "char", 0, 0, 1, 1))
        )
    assert ep.typed == "A"
    assert ep.presses_attempted == 1
    assert ep.presses_accepted == 1


@pytest.mark.parametrize("char", ["ß", "ﬁ"])
def test_length_changing_uppercase_keeps_one_char_per_press(char):
    """``str.upper`` expands these to two characters; one press must still
    add exactly one character so that one backspace removes exactly it."""
    assert len(char.upper()) == 2  # the case being guarded
    ep = Episode("ABC", 0.0)
    type_names(ep, "A")
    ep.apply(Key("X", char, "char", 0, 0, 1, 1))
    assert ep.typed == "A" + char
    assert len(ep.typed) == 2
    assert ep.presses_accepted == 2
    ep.apply(key("BACKSPACE"))
    assert ep.typed == "A"


def test_typed_property_is_a_snapshot_string():
    ep = Episode("ABC", 0.0)
    type_names(ep, "AB")
    snap = ep.typed
    type_names(ep, "C")
    assert snap == "AB"
    assert ep.typed == "ABC"


# ------------------------------------------------------------- rejections


def test_rejections_tallied_by_enum_name():
    ep = Episode("ABC", 0.0)
    ep.record_attempt(rejected(Reason.NO_KEY))
    ep.record_attempt(rejected(Reason.MOVING, "A"))
    ep.record_attempt(rejected(Reason.NO_KEY))
    ep.record_attempt(rejected(Reason.DEBOUNCE, "B"))
    ep.record_attempt(rejected(Reason.GLANCING))
    assert ep.typed == ""
    assert ep.presses_attempted == 5
    assert ep.presses_accepted == 0
    assert ep.rejections == {"NO_KEY": 2, "MOVING": 1, "DEBOUNCE": 1, "GLANCING": 1}


def test_rejections_accept_plain_string_reasons():
    ep = Episode("ABC", 0.0)
    ep.record_attempt(rejected("GLANCING"))
    ep.record_attempt(rejected(Reason.GLANCING))
    ep.record_attempt(rejected("TOO_CLOSE"))
    assert ep.rejections == {"GLANCING": 2, "TOO_CLOSE": 1}


def test_rejections_from_str_mixin_enum_are_keyed_by_plain_str():
    """press.Reason is ``(str, Enum)``: the tally must hold its ``.name`` as
    an exact ``str``, not the member itself (which yaml.safe_dump refuses)."""
    ep = Episode("ABC", 0.0)
    ep.record_attempt(rejected(StrReason.NO_KEY))
    ep.record_attempt(rejected(StrReason.MOVING, "A"))
    ep.record_attempt(rejected("NO_KEY"))  # the same reason as a plain str
    ep.record_attempt(rejected(StrReason.NO_KEY))
    res = ep.finish(1.0)
    for tally in (ep.rejections, res.rejections):
        assert tally == {"NO_KEY": 3, "MOVING": 1}
        assert all(type(k) is str for k in tally)
    assert list(res.rejections) == ["MOVING", "NO_KEY"]  # sorted by name
    assert yaml.safe_load(yaml.safe_dump(res.rejections)) == {
        "NO_KEY": 3,
        "MOVING": 1,
    }


def test_rejections_from_real_press_reason_are_plain_str():
    from autotype_sim.core.press import PressResult
    from autotype_sim.core.press import Reason as RealReason

    def result(accepted, reason, k=None):
        return PressResult(accepted, reason, k, (0.0, 0.0), 0.3, 0.1)

    ep = Episode("ABC", 0.0)
    assert isinstance(result(False, RealReason.NO_KEY), PressResultLike)
    ep.record_attempt(result(False, RealReason.NO_KEY))
    ep.record_attempt(result(False, RealReason.MOVING, key("A")))
    ep.record_attempt(result(True, RealReason.ACCEPTED, key("A")))
    ep.record_attempt(result(False, RealReason.NO_KEY))
    res = ep.finish(1.0)
    assert res.typed == "A"
    assert res.presses_attempted == 4
    assert res.presses_accepted == 1
    assert res.rejections == {"MOVING": 1, "NO_KEY": 2}
    assert all(type(k) is str for k in res.rejections)
    # The whole result must round-trip through YAML (RepresenterError before).
    dumped = yaml.safe_load(yaml.safe_dump(dataclasses.asdict(res)))
    assert dumped["rejections"] == {"MOVING": 1, "NO_KEY": 2}


def test_rejected_press_with_key_does_not_type():
    ep = Episode("ABC", 0.0)
    type_names(ep, "A")
    ep.record_attempt(rejected(Reason.DEBOUNCE, "B"))
    ep.record_attempt(rejected(Reason.MOVING, "BACKSPACE"))
    assert ep.typed == "A"
    assert ep.presses_attempted == 3
    assert ep.presses_accepted == 1


def test_rejections_property_is_a_copy():
    ep = Episode("ABC", 0.0)
    ep.record_attempt(rejected(Reason.NO_KEY))
    r = ep.rejections
    r["NO_KEY"] = 99
    r["BOGUS"] = 1
    assert ep.rejections == {"NO_KEY": 1}


def test_accepted_result_without_key_raises_and_counts_nothing():
    ep = Episode("ABC", 0.0)
    with pytest.raises(ValueError):
        ep.record_attempt(FakeResult(True, Reason.ACCEPTED, None))
    assert ep.presses_attempted == 0
    assert ep.presses_accepted == 0
    assert ep.typed == ""


def test_malformed_key_raises_before_counting():
    class Weird:
        name = "MACRO"
        char = "X"
        kind = "macro"

    ep = Episode("ABC", 0.0)
    with pytest.raises(ValueError):
        ep.record_attempt(FakeResult(True, "ACCEPTED", Weird()))
    assert ep.presses_attempted == 0
    assert ep.typed == ""


def test_record_attempt_returns_nothing_and_retains_no_keys():
    ep = Episode("ABC", 0.0)
    assert ep.record_attempt(accepted("A")) is None
    assert ep.record_attempt(rejected(Reason.MOVING, "B")) is None
    ep.apply(key("ENTER"))
    assert not any(_holds_key(v) for v in vars(ep).values())


def test_fake_result_satisfies_protocol():
    assert isinstance(accepted("A"), PressResultLike)
    assert isinstance(rejected(Reason.NO_KEY), PressResultLike)


# ---------------------------------------------------------------- finish


def test_finish_exact_match_after_correction():
    ep = Episode("hello", 100.0)
    type_names(ep, "HELL")
    ep.record_attempt(accepted("P"))  # typo
    ep.record_attempt(accepted("BACKSPACE"))
    ep.record_attempt(rejected(Reason.NO_KEY))
    ep.record_attempt(accepted("O"))
    ep.record_attempt(accepted("ENTER"))  # other: counted only
    res = ep.finish(112.5)
    assert isinstance(res, EpisodeResult)
    assert res.target == "HELLO"
    assert res.typed == "HELLO"
    assert res.exact_match is True
    assert res.edit_distance == 0
    assert res.presses_attempted == 9
    assert res.presses_accepted == 8
    assert res.rejections == {"NO_KEY": 1}
    assert res.elapsed == pytest.approx(12.5)


def test_finish_mismatch_reports_edit_distance():
    ep = Episode("KITTEN", 5.0)
    type_names(ep, "SITTING")
    res = ep.finish(6.0)
    assert res.typed == "SITTING"
    assert res.exact_match is False
    assert res.edit_distance == 3
    assert res.presses_attempted == res.presses_accepted == 7
    assert res.rejections == {}


def test_finish_with_nothing_typed():
    ep = Episode("ABC", 0.0)
    res = ep.finish(0.0)
    assert res.typed == ""
    assert res.exact_match is False
    assert res.edit_distance == 3
    assert res.elapsed == 0.0
    assert res.presses_attempted == 0
    assert res.rejections == {}


def test_exact_match_is_case_and_whitespace_strict():
    ep = Episode("AB1", 0.0)
    type_names(ep, ["A", "B", "1", "SPACE"])
    res = ep.finish(1.0)
    assert res.typed == "AB1 "
    assert res.exact_match is False
    assert res.edit_distance == 1


@pytest.mark.parametrize(
    "t0, t1", [(0.0, 1.0), (1.0e6, 1.0e6 + 42.25), (3.5, 3.5), (-2.0, 0.5)]
)
def test_elapsed_uses_injected_times_only(t0, t1):
    ep = Episode("ABC", t0)
    type_names(ep, "ABC")
    assert ep.finish(t1).elapsed == pytest.approx(t1 - t0)


def test_finish_rejects_time_before_start():
    with pytest.raises(ValueError):
        Episode("ABC", 10.0).finish(9.0)
    with pytest.raises(ValueError):
        Episode("ABC", 10.0).finish(float("nan"))


@pytest.mark.parametrize("t1", [float("inf"), float("-inf"), float("nan")])
def test_finish_rejects_non_finite_time(t1):
    """``t_start`` must be finite; so must ``t_now``, else ``elapsed`` is
    non-finite and the dashboard JSON would carry ``Infinity``."""
    ep = Episode("ABC", 0.0)
    type_names(ep, "A")
    with pytest.raises(ValueError):
        ep.finish(t1)
    assert ep.finished is False  # a rejected finish leaves the episode open
    res = ep.finish(2.0)
    assert math.isfinite(res.elapsed) and res.elapsed == 2.0
    assert res.typed == "A"


def test_finish_rejects_elapsed_overflow():
    ep = Episode("ABC", -1.0e308)
    with pytest.raises(ValueError):
        ep.finish(1.0e308)  # finite - finite overflows to inf
    assert ep.finished is False
    assert math.isfinite(ep.finish(0.0).elapsed)


def test_result_is_frozen_and_detached_from_episode():
    ep = Episode("ABC", 0.0)
    ep.record_attempt(rejected(Reason.NO_KEY))
    res = ep.finish(1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.typed = "X"  # type: ignore[misc]
    res.rejections["NO_KEY"] = 5
    assert ep.rejections == {"NO_KEY": 1}
    assert set(f.name for f in dataclasses.fields(res)) == {
        "target",
        "typed",
        "exact_match",
        "edit_distance",
        "presses_attempted",
        "presses_accepted",
        "rejections",
        "elapsed",
    }


def test_frozen_after_finish():
    ep = Episode("ABC", 0.0)
    type_names(ep, "AB")
    assert ep.finished is False
    res = ep.finish(2.0)
    assert ep.finished is True
    with pytest.raises(RuntimeError):
        ep.record_attempt(accepted("C"))
    with pytest.raises(RuntimeError):
        ep.record_attempt(rejected(Reason.NO_KEY))
    with pytest.raises(RuntimeError):
        ep.apply(key("C"))
    with pytest.raises(RuntimeError):
        ep.finish(3.0)
    assert ep.typed == "AB"
    assert ep.presses_attempted == 2
    assert ep.rejections == {}
    assert res.typed == "AB"


# ---------------------------------------------------------------- apply()


def test_apply_is_an_accepted_press():
    ep = Episode("ABC", 0.0)
    ep.apply(key("A"))
    ep.apply(key("BACKSPACE"))
    ep.apply(key("ESC"))
    ep.apply(key("B"))
    assert ep.typed == "B"
    assert ep.presses_attempted == 4
    assert ep.presses_accepted == 4
    with pytest.raises(ValueError):
        ep.apply(None)  # type: ignore[arg-type]
    assert ep.presses_attempted == 4
