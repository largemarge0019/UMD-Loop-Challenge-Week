"""Episode bookkeeping: the typed string, press tallies and the final score.

An episode starts when a launch key is issued and ends at ``finish()``.
Every ``/arm/press`` outcome is fed in through ``record_attempt``. Only
ACCEPTED presses touch the typed string, following the key's ``kind``
(The design spec section 5):

    char       -> append ``Key.char`` (letters uppercase, SPACE is ``' '``);
                  exactly one character per press, so a ``char`` key whose
                  ``char`` is not a single character is malformed
    backspace  -> drop the last typed character (no-op when already empty)
    other      -> counted as an accepted press, typed string unchanged

Rejected presses are tallied per reason name and never reach the typed
string, even when the result happens to carry the key it would have hit.

Launch keys are 3-6 characters from ``[A-Z0-9]``. Input is uppercased
before validation because letters are stored and compared uppercase
throughout; the typed string is uppercased the same way.

This module deliberately does not import ``core/press.py``. It accepts any
object exposing ``.accepted``, ``.reason`` and ``.key`` (``PressResultLike``)
and depends only on the ``Key`` shape from ``core/keymap.py``.

Timing is injected -- ``t_start`` at construction, ``t_now`` at ``finish`` --
never read from a wall clock, so episodes replay deterministically.

The episode retains no ``Key`` objects and ``record_attempt`` returns
nothing, so nothing about accepted keys leaks back to the caller; the score
arrives only in the ``EpisodeResult`` from ``finish()``. The node must keep
``typed`` to itself until then. After ``finish()`` the episode is frozen and
any further recording raises.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from autotype_sim.core.keymap import KINDS, Key

# The design spec section 5: 3-6 characters from [A-Z0-9], matched with fullmatch
# (so a trailing newline is rejected, unlike a bare ``$``).
LAUNCH_KEY_PATTERN = re.compile(r"[A-Z0-9]{3,6}")


@runtime_checkable
class PressResultLike(Protocol):
    """The slice of ``press.PressResult`` an episode needs.

    ``reason`` may be a plain string or an enum member (its ``.name`` is the
    tally label). ``key`` is the registered ``Key`` for accepted presses; for
    rejected presses it may be ``None`` or a key, and is ignored either way.
    """

    @property
    def accepted(self) -> bool: ...

    @property
    def reason(self) -> object: ...

    @property
    def key(self) -> Key | None: ...


def normalise_launch_key(launch_key: str) -> str:
    """Uppercase ``launch_key`` and require 3-6 characters of ``[A-Z0-9]``.

    Non-ASCII input is rejected outright rather than uppercased, because
    ``str.upper`` can change length (``'ß' -> 'SS'``) and smuggle a
    non-alphanumeric key past the pattern.
    """
    if not isinstance(launch_key, str):
        raise TypeError(
            f"launch_key must be a str, got {type(launch_key).__name__}"
        )
    target = launch_key.upper()
    if not launch_key.isascii() or LAUNCH_KEY_PATTERN.fullmatch(target) is None:
        raise ValueError(
            f"launch key must be 3-6 characters from [A-Z0-9], got {launch_key!r}"
        )
    return target


def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance (insert, delete, substitute; unit costs).

    Case-sensitive: callers compare uppercase strings. Two-row
    Wagner-Fischer, O(len(a) * len(b)) time, O(min(len)) memory.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a  # iterate over the longer string, keep the short row
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,  # delete ca
                    cur[j - 1] + 1,  # insert cb
                    prev[j - 1] + (ca != cb),  # substitute / match
                )
            )
        prev = cur
    return prev[-1]


def _reason_name(reason: object) -> str:
    """Tally label for a rejection: an enum's ``.name``, a str as-is, else ``str()``.

    Enums are tested before ``str`` because ``press.Reason`` mixes in ``str``:
    an ``isinstance(reason, str)`` test first would pass the member itself
    through, keying the tally by ``Reason`` objects instead of the plain
    ``str`` names that ``EpisodeResult.rejections`` promises (``type(k) is
    str`` fails and ``yaml.safe_dump`` refuses them).
    """
    if isinstance(reason, Enum):
        return reason.name
    if isinstance(reason, str):
        return str(reason)  # a str subclass collapses to an exact str
    name = getattr(reason, "name", None)
    return name if isinstance(name, str) else str(reason)


@dataclass(frozen=True)
class EpisodeResult:
    """Final score of one episode (the design spec section 8).

    ``edit_distance`` is the Levenshtein distance between ``typed`` and
    ``target`` (both uppercase); ``exact_match`` is ``typed == target``.
    ``rejections`` maps reason name -> count over every rejected press.
    ``elapsed`` is ``t_finish - t_start`` in seconds of injected time.
    """

    target: str
    typed: str
    exact_match: bool
    edit_distance: int
    presses_attempted: int
    presses_accepted: int
    rejections: dict[str, int]
    elapsed: float


@dataclass(eq=False)
class Episode:
    """One typing episode: target string, typed string and press tallies.

    ``launch_key`` is uppercased and validated (3-6 of ``[A-Z0-9]``) into
    ``target``. Feed every press outcome to ``record_attempt``; call
    ``finish(t_now)`` exactly once for the ``EpisodeResult``, after which
    the episode is frozen. No ``Key`` object is retained between calls.
    """

    launch_key: str
    t_start: float = 0.0
    target: str = field(init=False)
    _typed: list[str] = field(init=False, default_factory=list, repr=False)
    _attempted: int = field(init=False, default=0, repr=False)
    _accepted: int = field(init=False, default=0, repr=False)
    _rejections: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )
    _finished: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.target = normalise_launch_key(self.launch_key)
        self.launch_key = self.target
        self.t_start = float(self.t_start)
        if not math.isfinite(self.t_start):
            raise ValueError(f"t_start must be finite, got {self.t_start}")

    # ---------------------------------------------------------- state view

    @property
    def typed(self) -> str:
        """Typed string so far, uppercase.

        For the node's own bookkeeping only: the design spec forbids showing the
        member anything about accepted keys before ``finish()``.
        """
        return "".join(self._typed)

    @property
    def presses_attempted(self) -> int:
        return self._attempted

    @property
    def presses_accepted(self) -> int:
        return self._accepted

    @property
    def rejections(self) -> dict[str, int]:
        """Copy of the reason-name -> count tally of rejected presses."""
        return dict(self._rejections)

    @property
    def finished(self) -> bool:
        return self._finished

    # -------------------------------------------------------------- inputs

    def record_attempt(self, result: PressResultLike) -> None:
        """Count one ``/arm/press`` outcome.

        Accepted: ``presses_accepted`` grows by one and the key's kind rule
        updates the typed string. Rejected: tallied under the reason's name;
        the typed string is untouched even if the result carries a key.
        Nothing is returned. Raises ``RuntimeError`` once finished, and
        ``ValueError`` (recording nothing) for an accepted result without a
        key or with a malformed one.
        """
        self._check_open()
        if result.accepted:
            key = result.key
            if key is None:
                raise ValueError(
                    "an accepted press must carry the key it registered"
                )
            self._accept(key)
        else:
            name = _reason_name(result.reason)
            self._attempted += 1
            self._rejections[name] = self._rejections.get(name, 0) + 1

    def apply(self, key: Key) -> None:
        """The design spec's ``apply(key)``: record an accepted press of ``key``.

        Identical to ``record_attempt`` with an accepted result -- do not
        call both for the same press.
        """
        self._check_open()
        if key is None:
            raise ValueError("apply() needs the key that was pressed")
        self._accept(key)

    def finish(self, t_now: float) -> EpisodeResult:
        """Score the episode at injected time ``t_now`` and freeze it.

        ``elapsed = t_now - t_start``; ``t_now`` must be finite (like
        ``t_start``) and must not precede it, so ``elapsed`` is always a
        finite non-negative float. A second ``finish`` raises
        ``RuntimeError``; a rejected ``finish`` leaves the episode open.
        """
        self._check_open()
        t_now = float(t_now)
        if not math.isfinite(t_now):
            raise ValueError(f"t_now must be finite, got {t_now!r}")
        elapsed = t_now - self.t_start
        if elapsed < 0.0:
            raise ValueError(
                f"t_now={t_now!r} precedes t_start={self.t_start!r}"
            )
        if not math.isfinite(elapsed):  # finite - finite can still overflow
            raise ValueError(
                f"elapsed overflows for t_now={t_now!r}, t_start={self.t_start!r}"
            )
        typed = self.typed
        self._finished = True
        return EpisodeResult(
            target=self.target,
            typed=typed,
            exact_match=typed == self.target,
            edit_distance=levenshtein(typed, self.target),
            presses_attempted=self._attempted,
            presses_accepted=self._accepted,
            rejections=dict(sorted(self._rejections.items())),
            elapsed=elapsed,
        )

    # ------------------------------------------------------------ internals

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("episode already finished; it is frozen")

    def _accept(self, key: Key) -> None:
        """Apply the kind rule, then count the press (validation first)."""
        self._apply_kind(key)
        self._attempted += 1
        self._accepted += 1

    def _apply_kind(self, key: Key) -> None:
        """Typed-string rule by ``key.kind``; raises before mutating."""
        kind = key.kind
        if kind == "char":
            char = key.char
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError(
                    f"char key {key.name!r} must carry exactly one character, "
                    f"got {char!r}"
                )
            upper = char.upper()
            # str.upper can expand ('ß' -> 'SS'). One press must add exactly
            # one character so that one backspace pops exactly what it added
            # and len(typed) counts char presses; keep such glyphs as-is.
            self._typed.append(upper if len(upper) == 1 else char)
        elif kind == "backspace":
            if self._typed:
                self._typed.pop()
        elif kind == "other":
            pass
        else:
            raise ValueError(
                f"key {key.name!r} has kind {kind!r}; expected one of {KINDS}"
            )
