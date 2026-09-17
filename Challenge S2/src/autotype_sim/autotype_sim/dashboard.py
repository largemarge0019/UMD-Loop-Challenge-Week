"""Browser dashboard bridge: an aiohttp app in a daemon thread (rclpy-free).

The dashboard bridge speaks exactly this protocol::

    GET /                  web_dir/index.html
    GET /static/<file>     web_dir/<file>          (one flat directory, no dotfiles)
    GET /static/texture.png the PNG handed to the constructor (served from memory)
    GET /api/static        the StaticInfo dict handed to the constructor (serialised once)
    WS  /ws                State (text JSON) + Frame (binary: 0x01 + JPEG) pushed to
                           every socket; Command (text JSON) accepted from the browser

Threading convention
--------------------
The node owns one :class:`DashboardServer`. :meth:`DashboardServer.start`
spawns ``threading.Thread(daemon=True)`` running a private asyncio loop; every
aiohttp object lives on that loop. The node thread only ever calls
:meth:`publish_state` / :meth:`publish_frame` (thread-safe "latest value"
slots guarded by a lock) and reads :attr:`client_count`. Each publish sets a
single ``asyncio.Event`` via ``loop.call_soon_threadsafe``; the broadcaster
coroutine wakes on it and fans out to one sender task per socket. A sender
always transmits the *latest* slot values it has not yet sent, so a slow
socket never has more than one pending state and one pending frame -- older
ones are simply overwritten (frames are skipped, never queued).

Commands travel the other way: text frames are JSON-parsed and validated
against the protocol shapes (see :func:`parse_command`) and put on the
``queue.Queue`` the node drains at 50 Hz -- but only when ``teleop`` is true;
otherwise they are counted and dropped. Non-JSON text closes that socket with
close code 1003 (unsupported data).

The server only forwards what the node hands it (``static_info`` and the
State dicts are the protocol shapes; the board pose in State is what the
dashboard is *for* and it never touches a ROS topic). For unit tests
:func:`make_app` exposes
the bare ``web.Application`` so aiohttp's test utilities can drive it in
process, without the thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
from aiohttp import WSCloseCode, WSMsgType, web

from autotype_sim.core.config import JOINT_NAMES

__all__ = [
    "COMMAND_TYPES",
    "FRAME_MAGIC",
    "MAX_JOG_DEG_S",
    "DashboardServer",
    "encode_jpeg",
    "make_app",
    "parse_command",
]

FRAME_MAGIC = b"\x01"
"""First byte of every binary WS frame (the dashboard protocol "Frame")."""

COMMAND_TYPES = ("jog", "stop", "press", "done", "reset")
"""Command ``type`` values the browser may send (the dashboard protocol "Command")."""

WS_CLOSE_UNSUPPORTED = 1003  # == WSCloseCode.UNSUPPORTED_DATA: sent on non-JSON text
WS_HEARTBEAT_S = 10.0
WS_MAX_MSG_BYTES = 1 << 20
LOG_THROTTLE_S = 1.0
STOP_JOIN_TIMEOUT_S = 5.0
SHUTDOWN_TIMEOUT_S = 1.0

_MAX_SEED = 2**63 - 1  # fits int64 / numpy default_rng

# Explicit media types for the flat web directory. Python's ``mimetypes``
# reads the host's /etc/mime.types, which still says application/javascript
# on some distributions; browsers accept both but the protocol test pins
# text/javascript, and the dashboard must look identical on every host.
_MEDIA_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


# --------------------------------------------------------------------------- #
# JPEG helper
# --------------------------------------------------------------------------- #


def encode_jpeg(
    frame_bgr: np.ndarray, size: tuple[int, int] = (640, 360), quality: int = 70
) -> bytes:
    """JPEG-encode a BGR uint8 image after resizing it to ``size`` = ``(width, height)``.

    ``size`` follows OpenCV's ``(w, h)`` order. The resize (``INTER_AREA``, the
    right filter for downscaling a rendered camera image) is skipped when the
    frame already has that size. Returns the raw JPEG bytes *without* the
    ``0x01`` frame prefix -- :meth:`DashboardServer.publish_frame` adds it.
    """
    img = np.ascontiguousarray(frame_bgr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    w, h = int(size[0]), int(size[1])
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover - imencode only fails on unsupported input
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Command validation
# --------------------------------------------------------------------------- #


#: Largest |velocity_deg_s| a jog command may carry. Python ints are
#: unbounded and JSON has no integer limit, so a literal like ``10**400``
#: parses fine and only explodes later, in the node's 50 Hz timer, as
#: ``OverflowError: int too large to convert to float``. Bound it here, at the
#: edge: anything past this is not a jog, it is a malformed command. The limit
#: is far above any usable rate (v_max is ~60 deg/s) and the node clips to
#: v_max anyway.
MAX_JOG_DEG_S = 1e6


def _is_number(value: Any) -> bool:
    """True for a finite int/float that is not a bool (JSON ``true`` is not a velocity).

    Unbounded Python ints are rejected here rather than downstream: a value
    that cannot survive ``float()`` is not a number this protocol accepts.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    return isinstance(value, float) and math.isfinite(value)


def parse_command(obj: Any) -> dict[str, Any] | None:
    """Validate a decoded JSON value against the protocol Command shapes.

    Returns a normalised dict holding only the protocol keys, or ``None`` when
    the shape is wrong (not an object, unknown ``type``, missing/mistyped
    fields). Extra keys are ignored. Shapes (the dashboard protocol)::

        {"type": "jog", "joint": <one of JOINT_NAMES>, "velocity_deg_s": <finite number>}
        {"type": "stop"} | {"type": "press"} | {"type": "done"}
        {"type": "reset"} | {"type": "reset", "seed": <int >= 0>}

    Numbers keep their JSON type (``10`` stays ``10``); the node converts
    degrees to radians itself.
    """
    if not isinstance(obj, Mapping):
        return None
    kind = obj.get("type")
    if not isinstance(kind, str) or kind not in COMMAND_TYPES:
        return None
    if kind == "jog":
        joint = obj.get("joint")
        velocity = obj.get("velocity_deg_s")
        if not isinstance(joint, str) or joint not in JOINT_NAMES or not _is_number(velocity):
            return None
        if not -MAX_JOG_DEG_S <= velocity <= MAX_JOG_DEG_S:
            return None
        return {"type": "jog", "joint": joint, "velocity_deg_s": velocity}
    if kind == "reset":
        if "seed" not in obj or obj["seed"] is None:
            return {"type": "reset"}
        seed = obj["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
            return None
        return {"type": "reset", "seed": seed}
    return {"type": kind}


# --------------------------------------------------------------------------- #
# JSON safety for State dicts
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    """Recursively turn a State dict into plain JSON types.

    numpy scalars/arrays become Python numbers/lists and non-finite floats
    become ``null`` (``NaN`` is not JSON and ``JSON.parse`` in the page would
    drop the whole State). Anything else unknown is stringified rather than
    killing the 20 Hz publisher.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class _Client:
    """One connected browser: its socket plus the per-socket wake-up flag."""

    ws: web.WebSocketResponse
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class DashboardServer:
    """aiohttp dashboard server with thread-safe "latest value" inputs.

    Parameters
    ----------
    port
        TCP port; ``0`` asks the OS for an ephemeral one (see :attr:`port`).
    web_dir
        Directory holding ``index.html``, ``app.js``, ``style.css``.
    static_info
        The StaticInfo dict (``SimCore.static_info()``); serialised once.
    texture_png
        PNG bytes of ``SimCore.texture`` served at ``/static/texture.png``.
    teleop
        When false every Command is counted and dropped, never enqueued.
    command_queue
        ``queue.Queue`` the node drains; receives the dicts from :func:`parse_command`.
    host
        Bind address (default all interfaces, as the protocol states).
    logger
        Anything with ``info``/``warning``/``error`` taking one pre-formatted
        string (a ``logging.Logger`` or an rclpy logger). Default: ``logging``.
    """

    def __init__(
        self,
        *,
        port: int,
        web_dir: Path,
        static_info: dict,
        texture_png: bytes,
        teleop: bool,
        command_queue: "queue.Queue[dict[str, Any]]",
        host: str = "0.0.0.0",
        logger: Any = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.web_dir = Path(web_dir)
        self.teleop = bool(teleop)
        self.command_queue = command_queue
        self._static_json = json.dumps(_jsonable(dict(static_info)), separators=(",", ":"))
        self._texture_png = bytes(texture_png)
        self._log = logger if logger is not None else logging.getLogger("autotype_sim.dashboard")

        # Latest-value slots (node thread writes, loop thread reads).
        self._slot_lock = threading.Lock()
        self._state_json: str | None = None
        self._state_seq = 0
        self._frame: bytes | None = None
        self._frame_seq = 0

        # Loop-side objects, created on the loop in _on_startup.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._broadcaster: asyncio.Task | None = None
        self._clients: set[_Client] = set()

        # Thread-side objects.
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None

        # Counters (informational; loop thread writes).
        self.stats: dict[str, int] = {
            "commands_enqueued": 0,
            "commands_rejected": 0,  # valid JSON, wrong shape
            "commands_ignored": 0,  # teleop off
            "bad_json": 0,
            "binary_dropped": 0,
        }
        self._throttle: dict[str, float] = {}
        self._clock: Callable[[], float] = time.monotonic

    # -- public, any thread ------------------------------------------------ #

    @property
    def client_count(self) -> int:
        """Number of open dashboard sockets."""
        return len(self._clients)

    @property
    def url(self) -> str:
        """``http://host:port`` of the running site (host as given)."""
        return f"http://{self.host}:{self.port}"

    def publish_state(self, state: dict) -> None:
        """Store the latest State (JSON-encoded here) and wake the broadcaster.

        Thread-safe; only the newest State is ever sent. The dict is
        converted with :func:`_jsonable` at call time, so later mutation by
        the caller cannot leak into the frame.
        """
        text = json.dumps(_jsonable(state), separators=(",", ":"))
        with self._slot_lock:
            self._state_json = text
            self._state_seq += 1
        self._signal()

    def publish_frame(self, jpeg: bytes) -> None:
        """Store the latest camera JPEG (prefixed with ``0x01``) and wake the broadcaster.

        Thread-safe; a socket that has not yet sent the previous frame skips it.
        """
        data = FRAME_MAGIC + bytes(jpeg)
        with self._slot_lock:
            self._frame = data
            self._frame_seq += 1
        self._signal()

    def start(self) -> None:
        """Spawn the daemon thread and return once the site is listening.

        Raises ``OSError`` (port in use, bad host) or ``RuntimeError`` (already
        started / loop failure) instead of hanging. After a successful return
        :attr:`port` holds the bound port (resolved when ``0`` was requested).
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("DashboardServer already started")
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._thread_main, name="dashboard", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            self._thread.join(timeout=STOP_JOIN_TIMEOUT_S)
            self._thread = None
            err = self._start_error
            self._start_error = None
            raise err

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> None:
        """Close every socket, stop the site and join the thread (idempotent)."""
        thread = self._thread
        if thread is None:
            return
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None:
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except RuntimeError:  # loop already closed
                pass
        thread.join(timeout=timeout)
        if thread.is_alive():
            self._log.error("dashboard thread did not stop within %.1f s" % timeout)
        else:
            self._thread = None

    # -- thread body -------------------------------------------------------- #

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # pragma: no cover - defensive
            if not self._ready.is_set():
                self._start_error = exc
            else:
                self._log.error("dashboard loop crashed: %r" % (exc,))
        finally:
            self._ready.set()
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        runner = web.AppRunner(
            make_app(self), handle_signals=False, access_log=None, shutdown_timeout=SHUTDOWN_TIMEOUT_S
        )
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, reuse_address=True)
        try:
            await site.start()
        except OSError as exc:
            self._start_error = exc
            self._ready.set()
            await runner.cleanup()
            return
        addresses = runner.addresses
        if addresses:
            self.port = int(addresses[0][1])
        self._log.info("dashboard listening on %s" % self.url)
        self._ready.set()
        try:
            await self._stop_event.wait()
        finally:
            await runner.cleanup()
            self._log.info("dashboard stopped")

    # -- loop side ---------------------------------------------------------- #

    def _signal(self) -> None:
        """Set the broadcaster's event from any thread (no-op before startup)."""
        loop, wake = self._loop, self._wake
        if loop is None or wake is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:  # closed between the check and the call
            pass

    async def _on_startup(self, app: web.Application) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._broadcaster = asyncio.create_task(self._broadcast_loop(), name="dashboard-broadcaster")
        # Anything published before the loop existed is waiting in the slots.
        self._wake.set()

    async def _on_shutdown(self, app: web.Application) -> None:
        if self._broadcaster is not None:
            self._broadcaster.cancel()
            try:
                await self._broadcaster
            except (asyncio.CancelledError, Exception):
                pass
            self._broadcaster = None
        clients = list(self._clients)
        if clients:
            await asyncio.gather(
                *(c.ws.close(code=WSCloseCode.GOING_AWAY, message=b"server shutdown") for c in clients),
                return_exceptions=True,
            )

    async def _broadcast_loop(self) -> None:
        """Wake on the shared event, then poke every per-socket sender."""
        assert self._wake is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            for client in list(self._clients):
                client.wake.set()

    async def _sender(self, client: _Client) -> None:
        """Send the newest unsent State / Frame to one socket, forever.

        Reads the slots only when woken, and compares sequence numbers so a
        socket that fell behind transmits just the latest item of each kind.
        """
        sent_state = 0
        sent_frame = 0
        ws = client.ws
        try:
            while not ws.closed:
                await client.wake.wait()
                client.wake.clear()
                with self._slot_lock:
                    state_json, state_seq = self._state_json, self._state_seq
                    frame, frame_seq = self._frame, self._frame_seq
                if state_json is not None and state_seq > sent_state:
                    await ws.send_str(state_json)
                    sent_state = state_seq
                if frame is not None and frame_seq > sent_frame:
                    await ws.send_bytes(frame)
                    sent_frame = frame_seq
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # peer vanished mid-send
            self._log_throttled("send", "warning", "dashboard send failed: %r" % (exc,))
            if not ws.closed:
                try:
                    await ws.close(code=WSCloseCode.INTERNAL_ERROR)
                except Exception:
                    pass

    async def _ws_handler(self, request: web.Request) -> web.StreamResponse:
        # compress=False on purpose: with permessage-deflate negotiated, aiohttp's reader
        # (seen on 3.14.3) rejects a compressed client frame that follows a control frame --
        # e.g. the browser's pong to our heartbeat ping -- and closes the socket with 1002.
        # The stream is JPEG-dominated so compression buys nothing anyway.
        ws = web.WebSocketResponse(heartbeat=WS_HEARTBEAT_S, max_msg_size=WS_MAX_MSG_BYTES, compress=False)
        await ws.prepare(request)
        client = _Client(ws)
        self._clients.add(client)
        client.task = asyncio.create_task(self._sender(client), name="dashboard-sender")
        client.wake.set()  # newcomers get the latest State/Frame immediately
        self._log.info("dashboard client connected (%d total)" % len(self._clients))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._on_text(ws, msg.data)
                elif msg.type == WSMsgType.BINARY:
                    self.stats["binary_dropped"] += 1
                elif msg.type == WSMsgType.ERROR:
                    self._log_throttled("ws_error", "warning", "dashboard socket error: %r" % (ws.exception(),))
        finally:
            self._clients.discard(client)
            if client.task is not None:
                client.task.cancel()
                try:
                    await client.task
                except (asyncio.CancelledError, Exception):
                    pass
            self._log.info("dashboard client gone (%d total)" % len(self._clients))
        return ws

    async def _on_text(self, ws: web.WebSocketResponse, data: str) -> None:
        """Validate one Command frame; enqueue only when teleop is on."""
        try:
            obj = json.loads(data)
        except ValueError:
            self.stats["bad_json"] += 1
            self._log_throttled("bad_json", "warning", "dashboard: non-JSON text frame, closing socket (1003)")
            await ws.close(code=WS_CLOSE_UNSUPPORTED, message=b"invalid JSON")
            return
        cmd = parse_command(obj)
        if cmd is None:
            self.stats["commands_rejected"] += 1
            self._log_throttled("bad_shape", "warning", "dashboard: malformed command %s" % data[:120])
            return
        if not self.teleop:
            self.stats["commands_ignored"] += 1
            return
        self.command_queue.put(cmd)
        self.stats["commands_enqueued"] += 1

    # -- HTTP handlers ------------------------------------------------------ #

    async def _index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(
            self.web_dir / "index.html",
            headers={"Content-Type": _MEDIA_TYPES[".html"], "Cache-Control": "no-cache"},
        )

    async def _texture(self, request: web.Request) -> web.StreamResponse:
        return web.Response(
            body=self._texture_png, content_type="image/png", headers={"Cache-Control": "no-cache"}
        )

    async def _api_static(self, request: web.Request) -> web.StreamResponse:
        return web.Response(
            text=self._static_json, content_type="application/json", headers={"Cache-Control": "no-store"}
        )

    async def _static_file(self, request: web.Request) -> web.StreamResponse:
        """One file from the flat web directory; no dotfiles, no traversal."""
        name = request.match_info["name"]
        if not name or name.startswith(".") or "/" in name or "\\" in name:
            raise web.HTTPNotFound()
        root = self.web_dir.resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            raise web.HTTPNotFound()
        headers = {"Cache-Control": "no-cache"}
        media = _MEDIA_TYPES.get(path.suffix.lower())
        if media is not None:
            headers["Content-Type"] = media
        return web.FileResponse(path, headers=headers)

    # -- logging ------------------------------------------------------------ #

    def _log_throttled(self, key: str, level: str, message: str) -> None:
        """Emit ``message`` at most once per ``LOG_THROTTLE_S`` per ``key``."""
        now = self._clock()
        last = self._throttle.get(key)
        if last is not None and now - last < LOG_THROTTLE_S:
            return
        self._throttle[key] = now
        getattr(self._log, level)(message)


def make_app(server: DashboardServer) -> web.Application:
    """Build the aiohttp application bound to ``server`` (routes per the dashboard protocol).

    Used by :meth:`DashboardServer.start` on its private loop and by tests via
    ``aiohttp.test_utils.TestServer`` on the test's loop; the broadcaster is
    started in ``on_startup`` and every socket is closed in ``on_shutdown``.
    ``/static/texture.png`` is registered before the generic ``/static/{name}``
    route so the in-memory PNG wins over any file of that name.
    """
    app = web.Application()
    app.router.add_get("/", server._index)
    app.router.add_get("/static/texture.png", server._texture)
    app.router.add_get("/static/{name}", server._static_file)
    app.router.add_get("/api/static", server._api_static)
    app.router.add_get("/ws", server._ws_handler)
    app.on_startup.append(server._on_startup)
    app.on_shutdown.append(server._on_shutdown)
    return app
