"""Tests for autotype_sim/dashboard.py.

The aiohttp application is driven in-process with ``aiohttp.test_utils``
(``TestServer`` / ``TestClient`` on a per-test ``asyncio.run`` loop -- no
pytest-asyncio plugin is required); one test exercises the real daemon
thread on an ephemeral port. The StaticInfo handed to the server is a small
hand-written dict and the texture is a synthetic PNG, so the protocol layer is
tested without any geometry at all. Timing appears only as generous timeouts that bound a hang;
no assertion depends on the wall clock (the log throttle test injects its
own clock).
"""

from __future__ import annotations

import asyncio
import json
import queue
import socket
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from autotype_sim.core.config import JOINT_NAMES
from autotype_sim.dashboard import (
    COMMAND_TYPES,
    FRAME_MAGIC,
    DashboardServer,
    encode_jpeg,
    make_app,
    parse_command,
)

WEB_DIR = Path(__file__).resolve().parents[1] / "autotype_sim" / "web"
RECV_TIMEOUT = 5.0  # upper bound on any single receive; never reached when the code is right

STATIC_INFO = {
    "joint_names": list(JOINT_NAMES),
    "links": {"base_height": 0.3, "upper_arm": 0.6, "forearm": 0.4},
    "panel": {"w": 0.4, "h": 0.175},
    "keys": [{"name": "A", "kind": "char", "rect": [0.01, 0.02, 0.03, 0.04]}],
    "texture": {"px_per_m": 4000, "w": 1600, "h": 700},
    "teleop": False,
    "launch_key": "ROVER",
    "seed": 1,
}


def _texture_png() -> bytes:
    img = np.zeros((7, 16, 3), dtype=np.uint8)
    img[:, ::2] = (255, 128, 0)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


TEXTURE_PNG = _texture_png()


class _Log:
    """Minimal logger double with the three methods the server uses."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.records.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.records.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.records.append(("error", msg))


def make_server(*, teleop: bool = False, port: int = 0, host: str = "127.0.0.1") -> DashboardServer:
    return DashboardServer(
        port=port,
        web_dir=WEB_DIR,
        static_info=STATIC_INFO,
        texture_png=TEXTURE_PNG,
        teleop=teleop,
        command_queue=queue.Queue(),
        host=host,
        logger=_Log(),
    )


def state(t: float, **extra) -> dict:
    return {"t": t, "episode": {"status": "running", "seed": 1, "launch_key": "ROVER"}, **extra}


async def recv(ws: aiohttp.ClientWebSocketResponse) -> aiohttp.WSMessage:
    return await asyncio.wait_for(ws.receive(), RECV_TIMEOUT)


async def recv_until_text(ws: aiohttp.ClientWebSocketResponse, predicate, limit: int = 50) -> tuple[list[dict], int]:
    """Collect text frames until ``predicate(dict)`` holds; returns (texts, total_messages)."""
    texts: list[dict] = []
    total = 0
    while total < limit:
        msg = await recv(ws)
        total += 1
        if msg.type == WSMsgType.TEXT:
            d = json.loads(msg.data)
            texts.append(d)
            if predicate(d):
                return texts, total
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            raise AssertionError(f"socket closed early: {msg}")
    raise AssertionError("predicate never satisfied")


# --------------------------------------------------------------------------- #
# 1. HTTP routes
# --------------------------------------------------------------------------- #


def test_http_routes_serve_page_static_and_api():
    server = make_server()

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            r = await client.get("/")
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/html")
            assert await r.read() == (WEB_DIR / "index.html").read_bytes()

            r = await client.get("/static/app.js")
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/javascript")
            assert await r.read() == (WEB_DIR / "app.js").read_bytes()

            r = await client.get("/static/style.css")
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/css")

            r = await client.get("/api/static")
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("application/json")
            assert await r.json() == STATIC_INFO

            r = await client.get("/static/texture.png")
            assert r.status == 200
            assert r.headers["Content-Type"] == "image/png"
            assert await r.read() == TEXTURE_PNG

    asyncio.run(run())


def test_static_route_refuses_traversal_and_missing_files():
    server = make_server()

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            for path in ("/static/nope.js", "/static/.hidden", "/static/..%2Fdashboard.py", "/static/../setup.py"):
                r = await client.get(path)
                assert r.status == 404, path
            r = await client.get("/static/")
            assert r.status == 404

    asyncio.run(run())


def test_static_info_is_serialised_once_and_not_live():
    info = dict(STATIC_INFO)
    server = make_server()
    server_info = DashboardServer(
        port=0, web_dir=WEB_DIR, static_info=info, texture_png=TEXTURE_PNG,
        teleop=False, command_queue=queue.Queue(), logger=_Log(),
    )
    info["seed"] = 999  # mutation after construction must not leak

    async def run():
        async with TestClient(TestServer(make_app(server_info))) as client:
            assert (await (await client.get("/api/static")).json())["seed"] == STATIC_INFO["seed"]

    asyncio.run(run())
    del server


# --------------------------------------------------------------------------- #
# 2. WebSocket push
# --------------------------------------------------------------------------- #


def test_ws_state_and_frame_push():
    server = make_server()

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            ws = await client.ws_connect("/ws")
            assert server.client_count == 1

            st = state(1.5, q_deg=[0.0, 1.0, 2.0, 3.0, 4.0], reticle_px=None, reachable=[True, False])
            server.publish_state(st)
            msg = await recv(ws)
            assert msg.type == WSMsgType.TEXT
            assert json.loads(msg.data) == st

            jpeg = b"\xff\xd8\xff\xe0" + bytes(range(32)) + b"\xff\xd9"
            server.publish_frame(jpeg)
            msg = await recv(ws)
            assert msg.type == WSMsgType.BINARY
            assert msg.data[:1] == FRAME_MAGIC == b"\x01"
            assert msg.data[1:] == jpeg

            # Burst: at most one message per publish, at least one, last equals the latest.
            states = [state(10.0 + i) for i in range(5)]
            for s in states:
                server.publish_state(s)
            texts, total = await recv_until_text(ws, lambda d: d == states[-1])
            assert 1 <= total <= 5
            assert texts[-1] == states[-1]
            # what arrived is a subsequence of what was published, in order
            idx = [states.index(t) for t in texts]
            assert idx == sorted(idx) and len(set(idx)) == len(idx)

            await ws.close()

    asyncio.run(run())
    assert server.client_count == 0


def test_ws_new_client_gets_latest_state_and_frame_and_frame_is_not_resent():
    server = make_server()

    async def run():
        st = state(3.0, typed="RO")
        server.publish_state(st)  # before the loop even exists
        server.publish_frame(b"\xff\xd8\x01")
        async with TestClient(TestServer(make_app(server))) as client:
            ws = await client.ws_connect("/ws")
            m1, m2 = await recv(ws), await recv(ws)
            kinds = {m1.type: m1, m2.type: m2}
            assert set(kinds) == {WSMsgType.TEXT, WSMsgType.BINARY}
            assert json.loads(kinds[WSMsgType.TEXT].data) == st
            assert kinds[WSMsgType.BINARY].data == b"\x01\xff\xd8\x01"

            # A new state alone must not re-send the old frame.
            st2 = state(3.05, typed="RO")
            server.publish_state(st2)
            msg = await recv(ws)
            assert msg.type == WSMsgType.TEXT and json.loads(msg.data) == st2
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.receive(), 0.2)

            # Two sockets both receive.
            ws2 = await client.ws_connect("/ws")
            assert server.client_count == 2
            first = await recv(ws2)
            assert first.type in (WSMsgType.TEXT, WSMsgType.BINARY)
            await ws.close()
            await ws2.close()

    asyncio.run(run())


def test_state_json_is_sanitised():
    server = make_server()

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            ws = await client.ws_connect("/ws")
            server.publish_state({
                "t": np.float64(1.25),
                "q_deg": np.array([1.0, 2.0]),
                "n": np.int64(3),
                "flag": np.bool_(True),
                "reticle_px": float("nan"),
                "nested": {"inf": float("inf"), "ok": (1, 2)},
            })
            msg = await recv(ws)
            assert json.loads(msg.data) == {
                "t": 1.25, "q_deg": [1.0, 2.0], "n": 3, "flag": True,
                "reticle_px": None, "nested": {"inf": None, "ok": [1, 2]},
            }
            await ws.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# 3. Commands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"type": "jog", "joint": "head_pan", "velocity_deg_s": 10}, {"type": "jog", "joint": "head_pan", "velocity_deg_s": 10}),
        ({"type": "jog", "joint": "base_yaw", "velocity_deg_s": -2.5, "extra": 1}, {"type": "jog", "joint": "base_yaw", "velocity_deg_s": -2.5}),
        ({"type": "stop"}, {"type": "stop"}),
        ({"type": "press"}, {"type": "press"}),
        ({"type": "done"}, {"type": "done"}),
        ({"type": "reset"}, {"type": "reset"}),
        ({"type": "reset", "seed": None}, {"type": "reset"}),
        ({"type": "reset", "seed": 7}, {"type": "reset", "seed": 7}),
    ],
)
def test_parse_command_accepts_protocol_shapes(raw, expected):
    assert parse_command(raw) == expected
    assert expected["type"] in COMMAND_TYPES


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "jog"},
        {"type": "jog", "joint": "head_pan"},
        {"type": "jog", "velocity_deg_s": 1},
        {"type": "jog", "joint": "wrist", "velocity_deg_s": 1},
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": "10"},
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": True},
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": float("nan")},
        # Python ints are unbounded and JSON has no int limit: an out-of-range
        # value must die here, not later as an OverflowError inside the node's
        # 50 Hz timer (which kills the process mid-episode).
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": 10**400},
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": -(10**400)},
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": 10**9},
        {"type": "jog", "joint": "head_pan", "velocity_deg_s": 1e300},
        {"type": "reset", "seed": -1},
        {"type": "reset", "seed": 1.5},
        {"type": "reset", "seed": True},
        {"type": "reset", "seed": "7"},
        {"type": "fly"},
        {"type": 3},
        {},
        [],
        "press",
        None,
        42,
    ],
)
def test_parse_command_rejects_bad_shapes(raw):
    assert parse_command(raw) is None


def test_ws_commands_enqueued_only_when_teleop():
    server = make_server(teleop=True)

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "jog", "joint": "head_pan", "velocity_deg_s": 10})
            cmd = await asyncio.get_running_loop().run_in_executor(None, server.command_queue.get, True, RECV_TIMEOUT)
            assert cmd == {"type": "jog", "joint": "head_pan", "velocity_deg_s": 10}

            await ws.send_json({"type": "jog"})  # malformed: no joint
            await ws.send_json({"type": "reset", "seed": 7})  # valid, proves the bad one was skipped
            cmd = await asyncio.get_running_loop().run_in_executor(None, server.command_queue.get, True, RECV_TIMEOUT)
            assert cmd == {"type": "reset", "seed": 7}
            assert server.command_queue.empty()
            assert server.stats["commands_rejected"] == 1
            assert server.stats["commands_enqueued"] == 2
            assert not ws.closed  # a bad shape does not cost the connection
            await ws.close()

    asyncio.run(run())


def test_ws_commands_ignored_without_teleop():
    server = make_server(teleop=False)

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "jog", "joint": "head_pan", "velocity_deg_s": 10})
            await ws.send_json({"type": "press"})
            # Round-trip through the server so both commands are processed before we look.
            server.publish_state(state(0.0))
            await recv(ws)
            # The handler processes frames in order; publish once more to be sure both were seen.
            server.publish_state(state(0.05))
            await recv(ws)
            assert server.command_queue.empty()
            assert server.stats["commands_ignored"] == 2
            assert server.stats["commands_enqueued"] == 0
            await ws.close()

    asyncio.run(run())


def test_ws_invalid_json_closes_socket_with_1003_and_log_is_throttled():
    server = make_server(teleop=True)
    fake_now = [100.0]
    server._clock = lambda: fake_now[0]

    async def run():
        async with TestClient(TestServer(make_app(server))) as client:
            ws = await client.ws_connect("/ws")
            await ws.send_str("{not json")
            msg = await recv(ws)
            assert msg.type == WSMsgType.CLOSE
            assert msg.data == 1003
            assert ws.close_code == 1003
            assert server.command_queue.empty()

            # second offender within the same second: closed, but not logged again
            ws2 = await client.ws_connect("/ws")
            await ws2.send_str("nope")
            msg = await recv(ws2)
            assert msg.type == WSMsgType.CLOSE and msg.data == 1003
            fake_now[0] += 1.0
            ws3 = await client.ws_connect("/ws")
            await ws3.send_str("still nope")
            msg = await recv(ws3)
            assert msg.type == WSMsgType.CLOSE and msg.data == 1003

    asyncio.run(run())
    assert server.stats["bad_json"] == 3
    logged = [m for lvl, m in server._log.records if lvl == "warning" and "non-JSON" in m]
    assert len(logged) == 2
    assert server.client_count == 0


# --------------------------------------------------------------------------- #
# 4. Threaded server
# --------------------------------------------------------------------------- #


def test_threaded_server_start_serve_stop():
    server = make_server(teleop=False, port=0)
    server.start()
    try:
        assert server.port != 0
        assert server._thread is not None and server._thread.is_alive()

        async def run():
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{server.port}/api/static") as r:
                    assert r.status == 200
                    assert await r.json() == STATIC_INFO
                async with session.ws_connect(f"http://127.0.0.1:{server.port}/ws", compress=0) as ws:
                    st = state(2.0, typed="R")
                    server.publish_state(st)  # from the "ROS" (test) thread
                    server.publish_frame(b"\xff\xd8\x00")
                    msgs = [await recv(ws), await recv(ws)]
                    by_type = {m.type: m for m in msgs}
                    assert json.loads(by_type[WSMsgType.TEXT].data) == st
                    assert by_type[WSMsgType.BINARY].data == b"\x01\xff\xd8\x00"
                    assert server.client_count == 1
                    await ws.send_json({"type": "press"})  # teleop off: counted, not enqueued
                    # keep the socket open across stop() to prove shutdown closes it
                    server.stop(timeout=2.0)
                    assert server._thread is None  # joined within 2 s
                    msg = await recv(ws)
                    assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)

        asyncio.run(run())
        assert server.command_queue.empty()
        assert server.stats["commands_ignored"] == 1
    finally:
        server.stop(timeout=2.0)
    server.stop()  # idempotent


def test_start_raises_promptly_when_port_in_use():
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        server = make_server(port=port)
        with pytest.raises(OSError):
            server.start()
        assert server._thread is None
        server.stop()  # nothing to stop; must not raise
    finally:
        blocker.close()


def test_start_twice_is_an_error():
    server = make_server(port=0)
    server.start()
    try:
        with pytest.raises(RuntimeError):
            server.start()
    finally:
        server.stop(timeout=2.0)


# --------------------------------------------------------------------------- #
# encode_jpeg
# --------------------------------------------------------------------------- #


def test_encode_jpeg_resizes_and_produces_decodable_jpeg():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    data = encode_jpeg(frame)
    assert isinstance(data, bytes)
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (360, 640, 3)

    small = encode_jpeg(np.full((360, 640, 3), 90, np.uint8), quality=30)
    big = encode_jpeg(np.full((360, 640, 3), 90, np.uint8), quality=95)
    assert len(small) <= len(big)

    same = encode_jpeg(frame[:36, :64], size=(64, 36))
    assert cv2.imdecode(np.frombuffer(same, np.uint8), cv2.IMREAD_COLOR).shape == (36, 64, 3)
