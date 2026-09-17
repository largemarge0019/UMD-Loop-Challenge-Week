"""``sim_node``: the rclpy shell around :class:`autotype_sim.sim.SimCore`.

Everything that is not ROS message
marshalling lives in ``sim.py`` (SimCore) and ``dashboard.py``
(DashboardServer); this module only owns parameters, timers, publishers,
subscribers, the ``/sim/reset`` service, TF broadcasting and the dashboard
lifecycle. Every callback is short and delegates to ``SimCore``.

Conventions
-----------
* Time ``t`` handed to ``SimCore`` is the node clock (``get_clock().now()``)
  in float seconds; ``SimCore`` subtracts the episode start itself. No
  ``use_sim_time`` -- the simulator runs in real time.
* Units on the wire follow REP-103: radians, metres, seconds; the
  dashboard's degrees are produced by ``SimCore.static_info/snapshot``.
* TF entries from ``SimCore.tf_chain`` are ``(parent, child, R, t)`` with
  ``p_parent = R @ p_child + t``; ``R`` is converted to a unit quaternion
  ``(x, y, z, w)`` by :func:`quat_from_rot`. There is no board frame.
* The board pose and the press verdicts are not published or logged. rclpy
  mirrors EVERY logger call
  onto ``/rosout``, which is a plain ROS topic any member node can
  subscribe to, so ``/rosout`` is treated as a member-facing channel: at
  INFO/WARN no log line carries the board pose, a key name, the typed
  string or a press rejection reason. Per-press verdicts go to DEBUG and
  only when ``debug_press_feedback`` is on -- the same switch that gates
  ``/sim/press_feedback``. The episode banner keeps the seed and the launch
  key, both of which are public by design (and the seed no longer determines
  the pose -- see ``core/board.pose_rng``). The dashboard (an HTTP server in
  this process, never a ROS topic) is the only place they are shown.
* ``/camera/image_raw`` is packed by hand (no ``cv_bridge``): ``bgr8``,
  ``step = width * 3``, ``data = frame.tobytes()``.
"""

from __future__ import annotations

import math
import os
import queue
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import rclpy.logging
import rclpy.utilities
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import InvalidHandle
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

try:  # RCLError has no public alias; a build without it just loses the guard
    from rclpy._rclpy_pybind11 import RCLError
except ImportError:  # pragma: no cover - defensive
    class RCLError(Exception):  # type: ignore[no-redef]
        """Placeholder so ``except`` clauses below stay valid."""


from autotype_msgs.msg import EpisodeResult, JointVelocityCommand, PressFeedback
from autotype_sim.core.config import DEG, JOINT_NAMES, SimConfig
from autotype_sim.dashboard import DashboardServer, encode_jpeg
from autotype_sim.sim import (
    FRAME_CAMERA_OPTICAL,
    SimCore,
    build_default,
    default_texture_photo,
    resolve_private_dir,
)

PACKAGE_NAME = "autotype_sim"
NODE_NAME = "autotype_sim"

# Timer periods (the node design "Timers"). Reachability (5 Hz) is cached inside
# SimCore.snapshot (REACHABLE_PERIOD) and therefore needs no timer of its own.
TICK_PERIOD_S = 1.0 / 50.0
RENDER_PERIOD_S = 1.0 / 15.0
STATE_PERIOD_S = 1.0 / 20.0
JPEG_PERIOD_S = 1.0 / 10.0

JPEG_SIZE = (640, 360)  # (w, h)
JPEG_QUALITY = 70
TELEOP_DRAIN_MAX = 64  # commands applied per 50 Hz tick at most
WARN_THROTTLE_S = 1.0

# QoS profiles (the node design "Topics, QoS and framing").
QOS_RELIABLE_10 = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
QOS_RELIABLE_1 = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
QOS_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
QOS_IMAGE = qos_profile_sensor_data


# --------------------------------------------------------------------------- #
# Small pure helpers (no ROS)
# --------------------------------------------------------------------------- #


def quat_from_rot(R: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion ``(x, y, z, w)`` of a proper rotation matrix ``R`` (3x3).

    Shepperd's method: pick the largest of ``w, x, y, z`` from the trace and
    diagonal so no division is ill-conditioned. The result satisfies
    ``R @ v == q * v * q^-1`` with the ROS/tf2 ``(x, y, z, w)`` ordering and
    is normalised; the sign is whichever branch was chosen (``q`` and
    ``-q`` are the same rotation).
    """
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0  # 4w
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0  # 4x
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0  # 4y
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0  # 4z
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def _package_dir() -> Path:
    """Directory of the ``autotype_sim`` Python package (``nodes/`` parent)."""
    return Path(__file__).resolve().parents[1]


def find_share_dir(subdir: str) -> Path | None:
    """``share/autotype_sim/<subdir>`` via ament, else the source tree copy.

    Falls back to ``<package>/<subdir>`` so the node also runs from the
    repository without a ``colcon`` install (``PYTHONPATH=src/autotype_sim``).
    Returns ``None`` when neither directory exists.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory(PACKAGE_NAME)) / subdir
        if share.is_dir():
            return share
    except Exception:  # package not installed / ament index unavailable
        pass
    source = _package_dir() / subdir
    if source.is_dir():
        return source
    return None


#: A launch key that survives the parameter round trip as an integer, i.e. all
#: digits and the right length. Anything else that arrives as a number was
#: mangled by the YAML parser (see :func:`_launch_key_text`).
_PLAIN_KEY_RE = re.compile(r"[0-9]{3,6}")


def _yaml_trap_message(value: Any) -> str:
    return (
        f"launch key arrived as a {type(value).__name__} ({value!r}), not text: the parameter "
        "YAML parser read the key you typed as a number (this happens to keys spelled like "
        "'1E5' or '0X1F'), so the characters you meant are no longer recoverable. Choose a "
        "different launch key."
    )


def _launch_key_text(value: Any) -> str:
    """The ``launch_key`` parameter as the text the operator typed.

    ``launch_key`` is declared with ``dynamic_typing`` because a key of pure
    digits (``launch_key:=12345``) legitimately arrives as an int. But the
    parameter YAML parser inside rcl is more permissive than the one launch
    writes with: a perfectly legal A-Z/0-9 key such as ``1E5`` or ``0X1F`` is
    written unquoted by launch and then read back by rcl as a double or an int,
    and the original spelling is gone. ``str()``-ing that silently scores the
    episode against ``100000.0``. Name the trap instead of guessing.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):  # before int: bools are ints in Python
        raise ValueError(
            f"launch key arrived as a boolean ({value!r}): the parameter YAML parser read it "
            "as a YAML bool. Choose a different key."
        )
    if isinstance(value, int):
        text = str(value)
        if _PLAIN_KEY_RE.fullmatch(text):
            return text  # the documented all-digits case: launch_key:=12345
        raise ValueError(_yaml_trap_message(value))  # e.g. 0X1F read back as the int 31
    raise ValueError(_yaml_trap_message(value))


def result_table(res: Any, seed: int) -> str:
    """Multi-line, log-friendly rendering of an ``EpisodeResult``."""
    rej = ", ".join(f"{k}={v}" for k, v in sorted(res.rejections.items())) or "-"
    rows = [
        ("seed", str(seed)),
        ("target", res.target),
        ("typed", repr(res.typed)),
        ("exact_match", str(bool(res.exact_match))),
        ("edit_distance", str(int(res.edit_distance))),
        ("presses_attempted", str(int(res.presses_attempted))),
        ("presses_accepted", str(int(res.presses_accepted))),
        ("rejections", rej),
        ("elapsed_s", f"{float(res.elapsed):.2f}"),
    ]
    width = max(len(k) for k, _ in rows)
    line = "+" + "-" * (width + 2) + "+" + "-" * 34 + "+"
    body = "\n".join(f"| {k.ljust(width)} | {v.ljust(32)} |" for k, v in rows)
    return f"episode result\n{line}\n{body}\n{line}"


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


class SimNode(Node):
    """One node: physics, rendering, press evaluation and the dashboard.

    All callbacks run on the single-threaded executor,
    so ``SimCore`` is never touched concurrently; the dashboard thread only
    ever sees dicts/bytes handed over through thread-safe slots and the
    teleop ``queue.Queue`` drained in :meth:`_on_tick`.
    """

    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        log = self.get_logger()

        # -- parameters -------------------------------------------------- #
        self.declare_parameter("seed", 1)
        self.declare_parameter(
            "launch_key",
            "ROVER",
            ParameterDescriptor(
                dynamic_typing=True,  # `launch_key:=12345` arrives as an int; str() it below
                description="3-6 characters A-Z/0-9 the member must type",
            ),
        )

        self.declare_parameter("teleop", False)
        self.declare_parameter("debug_press_feedback", False)
        self.declare_parameter("dashboard_port", 8080)
        self.declare_parameter("private_dir", "")
        self.declare_parameter("texture_photo", "")

        seed = int(self.get_parameter("seed").value)
        launch_key = _launch_key_text(self.get_parameter("launch_key").value)
        self._teleop = bool(self.get_parameter("teleop").value)
        self._debug_feedback = bool(self.get_parameter("debug_press_feedback").value)
        dashboard_port = int(self.get_parameter("dashboard_port").value)
        private_dir = resolve_private_dir(str(self.get_parameter("private_dir").value))
        texture_param = str(self.get_parameter("texture_photo").value)
        texture_photo = Path(texture_param) if texture_param else default_texture_photo()

        # -- core -------------------------------------------------------- #
        self._cfg = SimConfig()
        t0 = self._now()
        # Raises FileNotFoundError (private dir), ValueError (launch key / seed):
        # main() reports it and exits non-zero.
        self._sim: SimCore = build_default(
            self._cfg, private_dir, launch_key, seed, texture_photo, t0=t0
        )
        cam = self._cfg.camera
        self._frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        self._frame_ready = False
        self._press_count = 0

        # -- publishers -------------------------------------------------- #
        self._pub_joints = self.create_publisher(JointState, "/joint_states", QOS_RELIABLE_10)
        self._pub_image = self.create_publisher(Image, "/camera/image_raw", QOS_IMAGE)
        self._pub_caminfo = self.create_publisher(CameraInfo, "/camera/camera_info", QOS_LATCHED)
        self._pub_launch_key = self.create_publisher(String, "/sim/launch_key", QOS_LATCHED)
        self._pub_result = self.create_publisher(EpisodeResult, "/sim/result", QOS_LATCHED)
        self._pub_feedback = (
            self.create_publisher(PressFeedback, "/sim/press_feedback", QOS_RELIABLE_10)
            if self._debug_feedback
            else None
        )
        self._tf = TransformBroadcaster(self)
        self._tf_static = StaticTransformBroadcaster(self)

        # -- subscribers / service --------------------------------------- #
        self.create_subscription(
            JointVelocityCommand, "/arm/cmd_joint_velocity", self._on_cmd, QOS_RELIABLE_1
        )
        self.create_subscription(Empty, "/arm/press", self._on_press_msg, QOS_RELIABLE_10)
        self.create_subscription(Empty, "/sim/done", self._on_done_msg, QOS_RELIABLE_10)
        self.create_service(Trigger, "/sim/reset", self._on_reset_srv)

        # -- dashboard --------------------------------------------------- #
        self._commands: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._dash: DashboardServer | None = None
        if dashboard_port > 0:
            self._dash = self._start_dashboard(dashboard_port)

        # -- one-shot publications --------------------------------------- #
        self._publish_camera_info()
        self._publish_static_tf()
        self._announce_episode()

        # -- timers ------------------------------------------------------ #
        self.create_timer(TICK_PERIOD_S, self._on_tick)
        self.create_timer(RENDER_PERIOD_S, self._on_render)
        self.create_timer(STATE_PERIOD_S, self._on_state)
        self.create_timer(JPEG_PERIOD_S, self._on_jpeg)

        warning = transport_warning()
        if warning is not None:
            log.warning(warning)
        log.info(
            f"sim_node ready: teleop={self._teleop} debug_press_feedback={self._debug_feedback} "
            f"dashboard={'off' if self._dash is None else self._dash.url}"
        )

    # ------------------------------------------------------------------ time

    def _now(self) -> float:
        """Node clock in float seconds (the ``t`` every SimCore call takes)."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- dashboard

    def _start_dashboard(self, port: int) -> DashboardServer | None:
        """Build and start the DashboardServer; ``None`` (and an error log) on failure."""
        log = self.get_logger()
        web_dir = find_share_dir("web")
        if web_dir is None or not (web_dir / "index.html").is_file():
            log.error("dashboard disabled: web/ directory (index.html) not found")
            return None
        ok, png = cv2.imencode(".png", self._sim.texture)
        if not ok:
            log.error("dashboard disabled: texture PNG encoding failed")
            return None
        server = DashboardServer(
            port=port,
            web_dir=web_dir,
            static_info=self._sim.static_info(teleop=self._teleop),
            texture_png=png.tobytes(),
            teleop=self._teleop,
            command_queue=self._commands,
            logger=log,
        )
        try:
            server.start()
        except (OSError, RuntimeError) as exc:
            log.error(f"dashboard failed to start on port {port} ({exc}); running without it")
            return None
        return server

    def _drain_teleop(self, t: float) -> None:
        """Apply queued dashboard Commands through the same paths as the topics."""
        for _ in range(TELEOP_DRAIN_MAX):
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            kind = cmd.get("type")
            # parse_command already bounds every value, but a teleop client is
            # untrusted input on an open port: one bad command must never
            # escape a timer callback and take the whole node down.
            try:
                if kind == "jog":
                    self._sim.command([cmd["joint"]], [float(cmd["velocity_deg_s"]) * DEG], t)
                elif kind == "stop":
                    self._sim.command(list(JOINT_NAMES), [0.0] * len(JOINT_NAMES), t)
                elif kind == "press":
                    self._do_press(t)
                elif kind == "done":
                    self._do_done(t)
                elif kind == "reset":
                    self._do_reset(cmd.get("seed"))
            except (ValueError, TypeError, OverflowError, KeyError) as exc:
                self.get_logger().warning(
                    f"dashboard command {kind!r} dropped: {exc!r}",
                    throttle_duration_sec=WARN_THROTTLE_S,
                )

    # ---------------------------------------------------------------- timers

    def _on_tick(self) -> None:
        """50 Hz: teleop queue -> command; step; /joint_states; dynamic TF."""
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if self._teleop:
            self._drain_teleop(t)
        self._sim.step(t)
        stamp = now.to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name = list(JOINT_NAMES)
        js.position = [float(v) for v in self._sim.q]
        js.velocity = [float(v) for v in self._sim.qd_measured]
        if not rclpy.ok():
            return  # shutdown raced this timer; publishing now raises RCLError
        self._pub_joints.publish(js)

        self._tf.sendTransform(
            [self._transform_msg(stamp, *entry) for entry in self._sim.tf_chain(static=False)]
        )

    def _on_render(self) -> None:
        """15 Hz: render the camera image into the reusable buffer and publish it."""
        # Stamp BEFORE rendering: the pixels show the arm pose as of now, and
        # rendering costs ~8-10 ms. Stamping afterwards would date the frame
        # later than its own content, which is the sensor latency
        # INTERFACES.md s.13 promises is not modelled.
        stamp = self.get_clock().now().to_msg()
        self._sim.render(out=self._frame)
        self._frame_ready = True
        cam = self._cfg.camera
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_CAMERA_OPTICAL
        msg.height = int(cam.height)
        msg.width = int(cam.width)
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(cam.width) * 3
        msg.data = self._frame.tobytes()
        self._pub_image.publish(msg)

    def _on_state(self) -> None:
        """20 Hz: State dict to the dashboard (reachability cached inside at 5 Hz)."""
        if self._dash is None:
            return
        self._dash.publish_state(self._sim.snapshot(self._now()))

    def _on_jpeg(self) -> None:
        """10 Hz: JPEG of the last rendered frame, only when someone is watching."""
        if self._dash is None or not self._frame_ready or self._dash.client_count == 0:
            return
        self._dash.publish_frame(encode_jpeg(self._frame, JPEG_SIZE, JPEG_QUALITY))

    # ------------------------------------------------------------ callbacks

    def _on_cmd(self, msg: JointVelocityCommand) -> None:
        """``/arm/cmd_joint_velocity``: by-name latch; bad messages dropped."""
        log = self.get_logger()
        names = [str(n) for n in msg.name]
        velocities = [float(v) for v in msg.velocity]
        if len(names) != len(velocities):
            log.warning(
                f"JointVelocityCommand dropped: {len(names)} names vs {len(velocities)} velocities",
                throttle_duration_sec=WARN_THROTTLE_S,
            )
            return
        try:
            unknown = self._sim.command(names, velocities, self._now())
        except ValueError as exc:
            log.warning(f"JointVelocityCommand dropped: {exc}", throttle_duration_sec=WARN_THROTTLE_S)
            return
        if unknown:
            log.warning(
                f"JointVelocityCommand: ignoring unknown joint name(s) {unknown}",
                throttle_duration_sec=WARN_THROTTLE_S,
            )

    def _on_press_msg(self, _msg: Empty) -> None:
        self._do_press(self._now())

    def _on_done_msg(self, _msg: Empty) -> None:
        self._do_done(self._now())

    def _on_reset_srv(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        seed = self._do_reset(None)
        response.success = True
        response.message = f"episode {self._sim.episode_index} started with seed {seed}"
        return response

    # ------------------------------------------------------- episode actions

    def _do_press(self, t: float) -> None:
        """Evaluate one press, log it (as the dashboard event does), maybe publish feedback."""
        res = self._sim.press(t)
        self._press_count += 1
        reason = str(res.reason)
        key = res.key.name if res.key is not None else ""
        # /rosout is a member-readable topic: a bare count only. The verdict
        # (key name / rejection reason) is DEBUG *and* gated on the same flag
        # as /sim/press_feedback.
        self.get_logger().info(f"press #{self._press_count} received")
        if self._debug_feedback:
            verdict = f"accepted key {key}" if res.accepted else f"rejected ({reason})"
            self.get_logger().debug(f"press #{self._press_count}: {verdict}")
        if self._pub_feedback is None:
            return
        fb = PressFeedback()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.accepted = bool(res.accepted)
        fb.reason = reason
        fb.key = key
        fb.board_x = float(res.board_xy[0]) if res.board_xy is not None else 0.0
        fb.board_y = float(res.board_xy[1]) if res.board_xy is not None else 0.0
        fb.range = float(res.range) if res.range is not None else 0.0
        fb.incidence = float(res.incidence) if res.incidence is not None else 0.0
        self._pub_feedback.publish(fb)

    def _do_done(self, t: float) -> None:
        """Finish the episode: score, publish ``/sim/result``, log the table."""
        log = self.get_logger()
        try:
            res = self._sim.done(t)
        except RuntimeError:
            log.warning("/sim/done ignored: episode already finished (call /sim/reset)")
            return
        except ValueError as exc:
            log.warning(f"/sim/done ignored: {exc}")
            return
        msg = EpisodeResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.seed = int(self._sim.seed)
        msg.target = str(res.target)
        msg.typed = str(res.typed)
        msg.exact_match = bool(res.exact_match)
        msg.edit_distance = int(res.edit_distance)
        msg.presses_attempted = int(res.presses_attempted)
        msg.presses_accepted = int(res.presses_accepted)
        reasons = sorted(res.rejections.items())
        msg.rejection_reasons = [str(k) for k, _ in reasons]
        msg.rejection_counts = [int(v) for _, v in reasons]
        msg.elapsed = float(res.elapsed)
        self._pub_result.publish(msg)
        # The table names the target and the typed string; keep it off /rosout
        # at default verbosity (the operator has the dashboard and /sim/result).
        log.info(
            f"episode finished: {int(res.presses_attempted)} presses, "
            f"{int(res.presses_accepted)} accepted, {float(res.elapsed):.2f} s"
        )
        log.debug(result_table(res, self._sim.seed))

    def _do_reset(self, seed: int | None) -> int:
        """Start the next episode (``seed`` or previous + 1); republish the launch key.

        ``/sim/result`` is TRANSIENT_LOCAL depth 1, so the finished episode's
        result stays latched in the middleware until the publisher that wrote
        it goes away. INTERFACES.md s.9.10 / s.11 promise the result is
        *cleared* on reset, so the publisher is recreated here: a node that
        subscribes after a reset then sees nothing until the new episode ends,
        instead of a result for an episode that is over.
        """
        self._recreate_result_publisher()
        used = self._sim.reset(seed, t0=self._now())
        self._press_count = 0
        self._announce_episode()
        return used

    def _recreate_result_publisher(self) -> None:
        """Drop the latched ``/sim/result`` sample by replacing its publisher."""
        old = self._pub_result
        self._pub_result = self.create_publisher(EpisodeResult, "/sim/result", QOS_LATCHED)
        try:
            self.destroy_publisher(old)
        except Exception as exc:  # never let a reset fail on teardown
            self.get_logger().warning(f"/sim/result publisher not replaced cleanly: {exc!r}")

    def _announce_episode(self) -> None:
        """Latched launch key + the episode banner (the node design lifecycle)."""
        msg = String()
        msg.data = self._sim.launch_key
        self._pub_launch_key.publish(msg)
        self.get_logger().info(
            f"episode {self._sim.episode_index} seed {self._sim.seed} "
            f"launch_key {self._sim.launch_key}"
        )

    # ---------------------------------------------------------- one-shots

    def _publish_camera_info(self) -> None:
        info = self._sim.camera_info()
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(info["frame_id"])
        msg.width = int(info["width"])
        msg.height = int(info["height"])
        msg.distortion_model = str(info["distortion_model"])
        msg.d = [float(v) for v in info["D"]]
        msg.k = [float(v) for v in info["K"]]
        msg.r = [float(v) for v in info["R"]]
        msg.p = [float(v) for v in info["P"]]
        self._pub_caminfo.publish(msg)

    def _publish_static_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._tf_static.sendTransform(
            [self._transform_msg(stamp, *entry) for entry in self._sim.tf_chain(static=True)]
        )

    @staticmethod
    def _transform_msg(
        stamp: Any, parent: str, child: str, R: np.ndarray, t: np.ndarray
    ) -> TransformStamped:
        """``TransformStamped`` for ``p_parent = R @ p_child + t``."""
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])
        qx, qy, qz, qw = quat_from_rot(R)
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        return msg

    # ------------------------------------------------------------ shutdown

    def destroy_node(self) -> bool:
        """Stop the dashboard thread before tearing the node down."""
        if self._dash is not None:
            try:
                self._dash.stop()
            except Exception as exc:  # never let shutdown raise
                self.get_logger().error(f"dashboard stop failed: {exc!r}")
            self._dash = None
        return super().destroy_node()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


# Fast DDS (the Jazzy default RMW) gives each participant a 512 KiB shared-memory
# segment. One 1280x720 bgr8 frame (2.76 MB) is split into RTPS fragments that all
# sit in that segment until the reader drains them, and with best-effort QoS an
# overflow silently drops the whole frame: measured 10 Hz delivered of 15 in a
# 4-vCPU VM and 6 Hz in a container. An 8 MB segment holds a full frame; readers
# keep their defaults because the fragment size (max_msg_size) is not touched.
FASTDDS_TRANSPORTS_ENV = "FASTDDS_BUILTIN_TRANSPORTS"
FASTDDS_SEGMENT_OPTION = "sockets_size=8MB"


def fastdds_transports_value(current: str | None) -> str:
    """``FASTDDS_BUILTIN_TRANSPORTS`` to run with, given the operator's value.

    Unset -> ``DEFAULT?sockets_size=8MB``. A value without a ``sockets_size``
    option (``SHM`` as in the Docker image, ``LARGE_DATA?max_msg_size=1MB``, ...)
    keeps its transport mode and gains the 8 MB segment. A value that already
    sizes its buffers is returned unchanged. Other RMWs ignore the variable.
    """
    if current is None or not current.strip():
        return f"DEFAULT?{FASTDDS_SEGMENT_OPTION}"
    if "sockets_size=" in current:
        return current
    return f"{current}{'&' if '?' in current else '?'}{FASTDDS_SEGMENT_OPTION}"


def transport_warning() -> str | None:
    """Why ``/camera/image_raw`` will drop frames under this configuration, or ``None``.

    rmw_fastrtps implements ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`` (and the
    older ``ROS_LOCALHOST_ONLY=1``) by disabling the builtin transports and pushing
    its own, unsizable ``SharedMemTransportDescriptor`` -- the variable above has no
    effect there. The Docker image isolates with ``FASTDDS_BUILTIN_TRANSPORTS=SHM``
    instead, which is why it does not set that range.
    """
    if "fastrtps" not in rclpy.utilities.get_rmw_implementation_identifier():
        return None
    rng = os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "").strip().upper()
    if rng == "LOCALHOST" or os.environ.get("ROS_LOCALHOST_ONLY", "").strip() == "1":
        return (
            "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST / ROS_LOCALHOST_ONLY: rmw_fastrtps replaces "
            "the builtin transports with a fixed 512 KiB shared-memory segment, so most 2.7 MB "
            "/camera/image_raw frames will be dropped (measured 6 Hz of 15). Use the default "
            "range with FASTDDS_BUILTIN_TRANSPORTS=SHM for host-local isolation instead."
        )
    return None


def configure_process_defaults() -> None:
    """Process-wide knobs that must be set before ``rclpy.init()`` / the first render.

    OpenCV's worker pool (``nproc - 1`` threads) makes the 15 Hz render ~2x faster
    but its workers spin-wait after every parallel region: measured 3 threads at
    ~10 % CPU each in a 4-vCPU VM, more than the render itself. Single-threaded,
    a frame still takes < 5 ms and the process drops from ~60 % to ~35 % of one
    core, leaving the rest to the members' nodes. ``OPENCV_FOR_THREADS_NUM``, when
    the operator sets it, wins.
    """
    os.environ[FASTDDS_TRANSPORTS_ENV] = fastdds_transports_value(os.environ.get(FASTDDS_TRANSPORTS_ENV))
    if "OPENCV_FOR_THREADS_NUM" not in os.environ:
        cv2.setNumThreads(1)


def main(args: list[str] | None = None) -> int:
    """``ros2 run autotype_sim sim_node``: init, spin, graceful shutdown."""
    configure_process_defaults()
    rclpy.init(args=args)
    node: SimNode | None = None
    code = 0
    try:
        try:
            node = SimNode()
        except (FileNotFoundError, ValueError, TypeError) as exc:
            rclpy.logging.get_logger(NODE_NAME).error(f"sim_node failed to start: {exc}")
            return 1
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (RCLError, InvalidHandle):
        # Ctrl-C / SIGTERM invalidates the context asynchronously, so spin can
        # lose the race while building its next wait set (or a timer can lose
        # it mid-publish). That is a normal shutdown, not a crash: no
        # traceback, exit 0.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
