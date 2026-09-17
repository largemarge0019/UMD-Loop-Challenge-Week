#!/usr/bin/env bash
# VM integration checklist for the sim node.
#
# Run INSIDE the VM / container, in a shell where `ros2` works, against a node
# that is ALREADY running (item 1 of the checklist is the operator's job):
#
#   source /opt/ros/jazzy/setup.bash && source install/setup.bash
#   ros2 launch autotype_sim challenge.launch.py seed:=3 launch_key:=MARS teleop:=true \
#        [debug_press_feedback:=true]
#   bash src/autotype_sim/test/integration/vm_checklist.sh
#
# Environment overrides: LAUNCH_KEY (MARS), DASH_PORT or DASHBOARD_PORT (8080), NODE_NAME (/autotype_sim),
# HZ_SECONDS (6). Prints PASS/FAIL/SKIP per item and exits non-zero on any FAIL.
# The script never reads the private geometry and never prints the board pose:
# every geometric assertion uses only the public forward kinematics of INTERFACES.md.

set -u
LAUNCH_KEY="${LAUNCH_KEY:-MARS}"
DASH_PORT="${DASH_PORT:-${DASHBOARD_PORT:-8080}}"
NODE_NAME="${NODE_NAME:-/autotype_sim}"
HZ_SECONDS="${HZ_SECONDS:-6}"
FAILS=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }
skip() { printf 'SKIP  %s\n' "$*"; }
info() { printf 'INFO  %s\n' "$*"; }
section() { printf '\n== %s ==\n' "$*"; }

# ros2 topic hz -> average rate (float) or empty. `--window` keeps it responsive.
topic_hz() {
    timeout "$HZ_SECONDS" ros2 topic hz --window 50 "$1" 2>/dev/null \
        | grep -E 'average rate' | tail -n 1 | sed -E 's/.*average rate: *([0-9.]+).*/\1/'
}

# ros2 topic echo --once into a file, retried: a fresh CLI node on a loaded VM can miss
# its discovery window and leave the file empty, which is not a node bug.
echo_once() { # outfile [ros2 topic echo args...]
    local out="$1"; shift
    local attempt
    for attempt in 1 2 3; do
        timeout 10 ros2 topic echo --once "$@" > "$out" 2>/dev/null
        [ -s "$out" ] && return 0
    done
    return 1
}

in_range() { # value lo hi
    python3 - "$1" "$2" "$3" <<'EOF'
import sys
v, lo, hi = (float(x) for x in sys.argv[1:4])
sys.exit(0 if lo <= v <= hi else 1)
EOF
}

# --------------------------------------------------------------------------- #
section "1. node is running (build + launch are done by hand before this script)"
# The daemon's graph cache can go stale after many node restarts on one VM; fall back
# to direct discovery before declaring the node absent.
if timeout 10 ros2 node list 2>/dev/null | grep -qx "$NODE_NAME" \
   || timeout 25 ros2 node list --no-daemon --spin-time 5 2>/dev/null | grep -qx "$NODE_NAME"; then
    pass "node $NODE_NAME is up"
else
    fail "node $NODE_NAME not found in 'ros2 node list' -- launch it first"
    echo "aborting: nothing else can pass without the node"; exit 1
fi

# --------------------------------------------------------------------------- #
section "2. rates, camera_info, latched launch key"
HZ_JS="$(topic_hz /joint_states)"
if [ -n "$HZ_JS" ] && in_range "$HZ_JS" 45 55; then pass "/joint_states ${HZ_JS} Hz (expect ~50)"; else fail "/joint_states rate '${HZ_JS:-none}' (expect ~50)"; fi
HZ_IMG="$(topic_hz /camera/image_raw)"
if [ -n "$HZ_IMG" ] && in_range "$HZ_IMG" 13 17; then pass "/camera/image_raw ${HZ_IMG} Hz (expect ~15)"; else fail "/camera/image_raw rate '${HZ_IMG:-none}' (expect ~15)"; fi

echo_once "$TMP/caminfo.txt" /camera/camera_info
if grep -qE '^k:' "$TMP/caminfo.txt" && grep -qE '^- 900\.0' "$TMP/caminfo.txt" && grep -q 'plumb_bob' "$TMP/caminfo.txt" \
   && grep -q 'frame_id: camera_optical_frame' "$TMP/caminfo.txt"; then
    pass "/camera/camera_info latched: K (fx=900), plumb_bob, camera_optical_frame"
else
    fail "/camera/camera_info missing or wrong (see $TMP/caminfo.txt)"; cat "$TMP/caminfo.txt" | head -20
fi

# A late subscriber must still get the launch key (transient_local).
if timeout 10 ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable /sim/launch_key 2>/dev/null \
   | grep -q "data: ${LAUNCH_KEY}$"; then
    pass "/sim/launch_key echoes ${LAUNCH_KEY} on a late (transient_local) subscriber"
else
    fail "/sim/launch_key did not echo ${LAUNCH_KEY}"
fi

echo_once "$TMP/img.txt" /camera/image_raw --no-arr
if grep -q "encoding: bgr8" "$TMP/img.txt" && grep -q "step: 3840" "$TMP/img.txt" && grep -q "width: 1280" "$TMP/img.txt" \
   && grep -q "frame_id: camera_optical_frame" "$TMP/img.txt"; then
    pass "/camera/image_raw bgr8 1280x720 step 3840 in camera_optical_frame"
else
    fail "/camera/image_raw header/encoding wrong (see $TMP/img.txt)"
fi

# --------------------------------------------------------------------------- #
section "3. TF: world -> camera_optical_frame resolves; no board frame anywhere; TF == FK"
timeout 6 ros2 run tf2_ros tf2_echo world camera_optical_frame > "$TMP/tf.txt" 2>&1
if grep -q "Translation:" "$TMP/tf.txt"; then
    pass "tf2_echo world camera_optical_frame returns a transform"
else
    fail "tf2_echo world camera_optical_frame returned nothing"; tail -5 "$TMP/tf.txt"
fi

timeout 20 ros2 topic list --no-daemon --spin-time 3 > "$TMP/topics.txt" 2>/dev/null
echo_once "$TMP/tf_static.txt" --qos-durability transient_local --qos-reliability reliable /tf_static
echo_once "$TMP/tf_dyn.txt" /tf
timeout 10 ros2 param list "$NODE_NAME" > "$TMP/params.txt" 2>/dev/null
# A negative check is only evidence when the captures are complete: require the frames
# and topics that MUST be there. "dashboard" (the dashboard_port parameter) is not the board.
if ! grep -qx '/camera/image_raw' "$TMP/topics.txt" || ! grep -q 'child_frame_id: camera_optical_frame' "$TMP/tf_static.txt" \
   || ! grep -q 'child_frame_id: shoulder_link' "$TMP/tf_dyn.txt" || ! grep -qx '  dashboard_port' "$TMP/params.txt"; then
    fail "board-leak check: captures incomplete (topics $(wc -l < "$TMP/topics.txt" | tr -d ' ') lines, tf_static $(wc -c < "$TMP/tf_static.txt" | tr -d ' ') B, tf $(wc -c < "$TMP/tf_dyn.txt" | tr -d ' ') B, params $(wc -l < "$TMP/params.txt" | tr -d ' ') lines) -- stale ros2 daemon? try 'ros2 daemon stop'"
elif ! grep -i board "$TMP/topics.txt" "$TMP/tf_static.txt" "$TMP/tf_dyn.txt" "$TMP/params.txt" | grep -vi dashboard | grep -q . \
   && ! grep -qiE 'kb_|key_area|layout' "$TMP/params.txt"; then
    pass "no board frame / placement on $(wc -l < "$TMP/topics.txt" | tr -d ' ') topics, /tf, /tf_static or $(wc -l < "$TMP/params.txt" | tr -d ' ') parameters"
else
    fail "something board-related leaked:"; grep -ni board "$TMP/topics.txt" "$TMP/tf_static.txt" "$TMP/tf_dyn.txt" "$TMP/params.txt" | grep -vi dashboard; grep -niE 'kb_|key_area|layout' "$TMP/params.txt"
fi

# TF vs public FK (INTERFACES.md s.4.3) while the arm is at rest at home.
echo_once "$TMP/js.txt" /joint_states || info "no /joint_states sample captured after 3 tries"
if python3 - "$TMP/js.txt" "$TMP/tf.txt" <<'EOF'
import math, re, sys
import numpy as np
js = open(sys.argv[1]).read(); tf = open(sys.argv[2]).read()
pos = re.search(r"position:\n((?:- .*\n)+)", js)
tr = re.search(r"Translation: \[([^\]]+)\]", tf); rot = re.search(r"Rotation: in Quaternion(?: \([a-z]+\))? \[([^\]]+)\]", tf)  # Jazzy prints "(xyzw)"
if pos is None or tr is None or rot is None:
    print(f"      cannot compare: joint_states sample {'ok' if pos else 'missing'} ({len(js)} bytes), tf2_echo {'ok' if tr and rot else 'missing'} ({len(tf)} bytes)")
    sys.exit(1)
q = [float(l[2:]) for l in pos.group(1).strip().splitlines()]
t = np.array([float(v) for v in tr.group(1).split(",")]); x, y, z, w = (float(v) for v in rot.group(1).split(","))
L0, L1, L2 = 0.30, 0.60, 0.40
c0, s0 = math.cos(q[0]), math.sin(q[0]); th1 = q[1]; phi = q[1] + q[2]
p_sh = np.array([0, 0, L0]); p_el = p_sh + L1 * np.array([math.cos(th1) * c0, math.cos(th1) * s0, math.sin(th1)])
p_head = p_el + L2 * np.array([math.cos(phi) * c0, math.cos(phi) * s0, math.sin(phi)])
X = np.array([math.cos(phi) * c0, math.cos(phi) * s0, math.sin(phi)]); Y = np.array([-s0, c0, 0.0]); Z = np.array([-math.sin(phi) * c0, -math.sin(phi) * s0, math.cos(phi)])
R_cam = np.column_stack([-Y, -Z, X])
R_tf = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
et, eR = float(np.abs(t - p_head).max()), float(np.abs(R_tf - R_cam).max())
print(f"      |t_tf - p_head| = {et:.2e}, |R_tf - R_cam| = {eR:.2e} (q = {[round(v,4) for v in q]})")
# tf2_echo prints 3 decimals (+-5e-4 per quaternion component, up to ~2e-3 in R) and the
# two samples are seconds apart while the plant idles with ~3e-3 rad of jitter; a wrong
# frame convention would be O(1), so 5e-3 still discriminates.
sys.exit(0 if et < 5e-3 and eR < 5e-3 else 1)
EOF
then pass "TF world->camera_optical_frame matches the public forward kinematics"; else fail "TF world->camera_optical_frame disagrees with the forward kinematics (arm moving? see above)"; fi

# --------------------------------------------------------------------------- #
section "4. command path + watchdog (head_pan 0.3 rad/s for 1 s, then silence)"
if python3 - <<'EOF'
import math, time, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from autotype_msgs.msg import JointVelocityCommand

class Probe(Node):
    def __init__(self):
        super().__init__("vm_checklist_probe")
        self.js = None
        self.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.pub = self.create_publisher(JointVelocityCommand, "/arm/cmd_joint_velocity",
                                         QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1))
    def _cb(self, m):
        self.js = m
    def state(self):
        i = list(self.js.name).index("head_pan")
        return float(self.js.position[i]), float(self.js.velocity[i])
    def spin_for(self, s):
        end = time.monotonic() + s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

rclpy.init()
p = Probe()
deadline = time.monotonic() + 10.0  # a fresh CLI participant can take a few seconds to discover the node
while p.js is None and time.monotonic() < deadline:
    rclpy.spin_once(p, timeout_sec=0.05)
if p.js is None:
    print("      no /joint_states received within 10 s"); sys.exit(1)
p.spin_for(0.5)
q0, _ = p.state()
msg = JointVelocityCommand(); msg.name = ["head_pan"]; msg.velocity = [0.3]
t_end = time.monotonic() + 1.0
while time.monotonic() < t_end:
    msg.header.stamp = p.get_clock().now().to_msg()
    p.pub.publish(msg)
    p.spin_for(0.05)  # 20 Hz
q1, v1 = p.state()
dq = q1 - q0
print(f"      head_pan moved {dq:+.3f} rad in 1 s (velocity now {v1:+.3f} rad/s)")
ok_move = 0.20 <= dq <= 0.36  # 0.3 rad/s minus the a_max ramp, +/- tracking noise
p.spin_for(0.6)  # watchdog 0.10 s + decel 0.10 s + margin, no commands
q2, v2 = p.state()
print(f"      after 0.6 s of silence: velocity {v2:+.4f} rad/s, drift {q2 - q1:+.4f} rad")
ok_stop = abs(v2) < 0.02 and abs(q2 - q1) < 0.06
p.destroy_node(); rclpy.shutdown()
sys.exit(0 if ok_move and ok_stop else 1)
EOF
then pass "head_pan tracked 0.3 rad/s and the watchdog zeroed it after silence"; else fail "command path / watchdog (see numbers above)"; fi

# --------------------------------------------------------------------------- #
section "5. /arm/press -> dashboard event (and PressFeedback when debug_press_feedback)"
# Watch the WS for the event count to grow; drive the press from ros2 topic pub.
if python3 - "$DASH_PORT" <<'EOF'
import asyncio, json, subprocess, sys
import aiohttp
port = sys.argv[1]
async def main():
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"http://localhost:{port}/ws") as ws:
            async def next_state(timeout=3.0):
                while True:
                    m = await asyncio.wait_for(ws.receive(), timeout)
                    if m.type == aiohttp.WSMsgType.TEXT:
                        return json.loads(m.data)
                    if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise RuntimeError("ws closed")
            before = await next_state()
            n0 = sum(1 for e in before["events"] if e["kind"] == "press")
            subprocess.run(["ros2", "topic", "pub", "--once", "/arm/press", "std_msgs/msg/Empty", "{}"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            # The blocking `ros2 topic pub` above froze this loop for a few seconds while the
            # node kept sending 20 Hz States, so the socket holds a backlog of pre-press States:
            # drain by wall-clock, not by a fixed count, or the press is missed.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 10.0
            while loop.time() < deadline:
                st = await next_state()
                presses = [e for e in st["events"] if e["kind"] == "press"]
                if len(presses) > n0:
                    print(f"      event: {presses[-1]['reason']} accepted={presses[-1]['accepted']} key={presses[-1]['key']}")
                    return 0
            print("      no press event appeared in State.events within 10 s"); return 1
sys.exit(asyncio.run(main()))
EOF
then pass "/arm/press produced a dashboard press event"; else fail "/arm/press: no dashboard event (dashboard on port $DASH_PORT?)"; fi

if ros2 topic info /sim/press_feedback 2>/dev/null | grep -qE 'Publisher count: [1-9]'; then
    ( timeout 15 ros2 topic echo --once /sim/press_feedback > "$TMP/fb.txt" 2>/dev/null ) &
    ECHO_PID=$!
    sleep 1.5
    timeout 20 ros2 topic pub --once /arm/press std_msgs/msg/Empty "{}" > /dev/null 2>&1
    wait $ECHO_PID
    if grep -qE '^reason: ' "$TMP/fb.txt"; then
        pass "PressFeedback published: $(grep -E '^reason: ' "$TMP/fb.txt")"
    else
        fail "PressFeedback not received although /sim/press_feedback has a publisher"
    fi
else
    skip "PressFeedback: node not launched with debug_press_feedback:=true (no publisher on /sim/press_feedback)"
fi

# --------------------------------------------------------------------------- #
section "6. /sim/done -> /sim/result with target ${LAUNCH_KEY}"
timeout 20 ros2 topic pub --once /sim/done std_msgs/msg/Empty "{}" > /dev/null 2>&1
sleep 1
echo_once "$TMP/result.txt" --qos-durability transient_local --qos-reliability reliable /sim/result
if grep -q "target: ${LAUNCH_KEY}$" "$TMP/result.txt" && grep -qE '^seed: ' "$TMP/result.txt" && grep -qE '^elapsed: ' "$TMP/result.txt"; then
    pass "/sim/result latched with target ${LAUNCH_KEY} ($(grep -E '^(typed|presses_attempted|presses_accepted):' "$TMP/result.txt" | tr '\n' ' '))"
else
    fail "/sim/result missing or wrong (see $TMP/result.txt)"; head -20 "$TMP/result.txt"
fi
SEED_DONE="$(grep -E '^seed: ' "$TMP/result.txt" | awk '{print $2}')"

# --------------------------------------------------------------------------- #
section "7. /sim/reset -> next seed, result cleared"
timeout 20 ros2 service call /sim/reset std_srvs/srv/Trigger > "$TMP/reset.txt" 2>&1
NEW_SEED="$(grep -oE 'seed [0-9]+' "$TMP/reset.txt" | head -n1 | awk '{print $2}')"
if grep -q "success=True" "$TMP/reset.txt" && [ -n "$NEW_SEED" ]; then
    if [ -n "$SEED_DONE" ] && [ "$NEW_SEED" -ne $((SEED_DONE + 1)) ]; then
        fail "/sim/reset named seed $NEW_SEED, expected $((SEED_DONE + 1))"
    else
        pass "/sim/reset succeeded, response names seed $NEW_SEED"
    fi
else
    fail "/sim/reset failed: $(cat "$TMP/reset.txt")"
fi
if python3 - "$DASH_PORT" "${NEW_SEED:-0}" <<'EOF'
import asyncio, json, sys
import aiohttp
port, seed = sys.argv[1], int(sys.argv[2])
async def main():
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"http://localhost:{port}/ws") as ws:
            for _ in range(40):
                m = await asyncio.wait_for(ws.receive(), 3.0)
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                st = json.loads(m.data)
                ok = st["episode"]["status"] == "running" and st["result"] is None and st["episode"]["seed"] == seed and st["typed"] == ""
                print(f"      State: status={st['episode']['status']} seed={st['episode']['seed']} result={st['result']} typed={st['typed']!r}")
                return 0 if ok else 1
    return 1
sys.exit(asyncio.run(main()))
EOF
then pass "dashboard State: running, seed $NEW_SEED, result cleared"; else fail "dashboard State after reset is not clean"; fi
if timeout 10 ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable /sim/launch_key 2>/dev/null | grep -q "data: ${LAUNCH_KEY}$"; then
    pass "launch key republished after reset"
else
    fail "launch key not available after reset"
fi
# INTERFACES.md s.9.10 / s.11: the topic is cleared too, not just the dashboard.
# A brand-new subscriber (transient_local, like a member node restarting between
# episodes) must receive NOTHING until the new episode ends.
if timeout 8 ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable /sim/result > "$TMP/stale.txt" 2>/dev/null && [ -s "$TMP/stale.txt" ]; then
    fail "/sim/result still latched after reset: a late subscriber gets the finished episode ($(grep -E '^seed:' "$TMP/stale.txt" | tr -d '\n'))"
else
    pass "/sim/result cleared after reset (a fresh late subscriber receives nothing)"
fi

# --------------------------------------------------------------------------- #
section "7b. /rosout carries no board pose or press verdicts at default verbosity"
# rclpy mirrors every logger call onto /rosout, which is a plain ROS topic. A
# member node must not be able to read the per-press verdict off it.
( timeout 12 ros2 topic echo /rosout > "$TMP/rosout.txt" 2>/dev/null ) &
ROSOUT_PID=$!
sleep 2
for _ in 1 2 3; do timeout 10 ros2 topic pub --once /arm/press std_msgs/msg/Empty "{}" > /dev/null 2>&1; sleep 0.5; done
wait $ROSOUT_PID 2>/dev/null || true
if grep -qE "press #[0-9]+ received" "$TMP/rosout.txt"; then
    pass "/rosout shows the bare press count"
else
    skip "no press line seen on /rosout in the sample window"
fi
LEAK="$(grep -oiE 'accepted key [A-Z0-9_]+|rejected \(|NO_INTERSECT|OUT_OF_REACH|TOO_CLOSE|GLANCING|NO_KEY|DEBOUNCE' "$TMP/rosout.txt" | sort -u | tr '\n' ' ')"
if [ -z "$LEAK" ]; then
    pass "/rosout carries no key name and no rejection reason"
else
    fail "/rosout leaked press detail: $LEAK"
fi

# --------------------------------------------------------------------------- #
section "8. dashboard HTTP"
N_KEYS="$(curl -sf "http://localhost:${DASH_PORT}/api/static" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["keys"]))' 2>/dev/null)"
if [ "${N_KEYS:-0}" = "87" ]; then pass "/api/static has 87 keys"; else fail "/api/static keys = '${N_KEYS:-none}' (expect 87)"; fi
if curl -sf "http://localhost:${DASH_PORT}/" | grep -qi '<html'; then pass "GET / serves index.html"; else fail "GET / did not serve index.html"; fi
if curl -sf "http://localhost:${DASH_PORT}/static/texture.png" | head -c 8 | grep -q 'PNG'; then pass "/static/texture.png is a PNG"; else fail "/static/texture.png missing"; fi
info "manual: open http://localhost:${DASH_PORT} (port-forwarded) and confirm the page is live"

# --------------------------------------------------------------------------- #
printf '\n'
if [ "$FAILS" -eq 0 ]; then
    echo "ALL PASS"
    exit 0
fi
echo "$FAILS FAILED"
exit 1
