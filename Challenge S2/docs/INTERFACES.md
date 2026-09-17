# INTERFACES — the simulator's complete reference

This document is everything the simulator promises. If a fact is not written
here, it is not promised. Every number below is read from the simulator's own
configuration and from the rig description; nothing is approximate unless it
says so.

You are writing a ROS 2 (Jazzy) node. There is **no starter code**. `INFO.md`
covers installation, how to run the simulator, and a suggested order of work.
This page is the contract: what exists and what every number is.

Contents

1. [Numbers at a glance](#1-numbers-at-a-glance)
2. [Conventions and frames](#2-conventions-and-frames)
3. [The arm](#3-the-arm)
4. [The camera](#4-the-camera)
5. [The TF tree](#5-the-tf-tree)
6. [The mounting panel and the keyboard](#6-the-mounting-panel-and-the-keyboard)
7. [Topics, services and QoS](#7-topics-services-and-qos)
8. [Message semantics, field by field](#8-message-semantics-field-by-field)
9. [Pressing a key](#9-pressing-a-key)
10. [Episode lifecycle](#10-episode-lifecycle)
11. [The dashboard](#11-the-dashboard)
12. [What is deliberately not provided](#12-what-is-deliberately-not-provided)

---

## 1. Numbers at a glance

| what | value |
|---|---|
| joints, in order | `base_yaw, shoulder_pitch, elbow_pitch, head_pan, head_tilt` |
| joint limits (deg) | `[−120, +120]`, `[−30, +100]`, `[−140, 0]`, `[−45, +45]`, `[−35, +35]` |
| `v_max` (rad/s) | `0.6, 0.6, 0.8, 1.0, 1.0` |
| `a_max` (rad/s²) | `1.5, 1.5, 2.0, 3.0, 3.0` |
| links (m) | base height 0.30, upper arm 0.60, forearm 0.40 |
| stylus stroke window (m) | 0.05 … 0.35 along the aim ray |
| home pose | `(0°, 85°, −113°, 0°, 0°)` |
| command watchdog | 0.10 s without a command → all joints decelerate to zero |
| press: max incidence / max joint speed / debounce | 55° / 0.02 rad/s / 0.15 s |
| camera | 1280 × 720, `fx = fy = 900`, `cx = 639.5`, `cy = 359.5`, no distortion, 15 Hz, `bgr8` |
| panel | 0.400 m × 0.175 m, black, four `DICT_4X4_50` markers of side 0.020 m, IDs 0–3 |
| marker centres (board frame, m) | 0: (0.016, 0.016) · 1: (0.384, 0.016) · 2: (0.384, 0.159) · 3: (0.016, 0.159) |
| keyboard | Redragon K552, ANSI 87-key TKL, somewhere inside the panel — **placement not provided** |
| launch key | 3–6 characters from `A–Z 0–9`, on `/sim/launch_key` |

---

## 2. Conventions and frames

* Units on every ROS interface: **metres, radians, seconds**. (The browser
  dashboard shows degrees for humans; ROS never does.)
* Rotation matrices are *world-from-local*: `P_world = R_WX @ p_local + t_WX`.
  The columns of `R_WX` are the local axes expressed in world.
* Frames (all right-handed):

| frame | origin | X | Y | Z |
|---|---|---|---|---|
| **world** | arm base, on the ground | forward (toward the panel) | left | up |
| **head** | tip of the forearm | neutral aim direction = camera optical axis | left | up-ish |
| **camera** (optical) | same point as the head origin | right (`= −Y_head`) | down (`= −Z_head`) | forward (`= +X_head`), OpenCV convention |
| **board** | mounting-panel **top-left** corner *as seen from the arm* | right | **down** | **into the panel** (away from the arm) |

  The board frame is the image-pixel convention (x right, y down) in metres.
  The panel's face normal pointing back toward the arm is therefore **−Z_board**,
  and the arm always sits on the negative-Z side of the panel.

* Joint vector order, everywhere (`/joint_states`, this document, the
  dashboard): `base_yaw, shoulder_pitch, elbow_pitch, head_pan, head_tilt`.
* The simulator runs in **real time** on the ROS clock. There is no
  `use_sim_time`. Nothing pauses while your code thinks.
* The simulator is one node, `/autotype_sim`. Where the panel is, is **not**
  anywhere in what it offers.

---

## 3. The arm

A 5-joint arm: a yaw base, a shoulder pitch, an elbow pitch, and a pan/tilt
head that carries a stylus. The only control interface is **joint velocity**.
There is no position controller, no Cartesian controller, no inverse
kinematics and no "go to key" service. The camera is rigidly attached to the
*forearm*, not to the pan/tilt head, so once the first three joints stop the
image stops moving, whatever pan and tilt do.

### 3.1 Joint table

Index = position in every 5-vector. Limits are hard stops: a joint driven into
a stop stays there with zero velocity and can leave again immediately.

| idx | name | q_min | q_max | v_max | a_max | positive direction, in words |
|---|---|---|---|---|---|---|
| 0 | `base_yaw` | −120° = −2.0944 rad | +120° = +2.0944 rad | 0.6 rad/s (34.4°/s) | 1.5 rad/s² (85.9°/s²) | turns the whole arm to **its left**, i.e. toward world +Y |
| 1 | `shoulder_pitch` | −30° = −0.5236 rad | +100° = +1.7453 rad | 0.6 rad/s (34.4°/s) | 1.5 rad/s² (85.9°/s²) | measured from horizontal; positive **raises** the upper arm (0 = horizontal forward, 90 = straight up) |
| 2 | `elbow_pitch` | −140° = −2.4435 rad | 0° = 0 rad | 0.8 rad/s (45.8°/s) | 2.0 rad/s² (114.6°/s²) | angle of the forearm **relative to the upper arm**; positive raises the forearm; 0 = straight; the elbow only ever bends *down* |
| 3 | `head_pan` | −45° = −0.7854 rad | +45° = +0.7854 rad | 1.0 rad/s (57.3°/s) | 3.0 rad/s² (171.9°/s²) | swings the stylus toward head **+Y (left)** |
| 4 | `head_tilt` | −35° = −0.6109 rad | +35° = +0.6109 rad | 1.0 rad/s (57.3°/s) | 3.0 rad/s² (171.9°/s²) | raises the stylus toward head **+Z (up)** |

The absolute pitch of the forearm — and of the camera axis — from horizontal
is `shoulder_pitch + elbow_pitch`. Decelerating from `v_max` to rest at `a_max`
takes 0.40 s for joints 0–2 and 0.33 s for the head, plus the tracking error
of section 3.4.

### 3.2 Link lengths, stylus, home pose

| constant | value |
|---|---|
| `base_height` (ground → shoulder axis) | 0.30 m |
| `upper_arm` (shoulder → elbow) | 0.60 m |
| `forearm` (elbow → head/camera origin) | 0.40 m |
| `stylus_min` | 0.05 m — the stylus reaches nothing closer than this along its ray |
| `stylus_max` | 0.35 m — and nothing farther than this |
| `q_home` | `(0°, 85°, −113°, 0°, 0°)` = `(0, 1.4835, −1.9722, 0, 0)` rad |

Every episode starts with the arm at rest at `q_home`: upper arm nearly
vertical, forearm pitched 28° below horizontal. At home the shoulder is at
world `(0, 0, 0.30)`, the elbow at `(0.052, 0, 0.898)`, the head/camera origin
at `(0.405, 0, 0.710)`, and the camera looks along `(0.883, 0, −0.469)`. From
this pose **all four markers are fully inside the image for every seed** — the
pose sampler enforces it — and in practice the whole panel is.

### 3.3 Kinematic definition (world frame)

With `c0, s0 = cos, sin(base_yaw)`, `θ1 = shoulder_pitch`, and
`φ = shoulder_pitch + elbow_pitch`:

```
p_shoulder = (0, 0, 0.30)
p_elbow    = p_shoulder + 0.60 · (cos θ1 · c0,  cos θ1 · s0,  sin θ1)
p_head     = p_elbow    + 0.40 · (cos φ  · c0,  cos φ  · s0,  sin φ)

X_head = ( cos φ · c0,   cos φ · s0,   sin φ)     # forearm direction = camera optical axis
Y_head = (−s0,           c0,           0    )
Z_head = (−sin φ · c0,  −sin φ · s0,   cos φ)
R_world_head = [X_head | Y_head | Z_head]           # columns

stylus direction in the head frame (pan = azimuth, tilt = elevation):
u_head = (cos tilt · cos pan,  cos tilt · sin pan,  sin tilt)
aim_world = R_world_head @ u_head
stylus tip at range r:  p_head + r · aim_world

camera:  R_world_cam = [−Y_head | −Z_head | X_head],   t_world_cam = p_head
```

Pan/tilt is a plain spherical aim. For a given head position there is exactly
one `(pan, tilt)` pointing at a given target, and the range to that target is
whatever the geometry gives; there is no extension axis and no redundancy.

### 3.4 How the plant behaves

* It is a velocity-mode actuator integrated at 50 Hz. A command sets a target
  velocity; the actuator's velocity moves toward that target at no more than
  `a_max`, so a step from `v_max` to zero takes several hundred milliseconds,
  not one tick.
* **Velocity tracking is imperfect, and the error grows with speed.** Two
  identical commands do not produce identical motion. The imperfection is
  deterministic within an episode — replaying an episode replays the same
  motion — but you are given neither its size nor its statistics.
* **`/joint_states` is exact.** The positions are the true positions; the
  velocities are the true finite-difference velocities; there is no latency
  and no encoder noise. Whatever the actuator did, where it ended up is
  visible exactly.
* Hard stops: a joint at `q_min`/`q_max` has its velocity zeroed. Commanding
  further into the stop does nothing.
* Watchdog (section 8.5): 0.10 s without a command message → target velocity
  zero on all five joints.

---

## 4. The camera

| parameter | value |
|---|---|
| image size `W × H` | 1280 × 720 |
| `fx`, `fy` | 900 px, 900 px |
| `cx`, `cy` | 639.5 px, 359.5 px (pixel centres on integer coordinates: the optical axis passes through the centre of the image) |
| horizontal FOV | 2·atan(640/900) = **70.8°** |
| vertical FOV | 2·atan(360/900) = 43.6° |
| distortion | **none** (`D = 0`) |
| rate | 15 Hz |
| frame | `camera_optical_frame` (OpenCV: z forward, x right, y down) |

The camera is an ideal pinhole with those intrinsics. A world point `P` has
camera-frame coordinates `P_C = R_world_camᵀ @ (P − t_world_cam)`; it is in
front of the camera when `P_C.z > 0`, and it images at

```
u = fx · P_C.x / P_C.z + cx
v = fy · P_C.y / P_C.z + cy
```

with `0 ≤ u ≤ W−1` and `0 ≤ v ≤ H−1` inside the frame.

Three properties worth underlining:

* The camera is fixed to the **forearm**. Panning and tilting the head does
  not move the image. Once `base_yaw`, `shoulder_pitch` and `elbow_pitch` are
  still, the image is still.
* The camera origin and the stylus origin are **the same point** (`p_head`),
  and the camera's optical axis is the stylus's neutral (pan = tilt = 0)
  direction. The stylus at rest is therefore a reticle at the exact image
  centre, and the *direction* to anything visible is fixed by its pixel alone
  through `K⁻¹`. The *range* and the *incidence* to it — both of which the
  press check uses (section 9) — are not fixed by the pixel.
* The field of view is narrow at typing distance. At 0.25 m from the panel the
  image spans about 0.36 m × 0.20 m of it, which is **less than the panel's
  0.40 m × 0.175 m**: from a typing pose some or all of the four markers are
  outside the frame. From the home pose the whole panel is in view, and a
  marker is then about 24 px across — roughly 3–4 px per marker module.

The image contains the rendered panel on a black background and nothing else:
no arm, no floor, no lighting model, no noise, no blur. `cv_bridge` is **not
installed** in the provided environment; the OpenCV that is provided is newer
than the one the system package is built against.

---

## 5. The TF tree

Published by the simulator: dynamic transforms at 50 Hz on `/tf`, static ones
once on `/tf_static`. Names follow REP-103/REP-105.

```
world
└── base_link              static, identity
    └── shoulder_link      Rz(base_yaw),  translation (0, 0, 0.30)
        └── upper_arm_link Ry(−shoulder_pitch)
            └── forearm_link   translation (0.60, 0, 0), then Ry(−elbow_pitch)
                ├── camera_link            static: translation (0.40, 0, 0), identity
                │                          (x forward along the forearm, z up — REP-103 body frame)
                │   └── camera_optical_frame   static: pure rotation, see below
                └── head_link              translation (0.40, 0, 0), then Rz(head_pan) · Ry(−head_tilt)
                    └── stylus_tip         static: translation (0.35, 0, 0)  (= stylus_max along the aim; for RViz)
```

`camera_link → camera_optical_frame` is the standard ROS body-to-optical
rotation. As a matrix whose columns are the optical x, y, z axes expressed in
`camera_link`:

```
R_link_optical = [[ 0,  0,  1],
                  [-1,  0,  0],
                  [ 0, -1,  0]]
```

i.e. optical x = −link y (right), optical y = −link z (down), optical z =
+link x (forward). Equivalent forms: roll-pitch-yaw `(−90°, 0, −90°)`;
quaternion `(x, y, z, w) = (−0.5, 0.5, −0.5, 0.5)` (a quaternion and its
negation are the same rotation, so `(0.5, −0.5, 0.5, −0.5)` is also correct).

Guarantees:

* The transform from `world` to `camera_optical_frame` equals the camera pose
  of section 3.3 (`R_world_cam`, `t_world_cam`) to numerical precision, and the
  x-axis of `head_link` expressed in `world` equals `aim_world`. The TF tree
  and the formulas agree.
* **There is no `board` frame**, now or ever. The panel pose is not on TF.
* The tf2 convention: a lookup with target frame `world` and source frame
  `camera_optical_frame` yields the pose of the camera *in* the world frame —
  `P_world = R @ P_optical + t`.
* Transforms are not available for the first moment after start; a lookup
  raises until the first transforms have arrived, roughly a second after
  launch. The same pose is derivable from the section 3.3 formulas and the
  `/joint_states` positions.

---

## 6. The mounting panel and the keyboard

### 6.1 The panel

A flat black panel, **0.400 m wide × 0.175 m high**, with one ArUco marker
near each corner and a keyboard fixed somewhere inside. Board-frame
coordinates (section 2: origin at the panel's top-left corner as seen from the
arm, x right, y down, z into the panel, metres):

| item | value |
|---|---|
| marker dictionary | `DICT_4X4_50` |
| marker side length | **0.020 m** (2 × 2 cm, black border included) |
| marker IDs | 0, 1, 2, 3 — one per corner, clockwise from top-left as seen from the arm |

| ID | corner | centre (x, y) | TL | TR | BR | BL |
|---|---|---|---|---|---|---|
| 0 | top-left | (0.016, 0.016) | (0.006, 0.006) | (0.026, 0.006) | (0.026, 0.026) | (0.006, 0.026) |
| 1 | top-right | (0.384, 0.016) | (0.374, 0.006) | (0.394, 0.006) | (0.394, 0.026) | (0.374, 0.026) |
| 2 | bottom-right | (0.384, 0.159) | (0.374, 0.149) | (0.394, 0.149) | (0.394, 0.169) | (0.374, 0.169) |
| 3 | bottom-left | (0.016, 0.159) | (0.006, 0.149) | (0.026, 0.149) | (0.026, 0.169) | (0.006, 0.169) |

All marker points have `z = 0`. Markers are printed **upright**: a marker's own
x axis runs along +x_board and its y axis (the ArUco convention has y up) along
−y_board, so the printed marker's TL, TR, BR, BL corners are in the order given
in the table. Each marker sits inside a white quiet zone, at least one module
wide, on the black panel. The board frame is right-handed with its z pointing
away from the camera, into the panel.

### 6.2 Where the panel can be

Per episode the panel is placed once and held fixed for the whole episode.
**You are not given the placement, nor how it is drawn, and the `seed` does not
determine it.** The placement varies from episode to episode within the sanity
band below.

What you are given is a deliberately rounded, deliberately generous **sanity
band**, and nothing finer:

* the panel sits roughly **0.9–1.2 m in front of the arm**, near its height,
  a little either side of straight ahead;
* it faces the arm, within about **±30° of yaw** and about **±15° of tilt**
  (pitch and roll together) of square-on;
* the orientation it is perturbed from is `x_board = −Y_world, y_board =
  −Z_world, z_board = +X_world` (panel facing the arm, top edge up).

The band is a coarse plausibility check and nothing more. An estimate outside
it is certainly wrong; an estimate inside it may still be several centimetres
and several degrees off, which is more than a key. A pose that satisfies the
band is not thereby the pose.

### 6.3 The keyboard

The keyboard is a **Redragon K552 — an ANSI 87-key tenkeyless (TKL)** — lying
flat on the panel, upright as seen from the arm, with its function row toward
the top edge of the panel. It is **not** centred on the panel, and you are
given no bound on where it sits. What the camera sees is a photograph of that
keyboard rendered onto the panel.

**Not provided, ever:** the keyboard's offset inside the panel, and the
position of any key in the board frame.

**Public, because it is the keyboard standard and not the rig:** the key
layout in key units. 1u = the standard Cherry MX pitch, **19.05 mm**. Every key
is 1u tall. Rows are listed top to bottom, with the top-left corner `(x, y)` of
each key's cell in units from the top-left of the key grid, and widths `w` in
units. The function row is at `y = 0`; the K552 has a **0.25u gap** below it
(measured, not assumed — many TKLs use 0.5u), so the remaining rows sit at
`y = 1.25, 2.25, 3.25, 4.25, 5.25`. The whole grid is 18.25u × 6.25u.

```
y=0     ESC@0  F1@2 F2@3 F3@4 F4@5  F5@6.5 F6@7.5 F7@8.5 F8@9.5  F9@11 F10@12 F11@13 F12@14
        PRTSC@15.25 SCRLK@16.25 PAUSE@17.25
y=1.25  `@0 1@1 2@2 3@3 4@4 5@5 6@6 7@7 8@8 9@9 0@10 -@11 =@12  BACKSPACE@13 w2
        INS@15.25 HOME@16.25 PGUP@17.25
y=2.25  TAB@0 w1.5  Q@1.5 W@2.5 E@3.5 R@4.5 T@5.5 Y@6.5 U@7.5 I@8.5 O@9.5 P@10.5 [@11.5 ]@12.5  \@13.5 w1.5
        DEL@15.25 END@16.25 PGDN@17.25
y=3.25  CAPS@0 w1.75  A@1.75 S@2.75 D@3.75 F@4.75 G@5.75 H@6.75 J@7.75 K@8.75 L@9.75 ;@10.75 '@11.75  ENTER@12.75 w2.25
y=4.25  LSHIFT@0 w2.25  Z@2.25 X@3.25 C@4.25 V@5.25 B@6.25 N@7.25 M@8.25 ,@9.25 .@10.25 /@11.25  RSHIFT@12.25 w2.75
        UP@16.25
y=5.25  LCTRL@0 LWIN@1.25 LALT@2.5 (w1.25 each)  SPACE@3.75 w6.25  RALT@10 RWIN@11.25 MENU@12.5 RCTRL@13.75 (w1.25 each)
        LEFT@15.25 DOWN@16.25 RIGHT@17.25
```

(Width `w` is 1 unless stated.) A key's centre is at `(x + w/2, y + 0.5)` units
from the grid's top-left corner. Where that grid corner sits on the panel is
not provided.

**Registration.** A press registers on a key when the stylus ray meets the
panel plane inside that key's rectangle: the key's cell, shrunk by a small
inset on every side, so that a cap is a little smaller than its cell. The
narrow gap between neighbouring caps belongs to no key (`NO_KEY`), as does
everything on the panel outside the keyboard. A 1u cell is 19.05 mm across, so
aiming at the centre of a 1u key leaves you the better part of a centimetre of
error in every direction before the ray leaves the cap.

**Key names and what they type.** Names are as in the table above. There are
three kinds:

| kind | keys | effect of an accepted press on the typed string |
|---|---|---|
| `char` | `A–Z`, `0–9`, `SPACE` (types `' '`), and the punctuation keys `` ` - = [ ] \ ; ' , . / `` (each types its unshifted glyph) | appends one character (letters uppercase) |
| `backspace` | `BACKSPACE` | removes the last character (no-op when the string is empty) |
| `other` | everything else: ESC, F1–F12, PRTSC, SCRLK, PAUSE, INS, HOME, PGUP, DEL, END, PGDN, TAB, CAPS, ENTER, LSHIFT, RSHIFT, LCTRL, LWIN, LALT, RALT, RWIN, MENU, RCTRL, UP, DOWN, LEFT, RIGHT | counted as an accepted press, string unchanged |

Punctuation is `char` on purpose: a mis-aim onto `;` puts a `;` into the typed
string, exactly as on the real rig, and it has to be backspaced out.

---

## 7. Topics, services and QoS

| name | type | direction | QoS | rate | `header.frame_id` |
|---|---|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | sim → you | reliable, keep-last 10 | 50 Hz | — |
| `/camera/image_raw` | `sensor_msgs/Image` (`bgr8`) | sim → you | **best-effort sensor-data** (see 7.1) | 15 Hz | `camera_optical_frame` |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | sim → you | reliable, **transient-local** (latched, published once) | once | `camera_optical_frame` |
| `/sim/launch_key` | `std_msgs/String` | sim → you | reliable, transient-local (latched) | at episode start and after every reset | — |
| `/sim/result` | `autotype_msgs/EpisodeResult` | sim → you | reliable, transient-local (latched) | once per `/sim/done` | — |
| `/sim/press_feedback` | `autotype_msgs/PressFeedback` | sim → you | reliable | per press, **only** when launched with `debug_press_feedback:=true` | — |
| `/arm/press` | `std_msgs/Empty` | **you → sim** | reliable | one message per press attempt | — |
| `/arm/cmd_joint_velocity` | `autotype_msgs/JointVelocityCommand` | **you → sim** | reliable, keep-last 1 | whatever you publish; the watchdog window is 0.10 s (section 8.5) | — |
| `/sim/done` | `std_msgs/Empty` | **you → sim** | reliable | once, at the end of the attempt | — |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | sim → you | standard tf2 | dynamic 50 Hz, statics once | see section 5 |
| `/sim/reset` | `std_srvs/Trigger` (service) | you → sim | service | on demand | — |

Launch arguments that change what is published: `seed` (default 1, selects the
episode), `launch_key` (default `ROVER`), `teleop` (default false),
`debug_press_feedback` (default false), `port` (dashboard, default 8080).

### 7.1 QoS profiles

`/camera/image_raw` is published with the **sensor-data profile: best-effort
reliability, volatile durability, keep-last history** — as a real camera driver
publishes. A subscriber created with the client library's default profile
(reliable) is *incompatible* with a best-effort publisher: it receives **zero
callbacks, and no error is raised**. The compatible profile is best-effort
reliability, volatile durability, keep-last history.

Three topics are **latched**: `/camera/camera_info`, `/sim/launch_key` and
`/sim/result`, all published with reliable reliability and **transient-local**
durability, keep-last depth 1. Each is published once, or once per episode. A
subscriber that requests transient-local durability still receives the last
message even though it started later; one that requests the default volatile
durability receives only messages published after it subscribes. The stored
message arrives shortly after the subscription is matched by discovery, not at
the instant the subscription is created — a latched topic read once,
immediately, reads empty.

`/joint_states`, `/arm/cmd_joint_velocity`, `/arm/press`, `/sim/done` and
`/sim/press_feedback` are compatible with the client library's defaults
(reliable, volatile).

`ros2 topic echo` and `ros2 topic hz` adapt their own subscription to the
publisher's QoS automatically; `ros2 topic info -v <topic>` prints the
publisher's profile.

---

## 8. Message semantics, field by field

### 8.1 `/joint_states` — `sensor_msgs/JointState` (50 Hz)

| field | meaning |
|---|---|
| `header.stamp` | simulator clock at this tick |
| `name[5]` | always the five names, in the order `base_yaw, shoulder_pitch, elbow_pitch, head_pan, head_tilt`; joints are nevertheless identified by name |
| `position[5]` | **true** joint angles, rad. No encoder noise, no latency. |
| `velocity[5]` | measured joint velocity, rad/s: `(position_now − position_previous) / 0.02 s`. This is the same number the press check uses for its "arm is still" test (section 9). Zero until the first tick. |
| `effort` | empty or zeros; not meaningful |

### 8.2 `/camera/image_raw` — `sensor_msgs/Image` (15 Hz)

| field | value |
|---|---|
| `height`, `width` | 720, 1280 |
| `encoding` | `"bgr8"` (OpenCV byte order) |
| `is_bigendian` | 0 |
| `step` | row stride in bytes, `width * 3 = 3840` |
| `data` | `height * step` bytes, row-major, top row first |
| `header.frame_id` | `camera_optical_frame` |

### 8.3 `/camera/camera_info` — `sensor_msgs/CameraInfo` (latched, once)

| field | value |
|---|---|
| `width`, `height` | 1280, 720 |
| `distortion_model` | `"plumb_bob"` |
| `d` | `[0, 0, 0, 0, 0]` — **there is no lens distortion** |
| `k` | `[900, 0, 639.5,  0, 900, 359.5,  0, 0, 1]` (row-major 3×3) |
| `r` | identity |
| `p` | `[k | 0]` |

The same numbers appear in section 4; the topic exists so that standard
tooling works.

### 8.4 `/sim/launch_key` — `std_msgs/String` (latched)

`data` is the string to be typed: 3–6 characters from `A–Z 0–9`, always
uppercase. Published at the start of every episode and after every reset. The
launch key does not change on reset; it is a launch argument.

### 8.5 `/arm/cmd_joint_velocity` — `autotype_msgs/JointVelocityCommand` (you publish)

```
std_msgs/Header header
string[]  name        # any subset of the five joint names, any order
float64[] velocity    # rad/s, same length as name
```

Rules the simulator applies, in this order:

1. If `len(name) != len(velocity)` the **whole message is dropped** with a
   warning. A `NaN`/`inf` velocity also drops the message.
2. Joints are matched **by name**. A name that is not one of the five is
   ignored (warning, throttled to 1/s); the rest of the message is still
   applied.
3. Each named joint's commanded velocity is replaced. **Joints omitted from a
   message keep their previous commanded velocity** — a message naming only
   `head_pan` does not stop the other four. A joint stops only when it is
   commanded `0.0` explicitly.
4. Commands are clipped to `±v_max` of each joint (section 3.1).
5. **Watchdog: if no command message arrives for 0.10 s, the target velocity of
   *all five* joints becomes zero** and the arm decelerates to rest. Any
   accepted message, even one naming a single joint, resets the watchdog for
   all joints. One message followed by silence therefore yields at most 0.10 s
   of motion.
6. After `/sim/done`, commands are ignored until the next reset.

`header.stamp` is informational; the watchdog is measured on the simulator's
clock at the moment the message arrives. The commanded velocity is a *target*:
the actuator ramps toward it at a bounded acceleration and tracks it
imperfectly (section 3.4).

### 8.6 `/arm/press` — `std_msgs/Empty` (you publish)

One message is one press attempt, evaluated **instantly** against the arm's
current state. The arm does not move during a press: the simulator checks
whether a stylus stroke from the head origin along the current aim direction
would land on a key (section 9). A press published after `/sim/done` is not
counted; the dashboard logs it as `FINISHED`.

### 8.7 `/sim/done` — `std_msgs/Empty` (you publish)

Ends the episode and triggers `/sim/result`. A `/sim/done` with no accepted
presses still produces a result, with the typed string empty. A second
`/sim/done` is ignored with a warning.

### 8.8 `/sim/result` — `autotype_msgs/EpisodeResult` (latched, once per `/sim/done`)

```
std_msgs/Header header
int64    seed                 # board-pose seed of this episode
string   target               # the launch key you were asked to type
string   typed                # what the simulator actually registered (uppercase)
bool     exact_match          # typed == target
uint32   edit_distance        # Levenshtein(typed, target): insertions + deletions + substitutions
uint32   presses_attempted    # every /arm/press sent during the episode
uint32   presses_accepted     # presses that registered a key (any kind, including BACKSPACE and "other")
string[] rejection_reasons    # parallel arrays: reason name -> how many presses failed with it
uint32[] rejection_counts
float64  elapsed              # seconds from episode start to /sim/done
```

This is the **only** place the simulator reports which keys registered.
Nothing about the typed string is exposed during the episode, except through
the debug topic of section 8.9, which is silent unless the simulator is launched
with `debug_press_feedback:=true`.

`edit_distance` is the Levenshtein distance between `typed` and `target`: `ROVR`
against `ROVER` is 1, `ROVERR` against `ROVER` is 1, `""` against `ROVER` is 5.

### 8.9 `/sim/press_feedback` — `autotype_msgs/PressFeedback` (debug only)

Published after every press **only** when the simulator is launched with
`debug_press_feedback:=true`. The argument defaults to false, and the topic
publishes nothing without it.

It is an instrument for taking the simulator apart while you learn it, and it
is not part of an episode. What it reports back is the simulator's own answer
to the geometry the task is about deriving; with it on, a run is a run with the
problem removed, and nothing it shows is evidence that a node works.

```
std_msgs/Header header
bool    accepted
string  reason      # ACCEPTED, NO_INTERSECT, OUT_OF_REACH, TOO_CLOSE, GLANCING, NO_KEY, MOVING, DEBOUNCE
string  key         # the key that registered when accepted; may name the key under the
                    # stylus for MOVING / DEBOUNCE; empty otherwise
float64 board_x     # stylus/panel hit point in the board frame, metres; meaningful from
float64 board_y     # NO_KEY onward (0 when the ladder stopped earlier)
float64 range       # metres along the aimed ray from the head origin to the panel plane;
float64 incidence   # radians between the aim and the panel normal (0 = square on).
                    # Both are meaningful for every reason except NO_INTERSECT (0 there).
```

### 8.10 `/sim/reset` — `std_srvs/Trigger` (service)

Starts a new episode with the **next seed** (`seed+1`, `seed+2`, …): a new
panel pose, the arm back at the home pose at rest, the typed string and the
result cleared, the same launch key republished. `response.success` is true and
`response.message` names the new seed. The latched `/sim/result` is cleared, so
a node subscribing after a reset sees no stale result.

---

## 9. Pressing a key

`/arm/press` evaluates a virtual stroke: from the head origin `o = p_head`,
along the current aim `d = aim_world` (section 3.3), toward the panel plane.
With `n = −z_board` (the panel normal toward the arm) and `p0` any point on
the panel, the checks run **in this order and stop at the first failure**:

| step | test | reason if it fails | threshold |
|---|---|---|---|
| 1 | the ray must point at the panel face and meet its (infinite) plane in front of the head: `d·n < 0` and `t = ((p0 − o)·n)/(d·n) > 0` | `NO_INTERSECT` | — |
| 2a | stroke length `t ≤ stylus_max` | `OUT_OF_REACH` | 0.35 m |
| 2b | stroke length `t ≥ stylus_min` | `TOO_CLOSE` | 0.05 m |
| 3 | incidence `arccos(−d·n) ≤ max_incidence` (angle between the aim and the panel normal; 0 = square on) | `GLANCING` | 55° |
| 4 | the hit point, in board coordinates, lies inside some key's rectangle (section 6.3) | `NO_KEY` | — |
| 5 | the arm is still: `max_i |velocity[i]|` from `/joint_states` `≤ max_joint_speed` | `MOVING` | 0.02 rad/s (≈ 1.15°/s), over all five joints |
| 6 | at least `debounce` seconds have passed since the **last accepted** press | `DEBOUNCE` | 0.15 s |
| — | otherwise | `ACCEPTED` | the key under the hit point registers |

Boundary behaviour: the reach limits are inclusive (exactly 0.35 m is fine);
incidence and speed reject only when *strictly above* the threshold; debounce
rejects only when *strictly less than* 0.15 s has elapsed since the last
accepted press. The first press of an episode is never `DEBOUNCE`.

Properties of the ladder:

* Geometry is checked before dynamics, so a press that misses every key while
  the arm is moving is reported `NO_KEY`, not `MOVING` — the dashboard log
  shows the more informative reason.
* `MOVING` is judged on the *measured* velocities that appear on
  `/joint_states`. Deceleration from full speed takes a few hundred
  milliseconds; the measured velocity of a resting arm is not exactly zero,
  but it is far below the threshold.
* Rejected presses cost nothing but a count in `rejection_reasons`; they never
  touch the typed string. Repeated accepted presses on the same key do: each
  one types the character again, and the debounce blocks only presses within
  0.15 s of the previous accepted one.
* The press does not move the arm. There is no stroke to wait out; the aim can
  change immediately afterwards.
* From a fixed `(base_yaw, shoulder_pitch, elbow_pitch)`, the set of keys
  pressable with pan/tilt alone is bounded by the pan and tilt limits, the
  stylus range window and the 55° incidence limit. The dashboard's green/red
  key overlay displays exactly that set for the current pose, for human eyes.

---

## 10. Episode lifecycle

```
launch (seed S, launch_key K)
   ├─ panel pose fixed for the episode; arm at rest at q_home; typed string ""
   ├─ /sim/launch_key ← K (latched);  /camera/*, /joint_states, /tf start
   │
   │   the episode runs; nothing is reported about what has been typed
   │
/sim/done
   ├─ /sim/result ← EpisodeResult (latched); commands ignored from now on
   └─ dashboard shows the RESULT panel
/sim/reset (service)
   └─ same as launch with seed S+1 (then S+2 …); result cleared; arm back at q_home
```

* `elapsed` counts from episode start (launch or reset) to `/sim/done`, in real
  seconds.
* Nothing about accepted keys is exposed before `/sim/done`, except on the
  debug topic.
* The simulator logs a bare press count on `/rosout` — never the key, never the
  rejection reason.

---

## 11. The dashboard

The simulator serves a web page at `http://localhost:8080` (launch argument
`port`). It is a monitoring view drawn for a person watching the run: the
panel and the stylus, the typed string so far, per-key reachability from the
current pose, joint bars, the command age and the watchdog LED, the camera
image with the stylus reticle, the seed, and the RESULT panel after
`/sim/done`.

It is a human interface, not a ROS interface, and it is outside the contract a
node is written against: section 7 is the whole of what a node is given. The
dashboard is where *you* watch what the simulator is doing, and it is what a
screen recording of a run captures.

With `teleop:=true` the dashboard also offers jog buttons and PRESS/DONE/RESET
buttons, which go through exactly the same command path and watchdog as a
node's messages. Whenever teleop is enabled the page shows a TELEOP strip. A
run driven that way demonstrates the person at the keyboard, not the node —
the arm is being operated, and nothing has been solved.

---

## 12. What is deliberately not provided

| not provided | why |
|---|---|
| The panel pose — nothing in the interface carries it: no `board` TF frame, no topic, no parameter, no log line — and the episode `seed`, which *is* public, does not determine it | estimating it from the markers is the perception task |
| Per-press verdicts on `/rosout` — the simulator logs a bare press count, never the key or the rejection reason | a per-press geometric readout would replace the perception task |
| The keyboard's offset inside the panel and any key's board-frame position | same task; the layout standard is public, its placement is not |
| A key → joint-angle solver, a "type this key" service, a position or Cartesian controller | writing the closed-loop control is the control task. The arm's forward kinematics are fully specified in section 3, so pointing the stylus is arithmetic you do yourself |
| Any position or trajectory interface — only joint **velocity** | as on the rover |
| Any characterisation of the actuator's tracking error, which the `seed` does not reproduce | discovering that the arm must be settled and verified is the point |
| Per-press feedback and the running typed string during an episode | as on the rover: the outcome arrives at the end. `debug_press_feedback` is a development instrument, not part of the episode |
| A dashboard a node can consume | the dashboard is a monitoring view drawn for a person; its HTTP and WebSocket ports are not an interface. A node has the topics above and nothing else |
| `cv_bridge` | the provided OpenCV is newer than the system one |
| A lighting model, distortion, motion blur, image noise, sensor latency | the hard parts are geometry and control, not image cleaning |
| Starter code | this document is the contract; the design is yours |
