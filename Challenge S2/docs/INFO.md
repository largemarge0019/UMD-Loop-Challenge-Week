# INFO — set it up, run it, and a path through the week

This is the practical page: how to get the simulator running on your machine,
how to drive it day to day, and a suggested order to build things in.

Every technical fact about the system — topics, message fields, QoS, joint
limits, camera intrinsics, the panel, the press rules — lives in
**[INTERFACES.md](INTERFACES.md)**. This page does not repeat any of it. When you
need a number, it is there.

---

## 1. What this is

A ROS 2 (Jazzy) simulator of the URC autonomous-typing task. A five-joint,
velocity-controlled arm with a camera on its forearm faces a panel carrying a
keyboard and four ArUco markers, and a launch key has to be typed on it.

What you are given: camera images, joint states, TF, the launch key, and a
browser dashboard drawn for *you* to watch the run on. What you build: a ROS 2
node that publishes joint velocities and press requests and gets the word
typed. The dashboard is a window for a person, not an input to a node — the
topics in INTERFACES.md are the whole of what a node is written against.

Read [INTERFACES.md](INTERFACES.md) before you write code. It is the whole
contract.

---

## 2. Before you start: check your architecture

You need an Ubuntu 24.04 machine with Docker on it (section 3), and nothing
else. ROS 2 Jazzy is inside the container images; a ROS 2 install on the machine
itself is neither required nor used, and the simulator is invisible to it —
discovery deliberately does not leave the containers (section 10). The machine
can be your desktop or a virtual machine — both work, and the only thing that
changes is the address you use for the dashboard (section 6).

**Run on: your Ubuntu machine**

```bash
uname -m
```

| result | tarball you need |
|---|---|
| `aarch64` / `arm64` | `urc-autotype-arm64.tar.gz` |
| `x86_64` / `amd64` | `urc-autotype-amd64.tar.gz` |

Both are published. They are not interchangeable: loading the wrong one fails
with `exec format error`, or falls back to emulation that is far too slow for a
50 Hz control loop.

The download and load commands in section 4 pick the right tarball for you from
one variable. Set it now, in the shell you will keep using:

```bash
ARCH=$(uname -m | sed 's/aarch64/arm64/; s/x86_64/amd64/')
echo "$ARCH"
```

It must print `arm64` or `amd64`. If you open a new terminal later, set it again.

If your Ubuntu is a VM, its architecture is the architecture of the laptop it
runs on — virtualisation does not change the CPU.

---

## 3. Install Docker

**Run on: your Ubuntu machine**

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

The last line puts you in the `docker` group so you do not need `sudo` for every
command. **It does not take effect in the shell you typed it in.** Log out and
back in, or run `newgrp docker`.

Check:

```bash
docker version
docker compose version
```

Expected: a `Client:` / `Server:` pair with no permission error, and a compose
version line. The reference machine runs Docker 29.1.3 and Docker Compose 2.40.3;
anything in that neighbourhood is fine. If you get `permission denied while
trying to connect to the Docker daemon socket`, the group change has not reached
your shell yet.

Ubuntu's `docker.io` package ships without `buildx`. That does not matter — you
never build an image, you only load one.

### About Python libraries

The dev container already has NumPy 1.26 and OpenCV 4.10 installed, so if you
work inside the container (which is the intended way — section 7) there is
nothing to install.

If you run anything **outside** the container, on your Ubuntu machine itself, be
aware that Ubuntu's own `python3-opencv` package is **4.6**, not 4.10, and the
two differ. Anything you run outside the container needs its own install and may
behave differently from the same code run inside.

---

## 4. Get the images and load them

You need two things:

| | what | size |
|---|---|---|
| the repo | `docker-compose.yml` and these docs | ~2 MB |
| one image tarball | both container images, for your architecture | **~316 MB** |

Clone the repo first. This page assumes `~/AutoTypingChallengeSim`, and every
command below runs from there:

**Run on: your Ubuntu machine**

```bash
git clone <url> ~/AutoTypingChallengeSim
cd ~/AutoTypingChallengeSim
```

The clone URL is on the repository's page, under the green **Code** button.

The tarball is **not** in the repo — it is a release asset, because GitHub
rejects files over 100 MB in git. Download it into the clone, either with the
GitHub CLI or with `curl`:

```bash
gh release download --pattern "urc-autotype-$ARCH.tar.gz*"
```

```bash
SLUG=$(git remote get-url origin | sed -E 's#(git@[^:]+:|https?://[^/]+/)##; s#\.git$##')
curl -L -O "https://github.com/$SLUG/releases/latest/download/urc-autotype-$ARCH.tar.gz"
curl -L -O "https://github.com/$SLUG/releases/latest/download/urc-autotype-$ARCH.tar.gz.sha256"
```

Or download both files from the Releases page in a browser and move them into
the clone. If your Ubuntu is a VM and it is easier to fetch the tarball on the
host, `scp` it across (`scp urc-autotype-<arch>.tar.gz <user>@<vm-ip>:~/AutoTypingChallengeSim/`, with `<arch>` being `arm64` or `amd64`). Copy the `.sha256` file too.
If you would rather keep it somewhere else, give `docker load -i` the path to
it instead of a bare filename; a wrong path fails with a plain `no such file or
directory` and nothing more.

Check it against the published checksum, then load it:

```bash
sha256sum -c "urc-autotype-$ARCH.tar.gz.sha256"
docker load -i "urc-autotype-$ARCH.tar.gz"
```

The tarball and its `.sha256` are ignored by git, so they will not show up as
changes in the clone.

One tarball restores **both** images — it is self-contained, every layer inside
it. It takes some seconds (3–17 s on the reference machine, longer on a slow
disk).

Check:

```bash
docker image ls urc-autotype
```

Expected: **one row for `urc-autotype:sim` and one for `urc-autotype:dev`**, in
either order. Do not match the columns against a sample — Docker changed this
table between releases, and on 29.x it merges repository and tag into one
`IMAGE` column and prints a `WARNING` line about human-readable output that is
not an error. What matters is that both rows are there. If only one is, or
neither, the `docker load` failed; rerun it and read the output.

| image | what it is |
|---|---|
| `urc-autotype:sim` | the simulator itself. It runs; you do not work in it |
| `urc-autotype:dev` | your container: the same ROS 2, the same OpenCV, the `autotype_msgs` message definitions you build against, and your workspace at `/ws` |

---

## 5. Start it

**Run on: your Ubuntu machine**, from the repo directory:

```bash
cd ~/AutoTypingChallengeSim
mkdir -p member_ws/src     # your workspace; bind-mounted into the dev container at /ws
docker compose up -d       # starts both containers in the background
```

Check both are up:

```bash
docker compose ps
```

Expected: two rows, `urc-autotype-sim` and `urc-autotype-dev`, both `Up`, and a
mapping of host port 8080 on the sim row. On current Docker that cell reads
`0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp` — one entry for IPv4 and one for
IPv6, the same mapping twice. What matters is that 8080 is there.

Read the simulator's log:

```bash
docker compose logs sim
```

Look for these two lines:

```
[autotype_sim]: dashboard listening on http://0.0.0.0:8080
[autotype_sim]: episode 0 seed 1 launch_key ROVER
```

Add `-f` to follow it live; `Ctrl-C` then stops following, not the simulator.

The `dev` container just idles until you open a shell in it (section 7). It
shares the `sim` container's network *and* IPC namespace, so as far as ROS 2 is
concerned the two containers are one host — discovery and the shared-memory
transport work between them exactly as they would inside a single container.

---

## 6. The dashboard

**Run on: your Ubuntu machine** — open a browser at:

* **`http://localhost:8080`** if Ubuntu is your desktop machine.
* **`http://<vm-ip>:8080`**, from the browser on the host, if Ubuntu is a VM.
  Find the address with `ip -4 addr show scope global` **inside the VM**. Take
  the line on the real network interface (`enp0s1`, `ens33`, `eth0`) — once
  Docker is installed this command also prints `172.17.0.1` on `docker0` and a
  `172.18.x.1` on a `br-…` bridge, which are Docker's internal bridges and are
  not reachable from your host. On UTM Shared Network, VMware NAT or VirtualBox
  Bridged, no tunnel and no port forwarding are needed. On VirtualBox's default
  NAT the VM's address is `10.0.2.x` and your host cannot reach it: add a port
  forward of host 8080 → guest 8080 and use `http://localhost:8080` from the
  host.

If the page is blank, wait a second and reload.

### What is on screen

* **Top bar** — the episode `seed`, `sim t`, a status pill and a connection
  indicator. `connecting`, or a red `DISCONNECTED` banner, means the simulator
  is not running or the page's live connection dropped.
* **Top-down panel** — the panel and its key rectangles, coloured **green where
  a key is reachable from the arm's current pose and red where it is not**, plus
  a dot showing where the stylus is pointing on the panel.
* **Camera** — exactly what is on the camera topic, with the stylus reticle
  drawn on it and an fps counter.
* **Typed** — the target string, and the string the simulator has actually
  registered. This is the readout that answers "did that press land?".
* **Arm** — a side view and a top view, a bar per joint in the canonical joint
  order, and underneath them the **command age**, the **watchdog LED** and
  `sim t`.
* **Events** — a scrolling log of presses and their verdicts, colour-keyed by
  outcome. When a press is rejected the reason appears here, and INTERFACES.md
  lists every reason and its threshold.
* **RESULT** — an overlay that appears when the episode ends, with the target,
  what was typed, presses attempted and accepted, elapsed time, edit distance
  and a rejection breakdown. Esc dismisses it.

**This page is a monitoring view drawn for a person.** It is outside the
interface a node is written against — INTERFACES.md lists that interface, and
the dashboard is not in it. This is how *you* see whether your node is getting
it right, and it is what a screen recording captures.

One practical note: do not view the dashboard through an ssh tunnel. It works,
but the in-page fps counter drops to 3–7 fps because the tunnel is the
bottleneck. Straight over the machine's own address it runs at about 10 fps of
camera and 20 Hz of state. It also makes for a much better screen recording.

---

## 7. Your shell and your first package

### 7.1 Get in

**Run on: your Ubuntu machine**, from `~/AutoTypingChallengeSim`:

```bash
docker compose exec dev bash
```

Note **`dev`**, not `sim`. Everything in the rest of this section runs **inside
the dev container**.

### 7.2 Check the environment

**Run inside the dev container:**

```bash
which ros2
```

Expected: `/opt/ros/jazzy/bin/ros2`. Every shell in the dev container loads ROS 2
and the `autotype_msgs` message definitions for you, and once you have built
your own workspace (7.3), every **new** shell loads your packages too.

If it prints nothing, load the environment by hand in that shell:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/autotype/install/setup.bash
```

Files under `/ws` survive `docker compose down`. Nothing else in the container
does, including anything you `pip install`.

### 7.3 Create, build and run a package

**Run inside the dev container:**

```bash
cd /ws/src
ros2 pkg create my_typist --build-type ament_python --license MIT \
    --node-name typist \
    --dependencies rclpy std_msgs std_srvs sensor_msgs geometry_msgs tf2_ros autotype_msgs
```

The package name must come **before** `--dependencies`, which otherwise swallows
it.

That creates `/ws/src/my_typist/` with a runnable stub at
`my_typist/my_typist/typist.py` and an entry point `typist` already wired into
`setup.py`.

```bash
cd /ws
colcon build --symlink-install
source /ws/install/setup.bash
ros2 run my_typist typist
```

Expected on the stub: `Hi from my_typist.`

`--symlink-install` means edits to existing `.py` files take effect on the next
`ros2 run` with no rebuild. Rebuild when you add a file, an entry point or a
dependency — always after changing `setup.py`.

`/ws` is bind-mounted from `./member_ws` in the repo, so the same files are at
`~/AutoTypingChallengeSim/member_ws/src/my_typist/…` on your Ubuntu machine. Edit
them with whatever editor you like there, and they survive `docker compose down`.

The container runs as uid 1000, gid 2000. The user half matches your own account
if that account is also uid 1000 (the first one an Ubuntu install creates always
is), so files the container writes under `member_ws/` are yours to edit. The
group half has no counterpart on your machine — it is the container's own group
for the shared-memory transport — so `ls -l` shows a bare numeric `2000` in the
group column for everything `colcon` writes. That is expected, not damage. To
tidy it up, or if ownership ever really is wrong, run
`sudo chown -R $USER:$USER ~/AutoTypingChallengeSim/member_ws` on your Ubuntu
machine — with the group half, `$USER:$USER`, or the numeric group stays.

Open a second shell whenever you want to watch topics while your node runs — just
`docker compose exec dev bash` again from another terminal.

### 7.4 Developing in VS Code

VS Code can open a window **inside** the dev container, so its editor, terminal
and Python autocomplete all use the container's ROS 2, OpenCV and message
packages. Nothing is installed in the container itself.

**Once**, on the computer you will run VS Code on, install
[VS Code](https://code.visualstudio.com/) (on Ubuntu:
`sudo snap install code --classic`) and the **Dev Containers** extension. If
Ubuntu is a VM and you would rather use VS Code on your laptop, also install the
**Remote - SSH** extension there.

**Each session:**

1. Start the simulator on your Ubuntu machine: `docker compose up -d` (section 5).
2. VS Code on your laptop with Ubuntu in a VM only: open the Command Palette
   (Ctrl+Shift+P, or ⇧⌘P on a Mac), run **Remote-SSH: Connect to Host…** and
   enter `<user>@<vm-ip>`. VS Code on the Ubuntu machine itself: skip this step.
3. In the Command Palette, run **Dev Containers: Attach to Running Container…**
   and choose **`/urc-autotype-dev`** — `dev`, not `sim`.
4. In the new window, **File → Open Folder…** and enter `/ws`.
5. **Terminal → New Terminal**. It runs inside the dev container with ROS 2
   already loaded, so `ros2 topic list` shows the simulator's topics.

From that terminal, build and run exactly as in 7.3. Files you save are the same
files as `member_ws/` on your Ubuntu machine.

**Make it automatic.** While attached, search the Command Palette for
**Dev Containers: Open Container Configuration File** and set it to:

```json
{
  "workspaceFolder": "/ws",
  "extensions": ["ms-python.python"]
}
```

From then on, attaching opens `/ws` and installs the Python extension in the
container for you.

**Good to know:**

- VS Code attaches as `member`, the container's normal user. Leave it that way:
  ROS 2 commands run as root inside the dev container stop that container
  receiving topics until it is recreated.
- After `docker compose down` and `up`, the next attach is slower while VS Code
  reinstalls its own server and extensions in the new container. Your files in
  `/ws` are untouched.
- If the editor underlines `import rclpy` even though your node runs, create
  `/ws/.vscode/settings.json`:

  ```json
  {
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.analysis.extraPaths": [
      "/opt/ros/jazzy/lib/python3.12/site-packages",
      "/opt/autotype/install/autotype_msgs/lib/python3.12/site-packages"
    ]
  }
  ```

---

## 8. Everyday commands

All of these run **on your Ubuntu machine**, from `~/AutoTypingChallengeSim`.

```bash
docker compose up -d              # start both containers
docker compose ps                 # are they up?
docker compose logs -f sim        # follow the simulator's log
docker compose exec dev bash      # your shell
docker compose down               # stop and remove both (fast; member_ws survives)
docker compose restart dev        # restart just your container
```

> **Avoid `docker compose restart sim` on its own.** It strands a running `dev`
> container — see the first row of the troubleshooting table. Use
> `docker compose down && docker compose up -d` instead.

### A new episode: seed and launch key

The seed selects the episode, and each episode has its own panel placement. The
launch key is the word to be typed, 3–6 characters from `A-Z 0-9`. Defaults are
seed `1` and key `ROVER`.

```bash
docker compose down
AUTOTYPE_SEED=7 AUTOTYPE_LAUNCH_KEY=MARS docker compose up -d
```

Or put

```
AUTOTYPE_SEED=7
AUTOTYPE_LAUNCH_KEY=MARS
```

in a file called `.env` next to `docker-compose.yml`, and plain
`docker compose up -d` picks it up.

Within an already-running simulator you can start a fresh episode without
restarting anything — **inside the dev container**:

```bash
ros2 service call /sim/reset std_srvs/srv/Trigger
```

It answers with the new episode number and seed, e.g.
`success=True, message='episode 1 started with seed 8'`.

### Development-only launch flags

The simulator has launch arguments meant for taking it apart while you learn
it. INTERFACES.md says what each one changes. Both change what the exercise
is: one hands back the geometry the task is about deriving, and the other puts
a person on the controls. A run with either of them on shows nothing about
whether a node can do the job.

```bash
docker compose down
AUTOTYPE_EXTRA_ARGS="teleop:=true debug_press_feedback:=true" docker compose up -d
```

### A second, isolated stack

Useful for running two seeds at once, or so one person's experiments do not
disturb another's. Give it its own compose project, container-name suffix and
host port:

```bash
AUTOTYPE_PORT=8091 AUTOTYPE_SUFFIX=-b AUTOTYPE_SEED=7 docker compose -p urcb up -d
docker compose -p urcb exec dev bash      # its shell
docker compose -p urcb down               # stop it
```

Its dashboard is on `:8091`. The two stacks do not see each other's topics —
each pair of containers has its own IPC namespace.

### Recording an attempt

Record with any screen recorder, capturing the browser window showing the
dashboard **and** the terminal where you started the simulator and where your
node is running, so the launch line and its arguments are readable, and let it
run until the RESULT overlay appears. Record against the direct address, not an
ssh tunnel — see the fps note in section 6.

---

## 9. Inspecting the running system

These are diagnostic commands. They tell you what exists and whether it is
healthy; they are not part of your node. All run **inside the dev container**.

```bash
ros2 node list                                # /autotype_sim
ros2 node info /autotype_sim                  # every topic and service it offers
ros2 topic list                               # 12 topics when the stack is healthy
ros2 topic info -v /camera/image_raw          # the publisher's exact QoS profile
ros2 topic hz /joint_states                   # ~50 Hz
ros2 topic hz /camera/image_raw               # ~15 Hz
ros2 topic echo --once /sim/launch_key        # data: ROVER
ros2 interface show autotype_msgs/msg/JointVelocityCommand
ros2 service list -t                          # /sim/reset [std_srvs/srv/Trigger]
ros2 param list                               # the simulator's parameters
```

`ros2 topic info -v` is the one to reach for when a subscription of yours is
silent: it prints the publisher's reliability, durability and history, and
INTERFACES.md says which profile each topic uses.

`ros2 topic echo` adapts to the publisher's QoS on its own here, so it shows the
camera and the latched topics without help. If you ever get silence from a topic
you can see in `ros2 topic hz`, force the match explicitly — for example
`ros2 topic echo /camera/image_raw --qos-reliability best_effort`, or
`--qos-durability transient_local` for a latched one. Note that this only fixes
the command line; a subscription in your own code carries whatever profile you
gave it.

Echoing the camera prints a wall of numbers. `--field header` limits it to the
header, which is usually all you want from the command line.

---

## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| **`ros2 topic list` in the dev container drops to just `/parameter_events` and `/rosout`** | most often you restarted `sim` under a running `dev` (`docker compose restart sim`, or a crash and restart). The two share one IPC namespace and one `/dev/shm`; when `sim` restarts, the shared-memory state `dev` discovered it through is gone, and `dev` does not re-discover it | `docker compose restart dev`, then open a fresh shell — or `docker compose down && docker compose up -d`. **`docker compose up -d` alone does not fix it**: it prints "Running" for both and changes nothing. It is not a stale `ros2 daemon` either |
| `docker: permission denied while trying to connect to the Docker daemon socket` | your user is in the `docker` group but this shell predates the change | log out and back in, or `newgrp docker`. Check that `groups` lists `docker` |
| `exec format error`, or the images refuse to start | wrong-architecture tarball | `uname -m` must match the tarball (section 2). Check `echo "$ARCH"` — if it is empty, set it again |
| `docker compose up`: "pull access denied" / image not found | an image was never loaded, or has a different tag | `docker load -i urc-autotype-<arch>.tar.gz`; `docker image ls urc-autotype` must show both `sim` and `dev` |
| The `sim` container disappears from `docker compose ps` seconds after `up -d`, while `dev` stays `Up` (so the stack looks half-healthy) | the simulator failed at startup; only `sim` dies, `dev` idles on regardless | `docker compose logs sim` — the reason is the **last few lines**, usually a single formatted `ERROR` line naming the cause rather than a Python traceback. Then `docker compose down` and fix what it names |
| `docker compose up`: "port is already allocated" / "address already in use" | something is already holding 8080 | `docker compose down`; `docker ps` and stop the leftover. Or run on another port: `AUTOTYPE_PORT=8091 docker compose up -d` |
| Dashboard unreachable from the browser | wrong address: Docker's `172.x` bridge address instead of the machine's, or a VM on VirtualBox NAT (`10.0.2.x`) that your host cannot route to | use `localhost:8080` on a desktop machine; on a VM use the address on the real interface, or add the 8080 port forward and use `localhost:8080` from the host (section 6) |
| Dashboard loads but shows no data, or a `DISCONNECTED` banner | the simulator is not running, or the page's live connection dropped | `docker compose ps`; `docker compose logs -f sim`; reload the page |
| Dashboard fps counter reads 3–7 | you are looking through an ssh tunnel | browse to the direct address (section 6) |
| **Your image callback never fires**, although `ros2 topic hz` shows the camera publishing at 15 Hz | a QoS incompatibility. No error is raised and no warning is printed; the subscription simply never matches | `ros2 topic info -v /camera/image_raw` shows the publisher's profile, and INTERFACES.md's QoS section gives it for every topic |
| **A topic that is published once never arrives** in your node | the same thing, in its durability form: latched topics are delivered to a compatible subscriber shortly after it subscribes, and not at all to an incompatible one | see the QoS section of INTERFACES.md |
| **The arm moves for a fraction of a second and stops**; the dashboard's watchdog LED lights | this is the watchdog, working as specified. It is a property of the system, not a fault | the watchdog window and what trips it are in INTERFACES.md |
| The arm keeps moving after you thought you stopped a joint | commands are per-joint and sticky — see the command message's field semantics in INTERFACES.md | — |
| `ros2: command not found` or `ModuleNotFoundError: autotype_msgs` in the dev shell | the environment is not sourced in this shell | section 7.2 |
| `colcon build` succeeds, but the node it built then fails with `ModuleNotFoundError: autotype_msgs` | an `ament_python` build does not resolve the dependencies in `package.xml`, so a build in an unsourced shell finishes cleanly and the missing package only shows up at run time | source it (7.2) in the shell you `ros2 run` from |
| Your edits to a `.py` file have no effect | built without `--symlink-install`, or you added a new file or entry point | rebuild with `colcon build --symlink-install`; always rebuild after changing `setup.py` |
| **`ros2` on your Ubuntu machine (outside the containers) sees only `/parameter_events` and `/rosout`** | DDS discovery deliberately does not leave the containers: the images pin their own ROS domain and a shared-memory-only transport, so nothing leaks onto a shared network | run every node **inside the dev container**. Do not chase this by reconfiguring the ROS 2 install on your host |
| `launch key arrived as a float/int … not text` at startup | your launch key is spelled like a number to a YAML parser (`1E5`, `0X1F`) so the characters cannot survive the round trip. Keys that look like YAML booleans (`YES`, `OFF`) are *not* affected | choose a different launch key |
| The result says nothing was typed, although you pressed | every press was rejected. The reason is in the dashboard's Events log and in the result's rejection breakdown | INTERFACES.md lists every rejection reason and its threshold |
| Presses do nothing and the arm is frozen | the episode has ended | `ros2 service call /sim/reset std_srvs/srv/Trigger` from the dev container |

---

## 11. A suggested development order

This is the one place on this page that offers an opinion, and it is only about
*order* — not about how to do any of it. The rungs are arranged so that each one
is small, each one is visible on the dashboard, and the hard parts arrive after
the tooling is already working. Nothing below tells you how to solve anything;
each rung says what to aim at and how you will know you got there.

If you are new to ROS 2, expect rungs 1–5 to take a day or two, and do not be
discouraged by how slow that feels. It is tool-learning, and it is paid back.

**1. Containers up, dashboard open.**
Done when `docker compose ps` shows both containers `Up` and the dashboard draws
a panel and an arm that hold still. *Sections 5 and 6.*

**2. A package that builds and runs.**
An empty node of your own, started with `ros2 run`, that prints something and
keeps running until you stop it. Done when you have built it once, edited a line
and seen the change take effect on the next run. *Section 7.*

**3. Read the arm.**
Subscribe to the joint-state topic and print what arrives. Done when you are
printing five joint names and five numbers at a steady rate, and the numbers
match the joint bars on the dashboard. Compare the names and their order against
the joint table in INTERFACES.md — that order is a convention you will rely on
for the rest of the week.

**4. Move one joint.**
Publish a velocity command and watch a single joint's bar move on the dashboard.
Done when you can make one joint of your choosing move in the direction you
intended, and stop. *The command message's semantics and the joint and velocity
limits are in INTERFACES.md.*

**5. Meet the watchdog.**
Stop publishing while the arm is moving, and watch the dashboard. Done when you
have seen the watchdog LED light and the command age climb, can say from those
two readouts alone when the arm was cut off and why — and your node then drives
a minute of ordinary motion without the LED lighting once. This rung costs ten
minutes if you go looking for it and an afternoon if you meet it later by
surprise.

**6. Reach and hold an angle.**
Get one joint to a target angle and keep it there. There is no position mode —
this is a loop you write. Done when the joint arrives within a tolerance you
chose, stays there without drifting or hunting, and the watchdog LED never
lights. Notice how the arm's actual motion compares with what you commanded; it
is not exact, and that gap matters later.

**7. See through the camera.**
Receive camera images and get one in front of your own eyes — saved, displayed,
whatever you like. Done when you have a frame you can look at and the panel is in
it. *If nothing arrives, the QoS section of INTERFACES.md is the first place to
look, and the troubleshooting table above says why there is no error message.*

**8. Find the panel.**
Work out where the panel is, in the world, from what the camera gives you. Done
when your node can state the panel's position and orientation, that estimate
lands inside the sanity band INTERFACES.md publishes for where the panel can be,
and it stays put — to within a few millimetres — when you move the arm and look
from somewhere else. That stability check is the real test; an estimate that
agrees with itself from two viewpoints is usually right, and one that does not is
always wrong.

**9. Find the keys.**
Work out where the individual keys are. The panel is not the keyboard. Done when
you can name a target key and say where it is in the world, and your answer holds
for keys at opposite ends of the board rather than only near one corner. *The key
layout, the key pitch and everything else that is published about the keyboard
are in INTERFACES.md — and that section is also explicit about what is not
published.* This is the hardest rung on the ladder. Expect it to take real time,
and check your answer against what you can see rather than against what you hope.

**10. Park somewhere useful.**
Get the arm into a pose from which the keys you care about are actually
reachable. Done when the dashboard's top-down panel shows the letters of your
launch key in **green**. That overlay is the simulator's own reachability check
drawn for your eyes: it tells you whether the pose you are in works, and nothing
about how to find a better one. *The stylus range, the head's pan and tilt cone and the incidence limit in
INTERFACES.md are what decide it.*

**11. One deliberate press.**
Aim at a key you chose in advance and press it once. Done when the dashboard's
Typed readout gains **the character you intended** — not merely a character — and
the Events log calls it accepted. Getting a random neighbour is a partial result,
not a pass. *When a press is rejected, the Events log names the reason and
INTERFACES.md gives the threshold behind it.*

**12. The whole word.**
Type the launch key, then declare the episode finished. Done when the result
overlay shows the typed string equal to the target and an edit distance of zero.
This rung is mostly about what your node does when it is *not* sure: a press you
decline to make costs less than a wrong character.

**13. Make it repeatable.**
Run it again on a seed you have never tuned against, from a cold start, without
touching anything. Done when it works twice in a row on unseen seeds. This is the
rung that separates something that once worked from something that works — and it
is the one worth the most.

Rungs 8, 9 and 11 are where the difficulty really is, and they are supposed to
be. If you are finding them hard you are doing the exercise, not failing it.
