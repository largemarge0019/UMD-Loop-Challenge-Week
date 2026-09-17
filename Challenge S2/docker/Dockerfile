# URC autonomous-typing challenge simulator.
#
# Four stages, two shippable targets.
#
#   deps  — ROS 2 Jazzy + every runtime dependency + the two unprivileged users.
#           Rebuilt rarely.
#   build — colcon-builds autotype_msgs + autotype_sim from src/. No private/.
#   dev   — TARGET. The members' container: the built packages (so imports and
#           message types work), a non-root 'member' (uid 1000) owning /ws.
#           Members get a shell here.
#   sim   — TARGET. The simulator: the same built packages plus the site
#           configuration directory, owned by the non-login user 'simsvc' at
#           0400 (dir 0500), with the sim process itself running as simsvc.
#           Nobody is asked to shell in here.
#
# Build both (the maintainer, from a checkout that has private/):
#   DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile --target dev -t urc-autotype:dev .
#   DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile --target sim -t urc-autotype:sim .
#
# 'dev' is declared BEFORE 'sim' so that `--target dev` never reaches the
# `COPY private/` line: the dev image can be built from a checkout that has no
# private/ at all.

FROM ros:jazzy-ros-base AS deps

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        python3-yaml \
        python3-aiohttp \
        python3-pytest \
        ros-jazzy-std-msgs \
        ros-jazzy-std-srvs \
        ros-jazzy-sensor-msgs \
        ros-jazzy-geometry-msgs \
        ros-jazzy-tf2-ros \
        ros-jazzy-tf2-ros-py \
        ros-jazzy-rosidl-default-generators \
        ros-jazzy-rosidl-default-runtime \
        nano vim-tiny less iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# The Ubuntu python3-opencv package is 4.6, which lacks the current ArUco API
# (cv2.aruco.ArucoDetector). Members' code and the simulator both use the pip
# build. numpy is pinned below 2 to match the apt-provided ROS Python bindings.
RUN pip3 install --break-system-packages --no-cache-dir \
        "numpy<2" \
        "opencv-contrib-python-headless==4.10.0.84"

# Users. 'member' replaces the base image's own uid-1000 'ubuntu' account so the
# bind-mounted workspace keeps the uid most Linux hosts use. Both 'member' and
# (in the sim stage) 'simsvc' have the SHARED PRIMARY GROUP 'ddsshm': Fast DDS
# creates its /dev/shm segments 0644 and umask does not affect them, so two
# processes running as different uids cannot write to each other's ring buffers.
# The shared primary group plus the group-rw relaxation in entrypoint.sh is what
# lets the sim (simsvc) and member nodes (member) talk over shared memory while
# still being different uids — which is exactly what keeps private/ unreadable
# from a member shell. See docker/entrypoint.sh.
RUN groupadd -g 2000 ddsshm \
    && userdel -r ubuntu \
    && useradd -m -u 1000 -g 2000 -s /bin/bash member

# `docker exec` does NOT run the ENTRYPOINT, so an interactive shell opened with
# `docker compose exec dev bash` would otherwise have no `ros2` on PATH and no
# autotype_msgs types. Source both overlays from .bashrc so the first thing a
# member types works. Each line is guarded with -f because this runs in the
# `deps` stage, before /opt/autotype or a member's /ws workspace exist.
RUN printf '%s\n' \
        '[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash' \
        '[ -f /opt/autotype/install/setup.bash ] && source /opt/autotype/install/setup.bash' \
        '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' \
        >> /home/member/.bashrc

# Isolation: shared-memory transport ONLY. The sim container and the dev
# container share one IPC namespace (and so one /dev/shm) and one network
# namespace, so to DDS they are a single host; with no UDP transport at all
# nothing can leak onto a shared university network.
# ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST is deliberately NOT used: rmw_fastrtps
# implements it by swapping in its own fixed 512 KiB SHM segment, which drops two
# of every three 2.7 MB camera frames; sim_node sizes its segment through
# FASTDDS_BUILTIN_TRANSPORTS (appending sockets_size=8MB), which that mode ignores.
ENV FASTDDS_BUILTIN_TRANSPORTS=SHM \
    ROS_DOMAIN_ID=42 \
    RCUTILS_COLORIZED_OUTPUT=1 \
    PYTHONUNBUFFERED=1

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod 0555 /entrypoint.sh

# ---------------------------------------------------------------------------
# The package build. No private/ is present here, and none is needed: private/
# is data read at runtime, never installed.
FROM deps AS build

WORKDIR /opt/autotype
COPY src/ ./src/
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --event-handlers console_direct+ \
    && rm -rf build log

# ---------------------------------------------------------------------------
# TARGET dev — what members receive and work in.
FROM deps AS dev

# The built install tree only: autotype_msgs message types, autotype_sim for
# imports, the public assets. No site configuration is copied here and no
# AUTOTYPE_PRIVATE_DIR is set, so the simulator cannot be launched from this
# image — it runs in the sim container, and this one talks to it over DDS.
COPY --from=build /opt/autotype/install /opt/autotype/install

RUN mkdir -p /ws/src && chown -R member:ddsshm /ws

ENV HOME=/home/member
USER member
WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
# Idle: the container exists to be `docker compose exec dev bash`-ed into.
CMD ["sleep", "infinity"]

# ---------------------------------------------------------------------------
# TARGET sim — the simulator plus its site configuration. Maintainer-built only.
FROM deps AS sim

COPY --from=build /opt/autotype/install /opt/autotype/install
COPY private/ /opt/autotype/private/

# simsvc: no login shell, no home of its own beyond a writable dir for ~/.ros,
# and the only account that can read /opt/autotype/private. 0500 on the
# directory and 0400 on the files means a shell forced into this container as
# 'member' (uid 1000) gets "Permission denied", and nothing here runs as root.
RUN useradd -r -u 999 -g 2000 -d /var/lib/simsvc -m -s /usr/sbin/nologin simsvc \
    && chown -R simsvc:ddsshm /opt/autotype/private \
    && chmod 0500 /opt/autotype/private \
    && find /opt/autotype/private -type f -exec chmod 0400 {} +

ENV AUTOTYPE_PRIVATE_DIR=/opt/autotype/private \
    HOME=/var/lib/simsvc

USER simsvc
WORKDIR /opt/autotype

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "autotype_sim", "challenge.launch.py"]
