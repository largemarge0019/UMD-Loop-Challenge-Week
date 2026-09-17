#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /opt/autotype/install/setup.bash
# If the member has built their own workspace, overlay it too.
if [ -f /ws/install/setup.bash ]; then
    source /ws/install/setup.bash
fi

# Shared-memory across the sim/dev split.
#
# The two containers share one IPC namespace, hence one /dev/shm, but run as two
# different uids on purpose (simsvc owns /opt/autotype/private; member cannot
# read it). Fast DDS creates its segments, ports and semaphores in /dev/shm with
# boost's default 0644 — a fixed mode that umask does not affect — so each side
# could only READ the other's ring buffers: discovery would quietly find
# nothing. Both users share the primary group 'ddsshm', so each container
# relaxes the files IT OWNS (no privileges needed for that) to group-rw and the
# two sides can write to each other. Files appear whenever a participant is
# created, including in member shells, so this keeps running in the background;
# it is a no-op in any container that is alone in its /dev/shm.
#
# Set AUTOTYPE_NO_SHM_GROUP_FIX=1 to switch it off.
if [ -z "${AUTOTYPE_NO_SHM_GROUP_FIX:-}" ]; then
    (
        while : ; do
            chmod g+rw /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
            sleep 0.2
        done
    ) &
fi

exec "$@"
