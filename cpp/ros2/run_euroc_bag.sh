#!/usr/bin/env bash
# Run the ROS 2 VIO node on a ROS 2 EuRoC bag played in real time.
#
#   sbatch -p short -c 4 --mem=16G -t 01:00:00 ros2/run_euroc_bag.sh \
#       BAG BACKEND OUT_DIR [SKIP_SECONDS] [CONFIG] [RATE]
#
# Run from cpp/. Needs the ROS 2 env (ROS2_ACTIVATE, default: the cluster's
# ros2_vio env) and the package built into MSCKF_WS (see README).
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 BAG BACKEND OUT_DIR [SKIP_SECONDS] [CONFIG] [RATE]" >&2; exit 2; }
BAG=$(realpath "$1"); BACKEND=$2; OUT=$3
# Always a float: "-p skip_seconds:=40" would be an integer parameter.
SKIP=$(printf '%.3f' "${4:-40}")
CONFIG=${5:-config/euroc_fixed.yaml}
RATE=${6:-1.0}
W=${MSCKF_WS:-/scratch/sbalamurugan/ros2_vio/msckf_ws}
ROS2_ACTIVATE=${ROS2_ACTIVATE:-/scratch/sbalamurugan/ros2_vio/activate.sh}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG=$(realpath "$CONFIG")
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
set +u
source "$ROS2_ACTIVATE"
source "$W/install/setup.bash"
set -u
# A separate DDS domain per job, so parallel jobs never see each other. IDs
# 20-99 stay clear of the ports Linux hands out as ephemeral (domains >101).
export ROS_DOMAIN_ID=$(( ${SLURM_JOB_ID:-$$} % 80 + 20 ))

# Run the binary directly: the "ros2 run" wrapper does not pass SIGINT on
# cleanly, and the node writes its statistics on shutdown.
"$W/install/msckf_vio_ros/lib/msckf_vio_ros/vio_node" --ros-args \
    -p config:="$CONFIG" -p backend:="$BACKEND" -p output_dir:="$OUT" \
    -p skip_seconds:="$SKIP" -p opencv_threads:=1 > "$OUT/node.log" 2>&1 &
NODE=$!
trap 'kill -INT "$NODE" 2>/dev/null || true' EXIT

# Wait until the node is up (it logs its backend once subscribed).
for _ in $(seq 60); do
  grep -q "backend" "$OUT/node.log" 2>/dev/null && break
  sleep 0.5
done

ros2 bag play "$BAG" --rate "$RATE" --topics /imu0 /cam0/image_raw /cam1/image_raw \
    > "$OUT/play.log" 2>&1

# Let the node drain its queue: wait until the timing log stops growing.
prev=-1
while true; do
  n=$(wc -l < "$OUT/timing_ros.csv")
  [[ "$n" == "$prev" ]] && break
  prev=$n
  sleep 3
done

kill -INT "$NODE"
trap - EXIT
status=0
wait "$NODE" || status=$?
echo "vio_node exited with status $status"
tail -1 "$OUT/node.log"
