#!/usr/bin/env bash
# Run the ROS 2 VIO node on a ROS 2 EuRoC bag played in real time.
#
#   sbatch -p short -c 4 --mem=16G -t 01:00:00 ros2/run_euroc_bag.sh \
#       BAG BACKEND OUT_DIR [SKIP_SECONDS] [CONFIG] [RATE]
#
# Run from cpp/. Needs the ros2_vio env (see README) and the package built with
#   colcon build --base-paths ros2 --build-base $W/build --install-base $W/install
set -euo pipefail
BAG=$1; BACKEND=$2; OUT=$3
# Always a float: "-p skip_seconds:=40" would be an integer parameter.
SKIP=$(printf '%.3f' "${4:-40}")
CONFIG=${5:-config/euroc_fixed.yaml}
RATE=${6:-1.0}
W=${MSCKF_WS:-/scratch/sbalamurugan/ros2_vio/msckf_ws}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG=$(realpath "$CONFIG")
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
set +u
source /scratch/sbalamurugan/ros2_vio/activate.sh
source "$W/install/setup.bash"
set -u
# A separate DDS domain per job, so parallel jobs never see each other.
export ROS_DOMAIN_ID=$(( ${SLURM_JOB_ID:-$$} % 200 + 20 ))

# Run the binary directly: the "ros2 run" wrapper does not pass SIGINT on
# cleanly, and the node writes its statistics on shutdown.
"$W/install/msckf_vio_ros/lib/msckf_vio_ros/vio_node" --ros-args \
    -p config:="$CONFIG" -p backend:="$BACKEND" -p output_dir:="$OUT" \
    -p skip_seconds:="$SKIP" -p opencv_threads:=1 > "$OUT/node.log" 2>&1 &
NODE=$!
sleep 5   # let the subscriptions come up before the first message

ros2 bag play "$BAG" --rate "$RATE" --topics /imu0 /cam0/image_raw /cam1/image_raw \
    > "$OUT/play.log" 2>&1

# Let the node drain its queue, then stop it (the destructor prints stats).
sleep 5
kill -INT $NODE
wait $NODE || true
tail -2 "$OUT/node.log"
