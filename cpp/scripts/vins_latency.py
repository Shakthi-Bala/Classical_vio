"""
End-to-end latency of VINS-Fusion from a bag recorded during a real-time run
(/vins_estimator/odometry + /imu0, see ros2 run scripts): for each pose,
  latency = receive time of the odometry - receive time of the IMU sample with
            the same timestamp (= when that frame's data was played).
This matches what vio_node logs as end-to-end latency (data in -> pose out).

Usage (ros2_vio env, which has `rosbags`):
  python scripts/vins_latency.py /scratch/.../latency_MH_01_easy/latency_bag results/vins/latency_MH_01_easy.csv
"""
import sys

import numpy as np
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def main():
    bag, out = Path(sys.argv[1]), sys.argv[2]
    imu_recv, odom = {}, []
    # rosbag2 sqlite bags carry no type definitions: use the Humble ones.
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        for conn, recv_ns, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if conn.topic == '/imu0':
                imu_recv[stamp] = recv_ns
            elif conn.topic == '/vins_estimator/odometry':
                odom.append((stamp, recv_ns))

    imu_stamps = np.array(sorted(imu_recv))
    rows = []
    for stamp, recv in odom:
        # IMU and camera share the clock in EuRoC; allow 3 ms for rounding.
        i = np.searchsorted(imu_stamps, stamp)
        best = min((j for j in (i - 1, i) if 0 <= j < len(imu_stamps)),
                   key=lambda j: abs(imu_stamps[j] - stamp))
        if abs(imu_stamps[best] - stamp) > 3_000_000:
            continue
        rows.append((stamp * 1e-9, (recv - imu_recv[imu_stamps[best]]) * 1e-6))
    lat = np.array([r[1] for r in rows])
    np.savetxt(out, np.array(rows), delimiter=',', header='timestamp,end_to_end_ms',
               comments='', fmt=['%.9f', '%.4f'])
    print(f'poses {len(lat)}  end-to-end mean {lat.mean():.2f}  p50 {np.median(lat):.2f}  '
          f'p95 {np.percentile(lat, 95):.2f}  max {lat.max():.2f} ms')


if __name__ == '__main__':
    main()
