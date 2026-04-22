
import os
import numpy as np
import cv2
from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF


CANVAS_W, CANVAS_H = 1024, 768
TRAJ_PAD = 50


class VIO(object):
    def __init__(self, config, img_queue, imu_queue, viewer=None, groundtruth=None):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        os.makedirs('output_frames', exist_ok=True)
        self.frame_idx = 0
        self.est_trajectory = []        # (x, y) for live canvas

        # Evaluation buffers
        self.est_timestamps     = []
        self.est_positions_eval = []
        self.est_rotations_eval = []

        if groundtruth is not None:
            gt_data = list(groundtruth)
            self.gt_timestamps       = np.array([gt.timestamp for gt in gt_data])
            self.gt_positions        = np.array([gt.p         for gt in gt_data])
            self.gt_quaternions_wxyz = np.array([gt.q         for gt in gt_data])
            self._init_trajectory_canvas()
        else:
            self.gt_timestamps       = None
            self.gt_positions        = None
            self.gt_quaternions_wxyz = None
            self.gt_canvas           = None

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def _init_trajectory_canvas(self):
        """Pre-draw the GT trajectory on a white canvas."""
        gt_x = self.gt_positions[:, 0]
        gt_y = self.gt_positions[:, 1]

        margin = 1.0
        self.world_x_min = gt_x.min() - margin
        self.world_x_max = gt_x.max() + margin
        self.world_y_min = gt_y.min() - margin
        self.world_y_max = gt_y.max() + margin

        scale_x = (CANVAS_W - 2 * TRAJ_PAD) / (self.world_x_max - self.world_x_min)
        scale_y = (CANVAS_H - 2 * TRAJ_PAD) / (self.world_y_max - self.world_y_min)
        self.world_scale = min(scale_x, scale_y)

        # Center trajectory in canvas
        traj_pixel_w = (self.world_x_max - self.world_x_min) * self.world_scale
        traj_pixel_h = (self.world_y_max - self.world_y_min) * self.world_scale
        self.traj_offset_x = (CANVAS_W - traj_pixel_w) / 2
        self.traj_offset_y = (CANVAS_H - traj_pixel_h) / 2

        # Draw GT trajectory as dotted gray line
        self.gt_canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
        gt_pixels = [self._world_to_pixel(p) for p in self.gt_positions]
        for i in range(1, len(gt_pixels)):
            if i % 2 == 0:
                cv2.line(self.gt_canvas, gt_pixels[i - 1], gt_pixels[i], (180, 180, 180), 1)

    def _world_to_pixel(self, world_pos):
        px = int((world_pos[0] - self.world_x_min) * self.world_scale + self.traj_offset_x)
        py = int(CANVAS_H - (world_pos[1] - self.world_y_min) * self.world_scale - self.traj_offset_y)
        return (px, py)

    def _draw_camera_frustum(self, canvas, pos_px, R):
        """Draw camera body (blue box) and axes (red=X, green=Z) at pos_px."""
        # Camera X and Z axes projected onto world XY plane
        cam_x = R[:2, 0].copy()
        cam_z = R[:2, 2].copy()
        for v in [cam_x, cam_z]:
            n = np.linalg.norm(v)
            if n > 1e-6:
                v /= n

        p = np.array(pos_px, dtype=float)
        size, half = 12, 6

        # Blue camera rectangle
        pts = np.array([
            p + size * cam_z + half * cam_x,
            p + size * cam_z - half * cam_x,
            p - half * cam_z - half * cam_x,
            p - half * cam_z + half * cam_x,
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, (255, 0, 0), 2)

        # Red X axis
        x_end = (p + 25 * cam_x).astype(int)
        cv2.line(canvas, tuple(p.astype(int)), tuple(x_end), (0, 0, 255), 2)

        # Green Z axis (forward)
        z_end = (p + 25 * cam_z).astype(int)
        cv2.line(canvas, tuple(p.astype(int)), tuple(z_end), (0, 255, 0), 2)

    def _save_frame(self, tracked_img, result, gt_pos):
        if self.gt_canvas is None:
            return

        canvas = self.gt_canvas.copy()

        # Draw estimated trajectory (solid black)
        if len(self.est_trajectory) > 1:
            for i in range(1, len(self.est_trajectory)):
                p1 = self._world_to_pixel(self.est_trajectory[i - 1])
                p2 = self._world_to_pixel(self.est_trajectory[i])
                cv2.line(canvas, p1, p2, (0, 0, 0), 1)

        # Draw camera frustum at latest estimated pose
        if result is not None and len(self.est_trajectory) > 0:
            pos_px = self._world_to_pixel(self.est_trajectory[-1])
            self._draw_camera_frustum(canvas, pos_px, result.cam0_pose.R)

        # Overlay tracked stereo image in bottom-left
        if tracked_img is not None:
            img = tracked_img.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            th = CANVAS_H // 3
            tw = CANVAS_W // 3
            img_resized = cv2.resize(img, (tw, th))

            # Ground truth text overlay on tracked image
            if gt_pos is not None:
                cv2.putText(img_resized,
                    f'GT:  {gt_pos[0]:.2f} {gt_pos[1]:.2f} {gt_pos[2]:.2f}',
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            if result is not None:
                est = result.pose.t
                cv2.putText(img_resized,
                    f'EST: {est[0]:.2f} {est[1]:.2f} {est[2]:.2f}',
                    (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            canvas[CANVAS_H - th:, :tw] = img_resized

        cv2.imwrite(f'output_frames/frame_{self.frame_idx:05d}.png', canvas)
        self.frame_idx += 1

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            result = self.image_processor.stareo_callback(img_msg)

            if result is not None:
                feature_msg, tracked_img = result
                self.feature_queue.put((feature_msg, tracked_img))

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            item = self.feature_queue.get()
            if item is None:
                self._run_evaluation()
                return

            feature_msg, tracked_img = item
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            # Look up nearest ground truth by timestamp
            gt_pos = None
            if self.gt_timestamps is not None:
                idx = int(np.searchsorted(self.gt_timestamps, feature_msg.timestamp))
                idx = min(idx, len(self.gt_timestamps) - 1)
                gt_pos = self.gt_positions[idx]

            # Accumulate estimated trajectory
            if result is not None:
                self.est_trajectory.append(result.pose.t[:2].copy())
                self.est_timestamps.append(feature_msg.timestamp)
                self.est_positions_eval.append(result.pose.t.copy())
                self.est_rotations_eval.append(result.pose.R.copy())

            self._save_frame(tracked_img, result, gt_pos)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)

    def _run_evaluation(self):
        if self.gt_timestamps is None or len(self.est_timestamps) < 5:
            print("[evaluate] Skipping — not enough data.")
            return
        print(f"\n[evaluate] Running evaluation on {len(self.est_timestamps)} frames...")
        from evaluate import generate_all_plots
        generate_all_plots(
            self.gt_timestamps,
            self.gt_positions,
            self.gt_quaternions_wxyz,
            self.est_timestamps,
            self.est_positions_eval,
            self.est_rotations_eval,
            output_dir='plots'
        )



if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/alien/YourDirectoryID_p4/MH_01_easy',
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    args = parser.parse_args()

    if args.view:
        from viewer import Viewer
        viewer = Viewer()
    else:
        viewer = None

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.)   # start from static state

    img_queue = Queue()
    imu_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue, viewer=viewer,
        groundtruth=dataset.groundtruth)

    duration = float('inf')
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
