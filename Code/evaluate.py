"""
Trajectory evaluation following Scaramuzza's methodology.

Reference:
  "A Tutorial on Quantitative Trajectory Evaluation for
   Visual(-Inertial) Odometry"
  Zhang & Scaramuzza, IROS 2018.
  https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def umeyama_alignment(src, dst, with_scale=False):
    """
    Umeyama (1991) least-squares alignment of point sets.
    Finds R, t, s  minimising  sum ||dst_i - s*R*src_i - t||^2.

    src, dst : [N, 3]
    Returns  : R [3,3], t [3], scale (float, =1 when with_scale=False)
    """
    n = len(src)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    var_s = np.mean(np.sum(sc ** 2, axis=1))
    cov   = (dc.T @ sc) / n

    U, sigma, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R     = U @ S @ Vt
    scale = float(np.dot(sigma, np.diag(S)) / var_s) if with_scale else 1.0
    t     = mu_d - scale * R @ mu_s
    return R, t, scale


def rotation_error_deg(R):
    """Geodesic rotation angle (degrees) from a rotation matrix."""
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def rot_to_euler_deg(R):
    """(roll, pitch, yaw) in degrees, ZYX convention."""
    a = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
    return float(a[2]), float(a[1]), float(a[0])   # roll, pitch, yaw


def interpolate_gt(gt_ts, gt_pos, gt_quat_wxyz, query_ts):
    """
    Linear position + SLERP orientation interpolation.
    gt_quat_wxyz : [N,4] Hamilton [qw, qx, qy, qz]
    Returns      : interp_pos [M,3], interp_rot [M,3,3]
    """
    query_ts  = np.clip(query_ts, gt_ts[0], gt_ts[-1])
    interp_pos = np.column_stack([
        np.interp(query_ts, gt_ts, gt_pos[:, i]) for i in range(3)
    ])
    q_sci  = np.column_stack([gt_quat_wxyz[:, 1], gt_quat_wxyz[:, 2],
                               gt_quat_wxyz[:, 3], gt_quat_wxyz[:, 0]])
    slerper    = Slerp(gt_ts, Rotation.from_quat(q_sci))
    interp_rot = slerper(query_ts).as_matrix()
    return interp_pos, interp_rot


def path_lengths(positions):
    """Cumulative path length (m) at each index."""
    seg = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


# ---------------------------------------------------------------------------
# Scaramuzza RPE
# ---------------------------------------------------------------------------

def compute_rpe(gt_pos, gt_rot, est_pos, est_rot, segment_lengths):
    """
    Relative Pose Error at multiple segment lengths (Scaramuzza IROS 2018).

    For each segment length L, every valid sub-trajectory starting at frame i
    and ending at frame j (where GT path i→j ≈ L) contributes one sample.

    Error formula (SE3):
        T_rel_gt  = T_gt[i]^{-1}  * T_gt[j]
        T_rel_est = T_est[i]^{-1} * T_est[j]
        E         = T_rel_gt^{-1} * T_rel_est
        e_trans   = ||t(E)|| / L * 100   [%]
        e_rot     = geodesic_angle(R(E)) [deg]

    Returns dict: {L: {'trans': ndarray [%], 'rot': ndarray [deg]}}
    """
    cum = path_lengths(gt_pos)
    N   = len(gt_pos)
    out = {}

    for L in segment_lengths:
        t_errs, r_errs = [], []

        for i in range(N):
            # advance j until GT path from i to j reaches L
            j = i + 1
            while j < N and (cum[j] - cum[i]) < L:
                j += 1
            if j >= N:
                break

            seg_len = cum[j] - cum[i]

            # Relative pose in GT frame i
            R_rel_gt = gt_rot[i].T @ gt_rot[j]
            t_rel_gt = gt_rot[i].T @ (gt_pos[j] - gt_pos[i])

            # Relative pose in EST frame i
            R_rel_est = est_rot[i].T @ est_rot[j]
            t_rel_est = est_rot[i].T @ (est_pos[j] - est_pos[i])

            # SE3 error:  E = T_rel_gt^{-1} * T_rel_est
            R_err = R_rel_gt.T @ R_rel_est
            t_err = R_rel_gt.T @ (t_rel_est - t_rel_gt)

            t_errs.append(np.linalg.norm(t_err) / seg_len * 100.0)
            r_errs.append(rotation_error_deg(R_err))

        out[L] = {'trans': np.array(t_errs), 'rot': np.array(r_errs)}

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(gt_timestamps, gt_positions, gt_quaternions_wxyz,
                       est_timestamps, est_positions_raw, est_rotations_raw,
                       output_dir='plots'):
    """
    Generate all evaluation plots following Scaramuzza IROS 2018.

    gt_quaternions_wxyz : [N,4] Hamilton [qw, qx, qy, qz]
    est_rotations_raw   : [M,3,3] body-to-world rotation matrices
    """
    os.makedirs(output_dir, exist_ok=True)

    gt_ts  = np.array(gt_timestamps)
    gt_pos = np.array(gt_positions)
    gt_q   = np.array(gt_quaternions_wxyz)
    est_ts  = np.array(est_timestamps)
    est_pos = np.array(est_positions_raw)
    est_rot = np.array(est_rotations_raw)

    if len(est_ts) < 5:
        print("[evaluate] Not enough poses to evaluate.")
        return

    N = len(est_ts)
    t_rel = est_ts - est_ts[0]

    # Interpolate GT at estimated timestamps
    gt_pos_i, gt_rot_i = interpolate_gt(gt_ts, gt_pos, gt_q, est_ts)

    # -----------------------------------------------------------------------
    # Umeyama (Sim3) alignment  — Scaramuzza §III-B
    # Stereo VIO has observable scale, so with_scale=False (SE3 alignment).
    # Change to True for monocular evaluation.
    # -----------------------------------------------------------------------
    R_align, t_align, scale = umeyama_alignment(est_pos, gt_pos_i, with_scale=False)
    aln_pos = (R_align @ est_pos.T).T * scale + t_align
    aln_rot = np.array([R_align @ R for R in est_rot])

    print(f"[evaluate] Umeyama alignment — scale = {scale:.6f}")

    # -----------------------------------------------------------------------
    # 1. Rotation Error over time  (yaw=red, pitch=green, roll=blue)
    # -----------------------------------------------------------------------
    roll_e  = np.zeros(N)
    pitch_e = np.zeros(N)
    yaw_e   = np.zeros(N)
    for i in range(N):
        R_err = gt_rot_i[i].T @ aln_rot[i]
        roll_e[i], pitch_e[i], yaw_e[i] = rot_to_euler_deg(R_err)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, yaw_e,   color='red',   lw=0.8, label='Yaw')
    ax.plot(t_rel, pitch_e, color='green', lw=0.8, label='Pitch')
    ax.plot(t_rel, roll_e,  color='blue',  lw=0.8, label='Roll')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (deg)')
    ax.set_title('Rotation Error over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_rotation_error.png'), dpi=150)
    plt.close(fig)
    print("Saved: 1_rotation_error.png")

    # -----------------------------------------------------------------------
    # 2. Translation Error over time  (x=red, y=green, z=blue)
    # -----------------------------------------------------------------------
    trans_e = aln_pos - gt_pos_i     # [N, 3]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, trans_e[:, 0], color='red',   lw=0.8, label='X')
    ax.plot(t_rel, trans_e[:, 1], color='green', lw=0.8, label='Y')
    ax.plot(t_rel, trans_e[:, 2], color='blue',  lw=0.8, label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Translation Error over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_translation_error.png'), dpi=150)
    plt.close(fig)
    print("Saved: 2_translation_error.png")

    # -----------------------------------------------------------------------
    # 3. Trajectory: X–Y and X–Z  (GT=black, estimated=blue)
    # -----------------------------------------------------------------------
    for yi, ylabel, fname in [(1, 'Y (m)', '3a_trajectory_xy.png'),
                               (2, 'Z (m)', '3b_trajectory_xz.png')]:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(gt_pos_i[:, 0], gt_pos_i[:, yi], color='deeppink', lw=1.2, label='Ground Truth')
        ax.plot(aln_pos[:, 0],  aln_pos[:, yi],  color='blue',     lw=1.0, label='Estimated')
        ax.scatter([gt_pos_i[0, 0]], [gt_pos_i[0, yi]], c='deeppink', s=60, zorder=5)
        ax.scatter([aln_pos[0, 0]],  [aln_pos[0, yi]],  c='blue',     s=60, zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel(ylabel)
        ax.set_title('Trajectory X vs Y' if yi == 1 else 'Trajectory X vs Z')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")

    # -----------------------------------------------------------------------
    # 4. Scale Drift vs Distance  (sliding window)
    # -----------------------------------------------------------------------
    cum_gt  = path_lengths(gt_pos_i)
    cum_est = path_lengths(aln_pos)
    total_dist = float(cum_gt[-1])

    win  = max(10, N // 20)
    step = max(1,  win // 2)
    s_vals, s_dist = [], []
    for i in range(0, N - win, step):
        dgt  = cum_gt[i + win]  - cum_gt[i]
        dest = cum_est[i + win] - cum_est[i]
        if dgt > 0.05:
            s_vals.append(dest / dgt)
            s_dist.append(cum_gt[i + win])   # right edge of window

    fig, ax = plt.subplots(figsize=(10, 4))
    if s_vals:
        ax.plot(s_dist, s_vals, color='blue', lw=1)
    ax.axhline(1.0, color='black', ls='--', lw=1, label='Ideal (1.0)')
    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Scale')
    ax.set_title('Scale Drift vs Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_scale_drift.png'), dpi=150)
    plt.close(fig)
    print("Saved: 4_scale_drift.png")

    # -----------------------------------------------------------------------
    # 5. Scaramuzza RPE — relative errors at multiple segment lengths
    #    Signature plot: mean ± std of trans error (%) and rot error (deg)
    #    for segment lengths = [5%, 10%, ..., 50%] of total path length.
    # -----------------------------------------------------------------------
    percentages    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    seg_lengths    = sorted(set(max(0.5, round(p * total_dist, 1))
                                for p in percentages))
    # Also include fixed metric lengths that fit within the trajectory
    fixed_lengths  = [l for l in [1, 2, 5, 10, 20, 50] if l < total_dist * 0.8]
    seg_lengths    = sorted(set(seg_lengths + fixed_lengths))

    rpe_data = compute_rpe(gt_pos_i, gt_rot_i, aln_pos, aln_rot, seg_lengths)

    # --- 5a. Relative translation error (%) vs segment length ---
    t_means = [rpe_data[L]['trans'].mean() for L in seg_lengths if len(rpe_data[L]['trans'])]
    t_stds  = [rpe_data[L]['trans'].std()  for L in seg_lengths if len(rpe_data[L]['trans'])]
    r_means = [rpe_data[L]['rot'].mean()   for L in seg_lengths if len(rpe_data[L]['rot'])]
    r_stds  = [rpe_data[L]['rot'].std()    for L in seg_lengths if len(rpe_data[L]['rot'])]
    valid_L = [L for L in seg_lengths if len(rpe_data[L]['trans'])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(valid_L, t_means, yerr=t_stds, fmt='b-o', capsize=4,
                 markersize=5, label='Mean ± Std')
    ax1.set_xlabel('Segment Length (m)')
    ax1.set_ylabel('Relative Translation Error (%)')
    ax1.set_title('RPE — Translation  [Scaramuzza IROS 2018]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.errorbar(valid_L, r_means, yerr=r_stds, fmt='r-o', capsize=4,
                 markersize=5, label='Mean ± Std')
    ax2.set_xlabel('Segment Length (m)')
    ax2.set_ylabel('Relative Rotation Error (deg)')
    ax2.set_title('RPE — Rotation  [Scaramuzza IROS 2018]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '5a_rpe_scaramuzza.png'), dpi=150)
    plt.close(fig)
    print("Saved: 5a_rpe_scaramuzza.png")

    # --- 5b. Relative errors vs distance traveled (time-series) ---
    delta_ts = max(0.5, total_dist / 100.0)
    dist_ts, rpe_t_ts, rpe_yaw_ts = [], [], []
    cum_gt2 = path_lengths(gt_pos_i)
    for i in range(N):
        j = i + 1
        while j < N and (cum_gt2[j] - cum_gt2[i]) < delta_ts:
            j += 1
        if j >= N:
            break
        sl = cum_gt2[j] - cum_gt2[i]
        R_rel_gt  = gt_rot_i[i].T @ gt_rot_i[j]
        t_rel_gt  = gt_rot_i[i].T @ (gt_pos_i[j] - gt_pos_i[i])
        R_rel_est = aln_rot[i].T @ aln_rot[j]
        t_rel_est = aln_rot[i].T @ (aln_pos[j] - aln_pos[i])
        R_err = R_rel_gt.T @ R_rel_est
        t_err = R_rel_gt.T @ (t_rel_est - t_rel_gt)
        rpe_t_ts.append(np.linalg.norm(t_err) / sl * 100.0)
        _, _, ye = rot_to_euler_deg(R_err)
        rpe_yaw_ts.append(abs(ye))
        dist_ts.append(cum_gt2[i])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if dist_ts:
        ax1.plot(dist_ts, rpe_t_ts,   color='blue', lw=0.8)
        ax2.plot(dist_ts, rpe_yaw_ts, color='red',  lw=0.8)
    ax1.set_ylabel('Relative Translation Error (%)')
    ax1.set_title(f'Relative Translation Error  (Δ={delta_ts:.2f} m)')
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Distance Traveled (m)')
    ax2.set_ylabel('Relative Yaw Error (deg)')
    ax2.set_title('Relative Yaw Error')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '5b_relative_errors_vs_dist.png'), dpi=150)
    plt.close(fig)
    print("Saved: 5b_relative_errors_vs_dist.png")

    # -----------------------------------------------------------------------
    # 6. ATE over time  (Scaramuzza §III-A — absolute drift build-up)
    # -----------------------------------------------------------------------
    ate_per_frame = np.linalg.norm(trans_e, axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, ate_per_frame, color='blue', lw=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ATE (m)')
    ax.set_title('Absolute Trajectory Error over Time  (Umeyama-aligned)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '6_ate_over_time.png'), dpi=150)
    plt.close(fig)
    print("Saved: 6_ate_over_time.png")

    # -----------------------------------------------------------------------
    # RMSE Report
    # -----------------------------------------------------------------------
    t_rmse_ax = np.sqrt(np.mean(trans_e ** 2, axis=0))
    t_rmse_3d = np.sqrt(np.mean(np.sum(trans_e ** 2, axis=1)))
    r_rmse_ax = np.sqrt(np.mean(np.vstack([roll_e, pitch_e, yaw_e]) ** 2, axis=1))
    r_rmse_3d = np.sqrt(np.mean(roll_e ** 2 + pitch_e ** 2 + yaw_e ** 2))
    s_rmse    = float(np.sqrt(np.mean((np.array(s_vals) - 1.0) ** 2))) if s_vals else float('nan')

    # RPE RMSE at each segment length
    rpe_rows = "\n".join(
        f"    L={L:5.1f} m :  trans {rpe_data[L]['trans'].mean():.3f} ± "
        f"{rpe_data[L]['trans'].std():.3f} %   "
        f"rot {rpe_data[L]['rot'].mean():.3f} ± "
        f"{rpe_data[L]['rot'].std():.3f} deg"
        for L in valid_L
    )

    report = (
        f"\n{'='*60}\n"
        f"  EVALUATION REPORT  [Scaramuzza IROS 2018 methodology]\n"
        f"  Frames: {N}    Path length: {total_dist:.2f} m\n"
        f"  Umeyama scale: {scale:.6f}\n"
        f"{'='*60}\n\n"
        f"ATE (Absolute Trajectory Error, Umeyama SE3 aligned):\n"
        f"  RMSE 3D  : {t_rmse_3d:.4f} m\n"
        f"  RMSE X   : {t_rmse_ax[0]:.4f} m\n"
        f"  RMSE Y   : {t_rmse_ax[1]:.4f} m\n"
        f"  RMSE Z   : {t_rmse_ax[2]:.4f} m\n\n"
        f"Rotation RMSE (after Umeyama alignment):\n"
        f"  Overall  : {r_rmse_3d:.4f} deg\n"
        f"  Roll     : {r_rmse_ax[0]:.4f} deg\n"
        f"  Pitch    : {r_rmse_ax[1]:.4f} deg\n"
        f"  Yaw      : {r_rmse_ax[2]:.4f} deg\n\n"
        f"Scale drift RMSE (deviation from 1.0): {s_rmse:.6f}\n\n"
        f"RPE (Relative Pose Error) [Scaramuzza]:\n"
        f"{rpe_rows}\n\n"
        f"{'='*60}\n"
    )

    print(report)
    with open(os.path.join(output_dir, 'rmse_report.txt'), 'w') as f:
        f.write(report)
    print(f"\nAll results saved to '{output_dir}/'")
