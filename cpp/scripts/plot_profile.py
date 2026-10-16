"""
Before/after per-stage latency chart from two profile_stages.csv files written
by main_euroc.

Usage (from cpp/):
  python scripts/plot_profile.py results/cpp_profile/profile_stages.csv \
      results/cpp/profile_stages.csv results/profile/stages_before_after.png
"""
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotstyle as ps

BEFORE_COLOR = ps.NEUTRAL    # the reference
AFTER_COLOR = ps.CPP_MSCKF
TEXT = ps.TEXT

# Leaf stages that together make up a frame (inclusive parents left out).
STAGES = [
    ('frontend/*/stereo_match', 'stereo matching (KLT L-R)'),
    ('frontend/track_features/klt_temporal', 'temporal KLT'),
    ('frontend/add_new_features/fast_detect', 'FAST detection'),
    ('frontend/build_pyramids', 'pyramid building (before: inside KLT)'),
    ('backend/*/measurement_update', 'EKF update (QR, gain, P)'),
    ('backend/imu_propagation', 'IMU propagation'),
    ('backend/*/feature_jacobian', 'feature Jacobians'),
    ('backend/*/gating_test', 'chi-squared gating'),
]


def load(path):
    with open(path) as f:
        return {row['stage']: float(row['ms_per_frame']) for row in csv.DictReader(f)}


def main():
    before, after, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    labels = [label for _, label in STAGES]
    b = [before.get(key, 0.0) for key, _ in STAGES]
    a = [after.get(key, 0.0) for key, _ in STAGES]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    y = range(len(STAGES))
    h = 0.38
    ax.barh([i - h / 2 - 0.01 for i in y], b, h, color=BEFORE_COLOR,
            label=f"before  ({before['frontend'] + before['backend']:.1f} ms/frame)")
    ax.barh([i + h / 2 + 0.01 for i in y], a, h, color=AFTER_COLOR,
            label=f"after  ({after['frontend'] + after['backend']:.1f} ms/frame)")
    for i, (vb, va) in enumerate(zip(b, a)):
        ax.annotate(f'{vb:.2f}', (vb, i - h / 2 - 0.01), va='center', fontsize=7, color=TEXT,
                    xytext=(3, 0), textcoords='offset points')
        ax.annotate(f'{va:.2f}', (va, i + h / 2 + 0.01), va='center', fontsize=7, color=TEXT,
                    xytext=(3, 0), textcoords='offset points')
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel('ms per frame (inclusive)', color=TEXT)
    ax.set_title('C++ MSCKF: where the time goes, before vs after optimization',
                 color=ps.TITLE)
    ps.style_axes(ax, grid='x')
    ax.legend(frameon=False, labelcolor=TEXT, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, facecolor=ps.SURFACE)
    print('wrote', out)


if __name__ == '__main__':
    main()
