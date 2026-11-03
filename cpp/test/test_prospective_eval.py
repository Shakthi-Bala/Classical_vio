"""Known-answer tests for partial-GT evaluation, not estimator performance tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prospective_eval import associate, partial_score


class PartialGroundTruthTest(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0., .9, 30)
        p = np.column_stack([t, t**2, np.zeros_like(t)])
        self.gt = np.vstack([np.column_stack([t, p]),
                             np.column_stack([t + 10., p + [10, 0, 0]])])
        self.rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = np.array([3., 4, 2])
        positions = (self.rotation.T @ (self.gt[:, 1:] - translation).T).T
        positions[30:] += self.rotation.T @ [1, 0, 0]
        self.traj = np.column_stack([self.gt[:, 0], positions])

    def test_start_only_alignment_preserves_end_drift(self):
        result = partial_score(self.gt, self.traj, 10.9)
        self.assertAlmostEqual(result['end_segment_rmse_m'], 1., places=10)
        self.assertTrue(result['reaches_end'])
        self.assertEqual(result['end_segment_samples'], 30)

    def test_missing_gt_interval_cannot_create_accuracy_samples(self):
        augmented = np.insert(self.traj, 30, [5., 999, 999, 999], axis=0)
        result = partial_score(self.gt, augmented, 10.9)
        self.assertEqual(result['start_alignment_samples'], 30)
        self.assertEqual(result['end_segment_samples'], 30)
        self.assertAlmostEqual(result['end_segment_rmse_m'], 1., places=10)

    def test_timestamp_tolerance(self):
        selected, _ = associate(np.array([[.5, 0., 0, 0]]), self.gt[[0, 29]])
        self.assertEqual(len(selected), 0)

    def test_rejects_single_gt_segment(self):
        with self.assertRaisesRegex(ValueError, 'Separate start/end'):
            partial_score(self.gt[:30], self.traj[:30], .9)

    def test_rejects_too_few_end_samples(self):
        with self.assertRaisesRegex(ValueError, 'At least 20'):
            partial_score(self.gt, self.traj[:40], 10.9)

    def test_rejects_nonfinite_trajectory(self):
        self.traj[-1, 1] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            partial_score(self.gt, self.traj, 10.9)

    def test_incomplete_trajectory_is_visible(self):
        self.assertFalse(partial_score(self.gt, self.traj, 11.5)['reaches_end'])

    def test_result_is_strict_json_serializable(self):
        import json
        json.dumps(partial_score(self.gt, self.traj, 10.9), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
