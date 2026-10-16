# Python vs C++ stereo MSCKF on EuRoC MH_01_easy

Start offset 40 s, deterministic replay (no threads), same compute node.
Accuracy from the unchanged Code/evaluate.py.

| | Python | C++ |
|---|---|---|
| poses | 2862 | 2862 |
| ATE RMSE [m] | 0.1289 | 0.1284 |
| rotation RMSE [deg] | 1.984 | 1.977 |
| RPE @ 10 m [% / deg] | 1.49 / 1.41 | 1.49 / 1.39 |
| frontend latency mean / p95 [ms] | 15.95 / 17.48 | 8.87 / 9.70 |
| backend latency mean / p95 [ms] | 18.59 / 44.02 | 3.55 / 7.83 |
| total latency mean / p95 [ms] | 34.54 / 60.22 | 12.42 / 16.72 |
| real-time factor | 1.45x | 4.02x |
| speed-up (total mean) | | 2.8x |

## Parity checks

- **Back end** (both filters fed the same Python feature file): max position difference 1.0e-09 m over 2862 poses.
- **Front end**: first 2882 of 2882 frames give identical features (ids and coordinates to 1e-6); mean features per frame Python 92.2, C++ 92.2.
