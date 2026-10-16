# Python vs C++ stereo MSCKF on EuRoC MH_01_easy

Start offset 40 s, deterministic replay (no threads), same compute node.

| | Python | C++ |
|---|---|---|
| ATE RMSE [m] | 0.1289 | 0.1284 |
| RPE @ 10 m [% / deg] | 1.48 / 1.41 | 1.49 / 1.39 |
| frontend latency mean / p95 [ms] | 15.95 / 17.48 | 6.34 / 8.71 |
| backend latency mean / p95 [ms] | 18.59 / 44.02 | 2.29 / 5.06 |
| total latency mean / p95 [ms] | 34.54 / 60.22 | 8.63 / 12.32 |
| real-time factor | 1.45x | 5.80x |
| speed-up (total mean) | | 4.0x |

## Parity checks

- Back end (both filters fed the same Python feature file): max position difference 1.0e-09 m over 2862 poses.
- Front end: the first 2882 of 2882 frames give identical features (ids, and coordinates to 1e-06); mean features per frame Python 92.2, C++ 92.2.
