# Python vs C++ stereo MSCKF on EuRoC MH_01_easy

Start offset 40 s, deterministic replay (no threads). Timing: one core, same compute node (compute-3-03).

| | Python | C++ |
|---|---|---|
| ATE RMSE [m] | 0.1289 | 0.1284 |
| RPE @ 10 m [% / deg] | 1.48 / 1.41 | 1.49 / 1.39 |
| frontend latency mean / p95 [ms] | 13.03 / 14.19 | 6.20 / 7.00 |
| backend latency mean / p95 [ms] | 12.13 / 28.57 | 1.86 / 4.08 |
| total latency mean / p95 [ms] | 25.16 / 41.75 | 8.06 / 10.34 |
| real-time factor | 1.99x | 6.20x |
| speed-up (total mean) | | 3.1x |

## Parity checks

- Back end (both filters fed the same Python feature file): max position difference 1.0e-09 m over 2862 poses.
- Front end: the first 2882 of 2882 frames give identical features (ids, and coordinates to 1e-06); mean features per frame Python 92.2, C++ 92.2.
