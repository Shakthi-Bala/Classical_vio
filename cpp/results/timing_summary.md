Per-frame latency on MH_01, 5 runs on compute-3-03, one core (mean ± standard deviation across runs).

| | mean [ms] | p95 [ms] | real-time factor |
|---|---|---|---|
| Python MSCKF | 25.17 ± 0.16 | 41.78 ± 0.33 | 1.99 ± 0.01 |
| C++ MSCKF, Python behaviour | 8.06 ± 0.01 | 10.34 ± 0.00 | 6.20 ± 0.00 |
| C++ MSCKF | 8.41 ± 0.01 | 11.17 ± 0.02 | 5.94 ± 0.01 |
| C++ sliding window | 17.41 ± 0.06 | 35.28 ± 0.14 | 2.87 ± 0.01 |
