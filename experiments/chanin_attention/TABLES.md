# Completed attention experiments

Three head seeds where n=3; values are mean [minimum, maximum] across seeds.
MSE is divided by mean-predictor MSE. The oracle error is E[||prediction − conditional mean||²] on held-out inputs, on the same scale.

| Architecture | Data | QK | n | MSE | Bayes MSE | Oracle error |
|---|---|---|---:|---:|---:|---:|
| content only | chanin_iid | full | 3 | 1.00006 [1.00004, 1.00008] | 1.00000 | 0.00005 [0.00004, 0.00005] |
| content only | chanin_iid | zero | 3 | 1.00006 [1.00003, 1.00009] | 1.00000 | 0.00005 [0.00004, 0.00006] |
| content only | independent_iid | full | 3 | 1.00005 [1.00004, 1.00005] | 1.00000 | 0.00006 [0.00004, 0.00007] |
| content only | independent_iid | zero | 3 | 1.00005 [1.00004, 1.00005] | 1.00000 | 0.00006 [0.00005, 0.00007] |
| content only | markov rho=0.7, hit=0.625 | full | 3 | 0.85542 [0.85541, 0.85545] | 0.82864 | 0.02695 [0.02693, 0.02696] |
| content only | markov rho=0.7, hit=0.625 | zero | 3 | 0.92469 [0.92467, 0.92470] | 0.82864 | 0.09614 [0.09613, 0.09614] |
| content only | markov rho=0.7, hit=1.0 | full | 3 | 0.54109 [0.54104, 0.54113] | 0.52607 | 0.01508 [0.01505, 0.01510] |
| content only | markov rho=0.7, hit=1.0 | zero | 3 | 0.81781 [0.81769, 0.81791] | 0.52607 | 0.29201 [0.29189, 0.29212] |
| content only | markov rho=0.7, hit=0.625 | diagonal | 3 | 0.85595 [0.85591, 0.85598] | 0.82864 | 0.02749 [0.02747, 0.02750] |
| content only | markov rho=0.7, hit=1.0 | diagonal | 3 | 0.54694 [0.54688, 0.54698] | 0.52607 | 0.02087 [0.02085, 0.02089] |
| content only | markov rho=0.3, hit=1.0 | full | 3 | 0.91648 [0.91642, 0.91653] | 0.91323 | 0.00332 [0.00330, 0.00335] |
| content only | markov rho=0.9, hit=1.0 | full | 3 | 0.23859 [0.23854, 0.23862] | 0.21510 | 0.02336 [0.02329, 0.02340] |
| relative position | chanin_iid | full | 3 | 1.00006 [1.00004, 1.00008] | 1.00000 | 0.00005 [0.00004, 0.00005] |
| relative position | chanin_iid | zero | 3 | 1.00006 [1.00003, 1.00009] | 1.00000 | 0.00005 [0.00004, 0.00006] |
| relative position | independent_iid | full | 3 | 1.00005 [1.00004, 1.00005] | 1.00000 | 0.00006 [0.00005, 0.00007] |
| relative position | independent_iid | zero | 3 | 1.00005 [1.00004, 1.00005] | 1.00000 | 0.00006 [0.00005, 0.00007] |
| relative position | markov rho=0.7, hit=0.625 | full | 3 | 0.84255 [0.84250, 0.84258] | 0.82864 | 0.01379 [0.01378, 0.01380] |
| relative position | markov rho=0.7, hit=0.625 | zero | 3 | 0.84299 [0.84296, 0.84301] | 0.82864 | 0.01427 [0.01426, 0.01428] |
| relative position | markov rho=0.7, hit=1.0 | full | 3 | 0.54088 [0.54082, 0.54091] | 0.52607 | 0.01488 [0.01485, 0.01491] |
| relative position | markov rho=0.7, hit=1.0 | zero | 3 | 0.54090 [0.54084, 0.54094] | 0.52607 | 0.01488 [0.01485, 0.01490] |

## Full-QK interventions

Weights and OV are held fixed during interventions. Confidence intervals inside each cell's metrics.json use independent sequences.

| Architecture | Data | Remove off-diagonal: ΔMSE | Remove all QK: ΔMSE | Replace off-diagonal by its mean: MSE |
|---|---|---:|---:|---:|
| content only | chanin_iid, rho=0.0, hit=1.0 | 0.00001 [0.00000, 0.00001] | 0.00001 [0.00000, 0.00001] | 1.00006 [1.00004, 1.00008] |
| content only | independent_iid, rho=0.0, hit=1.0 | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] | 1.00005 [1.00004, 1.00005] |
| content only | markov, rho=0.7, hit=0.625 | 0.00107 [0.00103, 0.00113] | 0.06913 [0.06911, 0.06915] | 0.85533 [0.85529, 0.85534] |
| content only | markov, rho=0.7, hit=1.0 | 0.00566 [0.00562, 0.00570] | 0.27928 [0.27926, 0.27931] | 0.54124 [0.54118, 0.54128] |
| content only | markov, rho=0.3, hit=1.0 | 0.00027 [0.00026, 0.00027] | 0.06294 [0.06292, 0.06295] | 0.91649 [0.91644, 0.91654] |
| content only | markov, rho=0.9, hit=1.0 | 0.01945 [0.01941, 0.01952] | 0.24495 [0.24492, 0.24499] | 0.23892 [0.23885, 0.23896] |
| relative position | chanin_iid, rho=0.0, hit=1.0 | 0.00001 [0.00000, 0.00001] | 0.00001 [0.00000, 0.00001] | 1.00006 [1.00004, 1.00008] |
| relative position | independent_iid, rho=0.0, hit=1.0 | 0.00000 [0.00000, 0.00000] | 0.00000 [-0.00000, 0.00001] | 1.00005 [1.00004, 1.00005] |
| relative position | markov, rho=0.7, hit=0.625 | 0.00028 [0.00026, 0.00030] | 0.00048 [0.00046, 0.00051] | 0.84253 [0.84251, 0.84256] |
| relative position | markov, rho=0.7, hit=1.0 | 0.00096 [0.00096, 0.00097] | 0.16639 [0.16524, 0.16783] | 0.54090 [0.54084, 0.54093] |
