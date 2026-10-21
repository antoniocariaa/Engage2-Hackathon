# Error analysis

Model: `lr_rw5_lead_3s` (OOF at the CV-selected iteration; validation from the train-only fit). Baseline: `rule_baseline`.

## lr_rw5_lead_3s / CV OOF

- samples 11923: top-1 0.7900, top-3 0.9013
- misses 1177: near misses (true rank 4–5) 404, rank 6–10 455, deep (>10) 318
- true-rank distribution (share): 1: 0.790, 2: 0.076, 3: 0.035, 4: 0.020, 5: 0.014, 6: 0.011, 7: 0.010, 8: 0.007, 9: 0.005, 10: 0.005, 11: 0.027 (11 = 11+)
- twin_AE samples 3535: top-3 0.9050 vs others 0.8997; twin confusion rate (truth assume/entry_coord, sibling ranked higher) 0.0389 over 3524 samples
- target aircraft has >1 task (l_n_flight>1): n=10207, top-3 0.8971; single-task aircraft top-3 0.9260

**by true_type**

| true_type | n | top1 | top3 |
|---|---|---|---|
| assume | 3003.0 | 0.957 | 0.976 |
| conflict | 351.0 | 0.587 | 0.897 |
| entry_conditions | 2.0 | 1.0 | 1.0 |
| entry_conflict | 31.0 | 0.613 | 0.645 |
| entry_coordination | 940.0 | 0.417 | 0.707 |
| exit_conditions | 407.0 | 0.543 | 0.84 |
| exit_conflict | 186.0 | 0.548 | 0.839 |
| exit_coordination | 704.0 | 0.339 | 0.565 |
| non_conformance | 86.0 | 0.616 | 0.93 |
| qos_improvement | 237.0 | 0.578 | 0.869 |
| return_to_route | 881.0 | 0.758 | 0.843 |
| transfer | 3145.0 | 0.952 | 0.983 |
| zone_conflict | 1950.0 | 0.775 | 0.921 |

**by type_group**

| type_group | n | top1 | top3 |
|---|---|---|---|
| conflict(0,1,6) | 568.0 | 0.576 | 0.864 |
| coord/handover(2,3,4,5) | 7792.0 | 0.834 | 0.909 |
| other | 3563.0 | 0.728 | 0.889 |

**by seq_bucket**

| seq_bucket | n | top1 | top3 |
|---|---|---|---|
| (0, 20] | 773.0 | 0.86 | 0.951 |
| (20, 30] | 2308.0 | 0.795 | 0.906 |
| (30, 40] | 4677.0 | 0.793 | 0.899 |
| (40, 50] | 3514.0 | 0.77 | 0.891 |
| (50, 60] | 651.0 | 0.771 | 0.897 |

**by scenario**

| scenario | n | top1 | top3 |
|---|---|---|---|
| 1 | 3771.0 | 0.766 | 0.874 |
| 2 | 4103.0 | 0.796 | 0.916 |
| 3 | 4049.0 | 0.806 | 0.913 |

**by atco**

| atco | n | top1 | top3 |
|---|---|---|---|
| 1 | 338.0 | 0.68 | 0.743 |
| 2 | 587.0 | 0.761 | 0.886 |
| 3 | 629.0 | 0.782 | 0.928 |
| 4 | 562.0 | 0.742 | 0.87 |
| 5 | 1031.0 | 0.779 | 0.931 |
| 6 | 374.0 | 0.789 | 0.925 |
| 7 | 497.0 | 0.819 | 0.936 |
| 8 | 555.0 | 0.732 | 0.836 |
| 9 | 703.0 | 0.853 | 0.959 |
| 10 | 350.0 | 0.649 | 0.771 |
| 11 | 956.0 | 0.851 | 0.957 |
| 14 | 611.0 | 0.856 | 0.931 |
| 15 | 1228.0 | 0.828 | 0.931 |
| 16 | 1066.0 | 0.697 | 0.826 |
| 17 | 1061.0 | 0.802 | 0.884 |
| 18 | 1375.0 | 0.834 | 0.929 |

**by session**

| session | n | top1 | top3 |
|---|---|---|---|
| 001sc1 | 338.0 | 0.68 | 0.743 |
| 002sc2 | 296.0 | 0.784 | 0.885 |
| 002sc3 | 291.0 | 0.739 | 0.887 |
| 003sc1 | 264.0 | 0.811 | 0.958 |
| 003sc3 | 365.0 | 0.762 | 0.907 |
| 004sc1 | 296.0 | 0.73 | 0.831 |
| 004sc2 | 266.0 | 0.756 | 0.914 |
| 005sc1 | 310.0 | 0.752 | 0.9 |
| 005sc2 | 366.0 | 0.784 | 0.945 |
| 005sc3 | 355.0 | 0.797 | 0.944 |
| 006sc2 | 374.0 | 0.789 | 0.925 |
| 007sc1 | 271.0 | 0.804 | 0.93 |
| 007sc3 | 226.0 | 0.836 | 0.942 |
| 008sc1 | 286.0 | 0.815 | 0.888 |
| 008sc2 | 269.0 | 0.643 | 0.781 |
| 009sc1 | 310.0 | 0.835 | 0.965 |
| 009sc3 | 393.0 | 0.868 | 0.954 |
| 010sc3 | 350.0 | 0.649 | 0.771 |
| 011sc2 | 544.0 | 0.825 | 0.95 |
| 011sc3 | 412.0 | 0.886 | 0.966 |
| 014sc1 | 262.0 | 0.908 | 0.931 |
| 014sc2 | 349.0 | 0.817 | 0.931 |
| 015sc1 | 366.0 | 0.79 | 0.888 |
| 015sc2 | 411.0 | 0.876 | 0.976 |
| 015sc3 | 451.0 | 0.816 | 0.925 |
| 016sc1 | 375.0 | 0.581 | 0.744 |
| 016sc2 | 384.0 | 0.716 | 0.844 |
| 016sc3 | 307.0 | 0.814 | 0.906 |
| 017sc1 | 315.0 | 0.822 | 0.917 |
| 017sc2 | 353.0 | 0.773 | 0.847 |
| 017sc3 | 393.0 | 0.812 | 0.891 |
| 018sc1 | 378.0 | 0.746 | 0.854 |
| 018sc2 | 491.0 | 0.892 | 0.986 |
| 018sc3 | 506.0 | 0.844 | 0.929 |

**predicted top-1 type (columns) vs true type (rows)**

| true_type | assume | conflict | entry_conditions | entry_conflict | entry_coordination | exit_conditions | exit_conflict | exit_coordination | non_conformance | qos_improvement | return_to_route | transfer | zone_conflict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| assume | 2896 | 7 | 0 | 0 | 20 | 0 | 9 | 10 | 0 | 1 | 10 | 20 | 30 |
| conflict | 17 | 232 | 0 | 0 | 0 | 7 | 4 | 7 | 0 | 1 | 0 | 21 | 62 |
| entry_conditions | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| entry_conflict | 0 | 0 | 0 | 20 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| entry_coordination | 106 | 0 | 0 | 0 | 548 | 3 | 1 | 64 | 2 | 5 | 45 | 44 | 122 |
| exit_conditions | 10 | 1 | 0 | 0 | 18 | 223 | 1 | 20 | 5 | 19 | 37 | 46 | 27 |
| exit_conflict | 11 | 11 | 0 | 0 | 9 | 0 | 105 | 1 | 0 | 1 | 5 | 7 | 36 |
| exit_coordination | 29 | 6 | 0 | 0 | 140 | 23 | 8 | 280 | 2 | 7 | 91 | 56 | 62 |
| non_conformance | 5 | 0 | 0 | 0 | 6 | 1 | 0 | 0 | 53 | 7 | 7 | 2 | 5 |
| qos_improvement | 10 | 0 | 0 | 0 | 3 | 33 | 0 | 4 | 1 | 137 | 39 | 8 | 2 |
| return_to_route | 41 | 10 | 0 | 0 | 28 | 11 | 0 | 24 | 9 | 6 | 677 | 45 | 30 |
| transfer | 12 | 8 | 0 | 0 | 19 | 8 | 2 | 20 | 0 | 2 | 14 | 3021 | 39 |
| zone_conflict | 86 | 46 | 0 | 0 | 60 | 14 | 14 | 20 | 9 | 13 | 61 | 60 | 1567 |

## lr_rw5_lead_3s / validation

- samples 1252: top-1 0.7931, top-3 0.9289
- misses 89: near misses (true rank 4–5) 38, rank 6–10 26, deep (>10) 25
- true-rank distribution (share): 1: 0.793, 2: 0.097, 3: 0.039, 4: 0.018, 5: 0.012, 6: 0.005, 7: 0.003, 8: 0.004, 9: 0.002, 10: 0.007, 11: 0.020 (11 = 11+)
- twin_AE samples 405: top-3 0.8790 vs others 0.9528; twin confusion rate (truth assume/entry_coord, sibling ranked higher) 0.0346 over 405 samples
- target aircraft has >1 task (l_n_flight>1): n=1071, top-3 0.9290; single-task aircraft top-3 0.9282

**by true_type**

| true_type | n | top1 | top3 |
|---|---|---|---|
| assume | 363.0 | 0.906 | 0.986 |
| conflict | 39.0 | 0.667 | 0.923 |
| entry_coordination | 128.0 | 0.273 | 0.656 |
| exit_conditions | 41.0 | 0.512 | 0.951 |
| exit_conflict | 11.0 | 0.818 | 1.0 |
| exit_coordination | 41.0 | 0.463 | 0.61 |
| qos_improvement | 20.0 | 0.45 | 0.8 |
| return_to_route | 56.0 | 0.857 | 1.0 |
| transfer | 382.0 | 0.937 | 0.971 |
| zone_conflict | 171.0 | 0.813 | 0.977 |

**by type_group**

| type_group | n | top1 | top3 |
|---|---|---|---|
| conflict(0,1,6) | 50.0 | 0.7 | 0.94 |
| coord/handover(2,3,4,5) | 914.0 | 0.811 | 0.917 |
| other | 288.0 | 0.753 | 0.965 |

**by seq_bucket**

| seq_bucket | n | top1 | top3 |
|---|---|---|---|
| (0, 20] | 121.0 | 0.777 | 1.0 |
| (20, 30] | 244.0 | 0.799 | 0.926 |
| (30, 40] | 601.0 | 0.799 | 0.917 |
| (40, 50] | 286.0 | 0.783 | 0.927 |

**by scenario**

| scenario | n | top1 | top3 |
|---|---|---|---|
| 1 | 777.0 | 0.761 | 0.923 |
| 2 | 475.0 | 0.846 | 0.939 |

**by atco**

| atco | n | top1 | top3 |
|---|---|---|---|
| 19 | 853.0 | 0.807 | 0.937 |
| 20 | 399.0 | 0.764 | 0.912 |

**by session**

| session | n | top1 | top3 |
|---|---|---|---|
| 019sc1 | 378.0 | 0.757 | 0.934 |
| 019sc2 | 475.0 | 0.846 | 0.939 |
| 020sc1 | 399.0 | 0.764 | 0.912 |

**predicted top-1 type (columns) vs true type (rows)**

| true_type | assume | conflict | entry_coordination | exit_conditions | exit_conflict | exit_coordination | non_conformance | qos_improvement | return_to_route | transfer | zone_conflict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assume | 333 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 26 |
| conflict | 0 | 30 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 3 |
| entry_coordination | 11 | 3 | 54 | 0 | 0 | 24 | 0 | 0 | 11 | 7 | 18 |
| exit_conditions | 0 | 0 | 0 | 21 | 0 | 2 | 0 | 18 | 0 | 0 | 0 |
| exit_conflict | 0 | 0 | 0 | 0 | 9 | 2 | 0 | 0 | 0 | 0 | 0 |
| exit_coordination | 0 | 4 | 12 | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 6 |
| qos_improvement | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 4 | 1 |
| return_to_route | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 48 | 1 | 4 |
| transfer | 4 | 0 | 1 | 2 | 0 | 8 | 6 | 1 | 0 | 358 | 2 |
| zone_conflict | 1 | 10 | 11 | 0 | 0 | 0 | 0 | 0 | 6 | 3 | 140 |

## rule_baseline / CV OOF

- samples 11923: top-1 0.2018, top-3 0.3560
- misses 7679: near misses (true rank 4–5) 1451, rank 6–10 2583, deep (>10) 3645
- true-rank distribution (share): 1: 0.202, 2: 0.086, 3: 0.069, 4: 0.063, 5: 0.059, 6: 0.052, 7: 0.043, 8: 0.042, 9: 0.043, 10: 0.036, 11: 0.306 (11 = 11+)
- twin_AE samples 3535: top-3 0.1001 vs others 0.4638; twin confusion rate (truth assume/entry_coord, sibling ranked higher) 0.2622 over 3524 samples
- target aircraft has >1 task (l_n_flight>1): n=10207, top-3 0.3710; single-task aircraft top-3 0.2663

**by true_type**

| true_type | n | top1 | top3 |
|---|---|---|---|
| assume | 3003.0 | 0.015 | 0.139 |
| conflict | 351.0 | 0.0 | 0.0 |
| entry_conditions | 2.0 | 0.0 | 0.0 |
| entry_conflict | 31.0 | 0.0 | 0.0 |
| entry_coordination | 940.0 | 0.0 | 0.0 |
| exit_conditions | 407.0 | 0.0 | 0.0 |
| exit_conflict | 186.0 | 0.0 | 0.0 |
| exit_coordination | 704.0 | 0.0 | 0.0 |
| non_conformance | 86.0 | 0.023 | 0.605 |
| qos_improvement | 237.0 | 0.0 | 0.0 |
| return_to_route | 881.0 | 0.06 | 0.451 |
| transfer | 3145.0 | 0.723 | 0.981 |
| zone_conflict | 1950.0 | 0.016 | 0.151 |

**by type_group**

| type_group | n | top1 | top3 |
|---|---|---|---|
| conflict(0,1,6) | 568.0 | 0.0 | 0.0 |
| coord/handover(2,3,4,5) | 7792.0 | 0.298 | 0.449 |
| other | 3563.0 | 0.024 | 0.209 |

**by seq_bucket**

| seq_bucket | n | top1 | top3 |
|---|---|---|---|
| (0, 20] | 773.0 | 0.287 | 0.467 |
| (20, 30] | 2308.0 | 0.217 | 0.424 |
| (30, 40] | 4677.0 | 0.196 | 0.357 |
| (40, 50] | 3514.0 | 0.184 | 0.306 |
| (50, 60] | 651.0 | 0.18 | 0.249 |

**by scenario**

| scenario | n | top1 | top3 |
|---|---|---|---|
| 1 | 3771.0 | 0.201 | 0.369 |
| 2 | 4103.0 | 0.21 | 0.361 |
| 3 | 4049.0 | 0.194 | 0.339 |

**by atco**

| atco | n | top1 | top3 |
|---|---|---|---|
| 1 | 338.0 | 0.225 | 0.322 |
| 2 | 587.0 | 0.199 | 0.376 |
| 3 | 629.0 | 0.207 | 0.38 |
| 4 | 562.0 | 0.219 | 0.342 |
| 5 | 1031.0 | 0.162 | 0.326 |
| 6 | 374.0 | 0.219 | 0.369 |
| 7 | 497.0 | 0.231 | 0.449 |
| 8 | 555.0 | 0.222 | 0.42 |
| 9 | 703.0 | 0.185 | 0.373 |
| 10 | 350.0 | 0.274 | 0.337 |
| 11 | 956.0 | 0.185 | 0.329 |
| 14 | 611.0 | 0.259 | 0.385 |
| 15 | 1228.0 | 0.182 | 0.31 |
| 16 | 1066.0 | 0.171 | 0.294 |
| 17 | 1061.0 | 0.188 | 0.338 |
| 18 | 1375.0 | 0.224 | 0.415 |

**by session**

| session | n | top1 | top3 |
|---|---|---|---|
| 001sc1 | 338.0 | 0.225 | 0.322 |
| 002sc2 | 296.0 | 0.226 | 0.399 |
| 002sc3 | 291.0 | 0.172 | 0.354 |
| 003sc1 | 264.0 | 0.208 | 0.386 |
| 003sc3 | 365.0 | 0.205 | 0.375 |
| 004sc1 | 296.0 | 0.189 | 0.301 |
| 004sc2 | 266.0 | 0.252 | 0.387 |
| 005sc1 | 310.0 | 0.148 | 0.361 |
| 005sc2 | 366.0 | 0.186 | 0.32 |
| 005sc3 | 355.0 | 0.149 | 0.301 |
| 006sc2 | 374.0 | 0.219 | 0.369 |
| 007sc1 | 271.0 | 0.181 | 0.428 |
| 007sc3 | 226.0 | 0.292 | 0.473 |
| 008sc1 | 286.0 | 0.213 | 0.458 |
| 008sc2 | 269.0 | 0.23 | 0.379 |
| 009sc1 | 310.0 | 0.2 | 0.403 |
| 009sc3 | 393.0 | 0.173 | 0.349 |
| 010sc3 | 350.0 | 0.274 | 0.337 |
| 011sc2 | 544.0 | 0.176 | 0.276 |
| 011sc3 | 412.0 | 0.197 | 0.4 |
| 014sc1 | 262.0 | 0.221 | 0.351 |
| 014sc2 | 349.0 | 0.287 | 0.41 |
| 015sc1 | 366.0 | 0.175 | 0.325 |
| 015sc2 | 411.0 | 0.212 | 0.336 |
| 015sc3 | 451.0 | 0.16 | 0.275 |
| 016sc1 | 375.0 | 0.187 | 0.288 |
| 016sc2 | 384.0 | 0.143 | 0.305 |
| 016sc3 | 307.0 | 0.186 | 0.287 |
| 017sc1 | 315.0 | 0.197 | 0.378 |
| 017sc2 | 353.0 | 0.221 | 0.408 |
| 017sc3 | 393.0 | 0.15 | 0.244 |
| 018sc1 | 378.0 | 0.265 | 0.45 |
| 018sc2 | 491.0 | 0.204 | 0.43 |
| 018sc3 | 506.0 | 0.213 | 0.374 |

**predicted top-1 type (columns) vs true type (rows)**

| true_type | assume | non_conformance | return_to_route | transfer | zone_conflict |
|---|---|---|---|---|---|
| assume | 98 | 24 | 195 | 2483 | 203 |
| conflict | 7 | 0 | 0 | 344 | 0 |
| entry_conditions | 0 | 0 | 0 | 2 | 0 |
| entry_conflict | 0 | 0 | 0 | 31 | 0 |
| entry_coordination | 13 | 0 | 106 | 766 | 55 |
| exit_conditions | 6 | 4 | 14 | 374 | 9 |
| exit_conflict | 16 | 0 | 13 | 151 | 6 |
| exit_coordination | 8 | 19 | 14 | 640 | 23 |
| non_conformance | 0 | 2 | 4 | 80 | 0 |
| qos_improvement | 4 | 11 | 3 | 193 | 26 |
| return_to_route | 0 | 0 | 90 | 791 | 0 |
| transfer | 0 | 0 | 0 | 3145 | 0 |
| zone_conflict | 22 | 8 | 129 | 1740 | 51 |

## rule_baseline / validation

- samples 1252: top-1 0.2412, top-3 0.4848
- misses 645: near misses (true rank 4–5) 126, rank 6–10 168, deep (>10) 351
- true-rank distribution (share): 1: 0.241, 2: 0.147, 3: 0.097, 4: 0.057, 5: 0.044, 6: 0.046, 7: 0.029, 8: 0.015, 9: 0.023, 10: 0.022, 11: 0.280 (11 = 11+)
- twin_AE samples 405: top-3 0.3111 vs others 0.5679; twin confusion rate (truth assume/entry_coord, sibling ranked higher) 0.3160 over 405 samples
- target aircraft has >1 task (l_n_flight>1): n=1071, top-3 0.5238; single-task aircraft top-3 0.2541

**by true_type**

| true_type | n | top1 | top3 |
|---|---|---|---|
| assume | 363.0 | 0.061 | 0.493 |
| conflict | 39.0 | 0.0 | 0.0 |
| entry_coordination | 128.0 | 0.0 | 0.0 |
| exit_conditions | 41.0 | 0.0 | 0.0 |
| exit_conflict | 11.0 | 0.0 | 0.0 |
| exit_coordination | 41.0 | 0.0 | 0.0 |
| qos_improvement | 20.0 | 0.0 | 0.0 |
| return_to_route | 56.0 | 0.0 | 0.75 |
| transfer | 382.0 | 0.733 | 0.99 |
| zone_conflict | 171.0 | 0.0 | 0.047 |

**by type_group**

| type_group | n | top1 | top3 |
|---|---|---|---|
| conflict(0,1,6) | 50.0 | 0.0 | 0.0 |
| coord/handover(2,3,4,5) | 914.0 | 0.33 | 0.609 |
| other | 288.0 | 0.0 | 0.174 |

**by seq_bucket**

| seq_bucket | n | top1 | top3 |
|---|---|---|---|
| (0, 20] | 121.0 | 0.347 | 0.843 |
| (20, 30] | 244.0 | 0.246 | 0.553 |
| (30, 40] | 601.0 | 0.226 | 0.441 |
| (40, 50] | 286.0 | 0.224 | 0.367 |

**by scenario**

| scenario | n | top1 | top3 |
|---|---|---|---|
| 1 | 777.0 | 0.238 | 0.501 |
| 2 | 475.0 | 0.246 | 0.459 |

**by atco**

| atco | n | top1 | top3 |
|---|---|---|---|
| 19 | 853.0 | 0.239 | 0.46 |
| 20 | 399.0 | 0.246 | 0.539 |

**by session**

| session | n | top1 | top3 |
|---|---|---|---|
| 019sc1 | 378.0 | 0.23 | 0.46 |
| 019sc2 | 475.0 | 0.246 | 0.459 |
| 020sc1 | 399.0 | 0.246 | 0.539 |

**predicted top-1 type (columns) vs true type (rows)**

| true_type | assume | non_conformance | return_to_route | transfer |
|---|---|---|---|---|
| assume | 39 | 3 | 39 | 282 |
| conflict | 0 | 0 | 0 | 39 |
| entry_coordination | 8 | 1 | 0 | 119 |
| exit_conditions | 2 | 0 | 4 | 35 |
| exit_conflict | 0 | 0 | 4 | 7 |
| exit_coordination | 5 | 3 | 0 | 33 |
| qos_improvement | 5 | 6 | 0 | 9 |
| return_to_route | 0 | 0 | 0 | 56 |
| transfer | 0 | 0 | 0 | 382 |
| zone_conflict | 6 | 0 | 4 | 161 |

## Feature importance (gain share, `lr_rw5_lead_3s` full-train fit)

|  | gain_share |
|---|---|
| L_last_minus_b_end | 0.3227 |
| L_next_input_minus_b_end | 0.1701 |
| L_last_minus_T | 0.1128 |
| task_index | 0.108 |
| L_next_input_minus_b_start | 0.0496 |
| k_distanceToExitPoint | 0.0159 |
| L_first_minus_b_start | 0.0133 |
| is_assumed | 0.0132 |
| x_task_age | 0.0127 |
| distance_to_zone | 0.0117 |
| distance_to_sector | 0.0088 |
| time_to_sector | 0.0081 |
| l_frac_type | 0.0077 |
| k_timeToExitPoint | 0.0076 |
| L_gone_minus_b_end | 0.0076 |
| L_gap_to_next_block | 0.0074 |
| l_n_flight | 0.0061 |
| k_headingClearanceAgeS | 0.0059 |
| a_since_any | 0.0059 |
| k_isClearedHeadingAssigned | 0.0057 |
| non_conformance_agreed_route_deviation_horizontally | 0.0056 |
| a_since_flight | 0.0054 |
| k_isClimbing | 0.0047 |
| L_input_in_block | 0.0045 |
| x_state_age | 0.0045 |
| k_outCoordRevision | 0.0038 |
| l_rank_absf1_type | 0.0037 |
| k_levelClearanceAgeS | 0.0037 |
| L_b_n | 0.0036 |
| l_rank_age_type | 0.0033 |

