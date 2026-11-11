# Generated result tables

All tables are written by `python -m hnl.cli reproduce`; numbers are reproducible from the committed seeds and manifest.

## 1. Random hypergraphs: cell summaries

| latency | q | n | P(consensus) | KM median T_c | RMST(T_c) | final D | O | max C |
|---|---|---|---|---|---|---|---|---|
| exponential | 0.50 | 30 | 0.000 | — | 100.000 | 0.462 | 0.068 | 0.335 |
| exponential | 1.00 | 30 | 0.067 | ∞ | 97.844 | 0.407 | 0.089 | 0.316 |
| exponential | 1.50 | 30 | 0.500 | ∞ | 75.070 | 0.306 | 0.124 | 0.296 |
| exponential | 2.50 | 30 | 1.000 | 21.000 | 26.048 | 0.180 | 0.110 | 0.244 |
| fixed | 0.50 | 30 | 0.000 | — | 100.000 | 0.408 | 0.090 | 0.428 |
| fixed | 1.00 | 30 | 0.233 | ∞ | 91.444 | 0.368 | 0.111 | 0.391 |
| fixed | 1.50 | 30 | 0.500 | ∞ | 77.997 | 0.296 | 0.120 | 0.377 |
| fixed | 2.50 | 30 | 0.867 | 35.000 | 46.519 | 0.213 | 0.137 | 0.348 |
| normal_s1 | 0.50 | 30 | 0.000 | — | 100.000 | 0.448 | 0.077 | 0.276 |
| normal_s1 | 1.00 | 30 | 0.100 | ∞ | 97.150 | 0.372 | 0.102 | 0.265 |
| normal_s1 | 1.50 | 30 | 0.400 | ∞ | 79.976 | 0.306 | 0.126 | 0.251 |
| normal_s1 | 2.50 | 30 | 0.967 | 41.000 | 40.074 | 0.187 | 0.133 | 0.235 |
| normal_s2 | 0.50 | 30 | 0.000 | — | 100.000 | 0.445 | 0.072 | 0.272 |
| normal_s2 | 1.00 | 30 | 0.067 | ∞ | 95.430 | 0.401 | 0.099 | 0.265 |
| normal_s2 | 1.50 | 30 | 0.600 | 93.000 | 78.503 | 0.261 | 0.126 | 0.247 |
| normal_s2 | 2.50 | 30 | 0.900 | 32.000 | 41.853 | 0.212 | 0.138 | 0.219 |
| normal_s4 | 0.50 | 30 | 0.000 | — | 100.000 | 0.432 | 0.073 | 0.300 |
| normal_s4 | 1.00 | 30 | 0.133 | ∞ | 94.648 | 0.383 | 0.090 | 0.289 |
| normal_s4 | 1.50 | 30 | 0.533 | 93.000 | 69.759 | 0.282 | 0.127 | 0.262 |
| normal_s4 | 2.50 | 30 | 0.967 | 35.000 | 36.951 | 0.193 | 0.132 | 0.231 |

## 2. Paired contrasts across q

| latency | metric | level vs q=0.5 | n | mean diff | 95% CI low | 95% CI high | p (Holm) | Cliff δ |
|---|---|---|---|---|---|---|---|---|
| exponential | final_diversity | 1.00 | 30 | -0.055 | -0.087 | -0.025 | 0.002 | -0.557 |
| exponential | final_diversity | 1.50 | 30 | -0.156 | -0.206 | -0.109 | 0.000 | -0.694 |
| exponential | final_diversity | 2.50 | 30 | -0.282 | -0.298 | -0.265 | 0.000 | -1.000 |
| fixed | final_diversity | 1.00 | 30 | -0.040 | -0.087 | 0.006 | 0.110 | -0.064 |
| fixed | final_diversity | 1.50 | 30 | -0.112 | -0.152 | -0.069 | 0.001 | -0.470 |
| fixed | final_diversity | 2.50 | 30 | -0.195 | -0.222 | -0.166 | 0.000 | -0.888 |
| normal_s1 | final_diversity | 1.00 | 30 | -0.076 | -0.114 | -0.036 | 0.001 | -0.449 |
| normal_s1 | final_diversity | 1.50 | 30 | -0.142 | -0.182 | -0.102 | 0.000 | -0.714 |
| normal_s1 | final_diversity | 2.50 | 30 | -0.261 | -0.281 | -0.241 | 0.000 | -1.000 |
| normal_s2 | final_diversity | 1.00 | 30 | -0.044 | -0.075 | -0.016 | 0.004 | -0.353 |
| normal_s2 | final_diversity | 1.50 | 30 | -0.184 | -0.220 | -0.145 | 0.000 | -0.844 |
| normal_s2 | final_diversity | 2.50 | 30 | -0.233 | -0.262 | -0.196 | 0.000 | -0.897 |
| normal_s4 | final_diversity | 1.00 | 30 | -0.049 | -0.088 | -0.013 | 0.018 | -0.274 |
| normal_s4 | final_diversity | 1.50 | 30 | -0.150 | -0.194 | -0.105 | 0.000 | -0.707 |
| normal_s4 | final_diversity | 2.50 | 30 | -0.239 | -0.258 | -0.219 | 0.000 | -0.990 |
| exponential | converged | 1.00 | 30 | 0.067 | 0.000 | 0.167 | 0.495 | 0.067 |
| exponential | converged | 1.50 | 30 | 0.500 | 0.333 | 0.667 | 0.000 | 0.500 |
| exponential | converged | 2.50 | 30 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| fixed | converged | 1.00 | 30 | 0.233 | 0.100 | 0.400 | 0.015 | 0.233 |
| fixed | converged | 1.50 | 30 | 0.500 | 0.300 | 0.667 | 0.000 | 0.500 |
| fixed | converged | 2.50 | 30 | 0.867 | 0.733 | 0.967 | 0.000 | 0.867 |
| normal_s1 | converged | 1.00 | 30 | 0.100 | 0.000 | 0.200 | 0.259 | 0.100 |
| normal_s1 | converged | 1.50 | 30 | 0.400 | 0.233 | 0.567 | 0.002 | 0.400 |
| normal_s1 | converged | 2.50 | 30 | 0.967 | 0.900 | 1.000 | 0.000 | 0.967 |
| normal_s2 | converged | 1.00 | 30 | 0.067 | 0.000 | 0.167 | 0.505 | 0.067 |
| normal_s2 | converged | 1.50 | 30 | 0.600 | 0.433 | 0.767 | 0.000 | 0.600 |
| normal_s2 | converged | 2.50 | 30 | 0.900 | 0.767 | 1.000 | 0.000 | 0.900 |
| normal_s4 | converged | 1.00 | 30 | 0.133 | 0.033 | 0.267 | 0.121 | 0.133 |
| normal_s4 | converged | 1.50 | 30 | 0.533 | 0.367 | 0.700 | 0.000 | 0.533 |
| normal_s4 | converged | 2.50 | 30 | 0.967 | 0.900 | 1.000 | 0.000 | 0.967 |
| exponential | oscillation_index | 1.00 | 30 | 0.021 | 0.015 | 0.026 | 0.000 | 0.791 |
| exponential | oscillation_index | 1.50 | 30 | 0.056 | 0.048 | 0.062 | 0.000 | 1.000 |
| exponential | oscillation_index | 2.50 | 30 | 0.041 | 0.031 | 0.053 | 0.000 | 0.871 |
| fixed | oscillation_index | 1.00 | 30 | 0.021 | 0.012 | 0.030 | 0.000 | 0.540 |
| fixed | oscillation_index | 1.50 | 30 | 0.030 | 0.018 | 0.041 | 0.000 | 0.667 |
| fixed | oscillation_index | 2.50 | 30 | 0.048 | 0.034 | 0.062 | 0.000 | 0.816 |
| normal_s1 | oscillation_index | 1.00 | 30 | 0.025 | 0.015 | 0.034 | 0.000 | 0.596 |
| normal_s1 | oscillation_index | 1.50 | 30 | 0.048 | 0.037 | 0.061 | 0.000 | 0.873 |
| normal_s1 | oscillation_index | 2.50 | 30 | 0.056 | 0.047 | 0.064 | 0.000 | 0.964 |
| normal_s2 | oscillation_index | 1.00 | 30 | 0.028 | 0.020 | 0.035 | 0.000 | 0.778 |
| normal_s2 | oscillation_index | 1.50 | 30 | 0.054 | 0.045 | 0.064 | 0.000 | 0.987 |
| normal_s2 | oscillation_index | 2.50 | 30 | 0.067 | 0.057 | 0.076 | 0.000 | 0.967 |
| normal_s4 | oscillation_index | 1.00 | 30 | 0.017 | 0.010 | 0.025 | 0.000 | 0.613 |
| normal_s4 | oscillation_index | 1.50 | 30 | 0.054 | 0.044 | 0.064 | 0.000 | 0.956 |
| normal_s4 | oscillation_index | 2.50 | 30 | 0.059 | 0.047 | 0.071 | 0.000 | 0.898 |
| exponential | max_cascade | 1.00 | 30 | -0.019 | -0.030 | -0.009 | 0.002 | -0.464 |
| exponential | max_cascade | 1.50 | 30 | -0.039 | -0.051 | -0.027 | 0.000 | -0.689 |
| exponential | max_cascade | 2.50 | 30 | -0.091 | -0.104 | -0.078 | 0.000 | -0.980 |
| fixed | max_cascade | 1.00 | 30 | -0.037 | -0.061 | -0.013 | 0.008 | -0.411 |
| fixed | max_cascade | 1.50 | 30 | -0.051 | -0.074 | -0.027 | 0.000 | -0.513 |
| fixed | max_cascade | 2.50 | 30 | -0.080 | -0.108 | -0.052 | 0.000 | -0.739 |
| normal_s1 | max_cascade | 1.00 | 30 | -0.011 | -0.021 | -0.002 | 0.036 | -0.320 |
| normal_s1 | max_cascade | 1.50 | 30 | -0.025 | -0.034 | -0.016 | 0.000 | -0.627 |
| normal_s1 | max_cascade | 2.50 | 30 | -0.041 | -0.050 | -0.032 | 0.000 | -0.832 |
| normal_s2 | max_cascade | 1.00 | 30 | -0.007 | -0.015 | 0.000 | 0.092 | -0.211 |
| normal_s2 | max_cascade | 1.50 | 30 | -0.025 | -0.035 | -0.016 | 0.000 | -0.647 |
| normal_s2 | max_cascade | 2.50 | 30 | -0.053 | -0.062 | -0.043 | 0.000 | -0.930 |
| normal_s4 | max_cascade | 1.00 | 30 | -0.011 | -0.022 | 0.000 | 0.056 | -0.239 |
| normal_s4 | max_cascade | 1.50 | 30 | -0.038 | -0.047 | -0.029 | 0.000 | -0.752 |
| normal_s4 | max_cascade | 2.50 | 30 | -0.069 | -0.082 | -0.056 | 0.000 | -0.946 |

## 3. Paired contrasts across latency distributions (reference: fixed)

| q | metric | level | n | mean diff | 95% CI low | 95% CI high | p (Holm) | Cliff δ |
|---|---|---|---|---|---|---|---|---|
| 0.50 | final_diversity | exponential | 30 | 0.054 | 0.029 | 0.076 | 0.001 | 0.627 |
| 0.50 | final_diversity | normal_s1 | 30 | 0.040 | 0.018 | 0.063 | 0.006 | 0.453 |
| 0.50 | final_diversity | normal_s2 | 30 | 0.037 | 0.018 | 0.058 | 0.006 | 0.421 |
| 0.50 | final_diversity | normal_s4 | 30 | 0.024 | 0.001 | 0.046 | 0.048 | 0.284 |
| 1.00 | final_diversity | exponential | 30 | 0.039 | -0.009 | 0.085 | 0.471 | 0.090 |
| 1.00 | final_diversity | normal_s1 | 30 | 0.004 | -0.048 | 0.053 | 1.000 | -0.030 |
| 1.00 | final_diversity | normal_s2 | 30 | 0.034 | -0.017 | 0.089 | 0.710 | 0.028 |
| 1.00 | final_diversity | normal_s4 | 30 | 0.015 | -0.037 | 0.065 | 1.000 | 0.020 |
| 1.50 | final_diversity | exponential | 30 | 0.010 | -0.056 | 0.079 | 1.000 | 0.016 |
| 1.50 | final_diversity | normal_s1 | 30 | 0.010 | -0.045 | 0.065 | 1.000 | 0.088 |
| 1.50 | final_diversity | normal_s2 | 30 | -0.036 | -0.097 | 0.026 | 1.000 | -0.212 |
| 1.50 | final_diversity | normal_s4 | 30 | -0.014 | -0.075 | 0.043 | 1.000 | -0.078 |
| 2.50 | final_diversity | exponential | 30 | -0.033 | -0.062 | -0.009 | 0.036 | -0.267 |
| 2.50 | final_diversity | normal_s1 | 30 | -0.026 | -0.057 | 0.000 | 0.347 | -0.207 |
| 2.50 | final_diversity | normal_s2 | 30 | -0.001 | -0.036 | 0.038 | 0.981 | 0.023 |
| 2.50 | final_diversity | normal_s4 | 30 | -0.020 | -0.052 | 0.011 | 0.502 | -0.087 |
| 0.50 | converged | exponential | 30 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.50 | converged | normal_s1 | 30 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.50 | converged | normal_s2 | 30 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.50 | converged | normal_s4 | 30 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 1.00 | converged | exponential | 30 | -0.167 | -0.333 | 0.000 | 0.502 | -0.167 |
| 1.00 | converged | normal_s1 | 30 | -0.133 | -0.300 | 0.033 | 0.588 | -0.133 |
| 1.00 | converged | normal_s2 | 30 | -0.167 | -0.367 | 0.000 | 0.528 | -0.167 |
| 1.00 | converged | normal_s4 | 30 | -0.100 | -0.300 | 0.067 | 0.588 | -0.100 |
| 1.50 | converged | exponential | 30 | 0.000 | -0.267 | 0.267 | 1.000 | 0.000 |
| 1.50 | converged | normal_s1 | 30 | -0.100 | -0.333 | 0.133 | 1.000 | -0.100 |
| 1.50 | converged | normal_s2 | 30 | 0.100 | -0.167 | 0.367 | 1.000 | 0.100 |
| 1.50 | converged | normal_s4 | 30 | 0.033 | -0.233 | 0.300 | 1.000 | 0.033 |
| 2.50 | converged | exponential | 30 | 0.133 | 0.033 | 0.267 | 0.508 | 0.133 |
| 2.50 | converged | normal_s1 | 30 | 0.100 | -0.033 | 0.233 | 1.000 | 0.100 |
| 2.50 | converged | normal_s2 | 30 | 0.033 | -0.100 | 0.200 | 1.000 | 0.033 |
| 2.50 | converged | normal_s4 | 30 | 0.100 | -0.033 | 0.267 | 1.000 | 0.100 |
| 0.50 | oscillation_index | exponential | 30 | -0.021 | -0.030 | -0.013 | 0.000 | -0.607 |
| 0.50 | oscillation_index | normal_s1 | 30 | -0.012 | -0.022 | -0.003 | 0.023 | -0.311 |
| 0.50 | oscillation_index | normal_s2 | 30 | -0.018 | -0.026 | -0.010 | 0.001 | -0.507 |
| 0.50 | oscillation_index | normal_s4 | 30 | -0.017 | -0.025 | -0.009 | 0.002 | -0.467 |
| 1.00 | oscillation_index | exponential | 30 | -0.022 | -0.030 | -0.014 | 0.000 | -0.662 |
| 1.00 | oscillation_index | normal_s1 | 30 | -0.009 | -0.020 | 0.003 | 0.156 | -0.249 |
| 1.00 | oscillation_index | normal_s2 | 30 | -0.011 | -0.022 | -0.001 | 0.102 | -0.307 |
| 1.00 | oscillation_index | normal_s4 | 30 | -0.021 | -0.030 | -0.011 | 0.001 | -0.549 |
| 1.50 | oscillation_index | exponential | 30 | 0.005 | -0.006 | 0.015 | 0.885 | 0.156 |
| 1.50 | oscillation_index | normal_s1 | 30 | 0.006 | -0.006 | 0.018 | 0.885 | 0.171 |
| 1.50 | oscillation_index | normal_s2 | 30 | 0.007 | -0.002 | 0.016 | 0.694 | 0.169 |
| 1.50 | oscillation_index | normal_s4 | 30 | 0.007 | -0.005 | 0.020 | 0.885 | 0.171 |
| 2.50 | oscillation_index | exponential | 30 | -0.027 | -0.043 | -0.011 | 0.010 | -0.491 |
| 2.50 | oscillation_index | normal_s1 | 30 | -0.004 | -0.016 | 0.006 | 1.000 | -0.147 |
| 2.50 | oscillation_index | normal_s2 | 30 | 0.001 | -0.014 | 0.015 | 1.000 | 0.016 |
| 2.50 | oscillation_index | normal_s4 | 30 | -0.006 | -0.021 | 0.008 | 1.000 | -0.089 |

## 4. Are oscillations transient?

| latency | q | n | O first half | O second half | ratio | diff (second−first) | 95% CI low | 95% CI high | p |
|---|---|---|---|---|---|---|---|---|---|
| exponential | 0.50 | 30 | 0.066 | 0.070 | 1.092 | 0.004 | -0.001 | 0.010 | 0.145 |
| exponential | 1.00 | 30 | 0.083 | 0.092 | 1.169 | 0.010 | -0.002 | 0.022 | 0.118 |
| exponential | 1.50 | 30 | 0.118 | 0.098 | 0.999 | -0.020 | -0.052 | 0.010 | 0.234 |
| exponential | 2.50 | 30 | 0.112 | 0.013 | 0.113 | -0.099 | -0.118 | -0.082 | 0.000 |
| fixed | 0.50 | 30 | 0.079 | 0.096 | 1.332 | 0.017 | 0.004 | 0.029 | 0.017 |
| fixed | 1.00 | 30 | 0.093 | 0.118 | 1.507 | 0.025 | -0.001 | 0.048 | 0.061 |
| fixed | 1.50 | 30 | 0.116 | 0.102 | 1.104 | -0.014 | -0.049 | 0.020 | 0.435 |
| fixed | 2.50 | 30 | 0.151 | 0.052 | 0.427 | -0.099 | -0.131 | -0.065 | 0.000 |
| normal_s1 | 0.50 | 30 | 0.069 | 0.083 | 1.265 | 0.014 | 0.005 | 0.023 | 0.007 |
| normal_s1 | 1.00 | 30 | 0.093 | 0.103 | 1.256 | 0.010 | -0.011 | 0.030 | 0.378 |
| normal_s1 | 1.50 | 30 | 0.118 | 0.107 | 1.060 | -0.011 | -0.042 | 0.019 | 0.514 |
| normal_s1 | 2.50 | 30 | 0.143 | 0.036 | 0.388 | -0.107 | -0.141 | -0.068 | 0.000 |
| normal_s2 | 0.50 | 30 | 0.071 | 0.071 | 1.065 | -0.000 | -0.008 | 0.008 | 0.925 |
| normal_s2 | 1.00 | 30 | 0.098 | 0.091 | 1.040 | -0.007 | -0.027 | 0.011 | 0.522 |
| normal_s2 | 1.50 | 30 | 0.119 | 0.111 | 1.186 | -0.007 | -0.043 | 0.027 | 0.699 |
| normal_s2 | 2.50 | 30 | 0.145 | 0.039 | 0.378 | -0.106 | -0.141 | -0.069 | 0.000 |
| normal_s4 | 0.50 | 30 | 0.073 | 0.071 | 1.018 | -0.002 | -0.010 | 0.005 | 0.655 |
| normal_s4 | 1.00 | 30 | 0.085 | 0.087 | 1.161 | 0.002 | -0.016 | 0.019 | 0.826 |
| normal_s4 | 1.50 | 30 | 0.128 | 0.081 | 0.813 | -0.047 | -0.084 | -0.010 | 0.022 |
| normal_s4 | 2.50 | 30 | 0.140 | 0.031 | 0.273 | -0.108 | -0.140 | -0.075 | 0.000 |

## 5. Dataset hypergraph: cell summaries (random initial opinions)

| scope | latency | q | n | P(consensus) | KM median T_c | final D | components absorbed |
|---|---|---|---|---|---|---|---|
| active | exponential | 0.50 | 30 | 0.000 | — | 0.461 | 0.950 |
| active | exponential | 1.00 | 30 | 0.000 | — | 0.450 | 0.983 |
| active | exponential | 1.50 | 30 | 0.000 | — | 0.449 | 0.986 |
| active | exponential | 2.50 | 30 | 0.000 | — | 0.450 | 0.970 |
| active | fixed | 0.50 | 30 | 0.000 | — | 0.454 | 0.953 |
| active | fixed | 1.00 | 30 | 0.000 | — | 0.463 | 0.982 |
| active | fixed | 1.50 | 30 | 0.000 | — | 0.450 | 0.984 |
| active | fixed | 2.50 | 30 | 0.000 | — | 0.436 | 0.971 |
| active | normal_s1 | 0.50 | 30 | 0.000 | — | 0.456 | 0.953 |
| active | normal_s1 | 1.00 | 30 | 0.000 | — | 0.463 | 0.981 |
| active | normal_s1 | 1.50 | 30 | 0.000 | — | 0.437 | 0.982 |
| active | normal_s1 | 2.50 | 30 | 0.000 | — | 0.433 | 0.975 |
| active | normal_s2 | 0.50 | 30 | 0.000 | — | 0.449 | 0.942 |
| active | normal_s2 | 1.00 | 30 | 0.000 | — | 0.454 | 0.982 |
| active | normal_s2 | 1.50 | 30 | 0.000 | — | 0.437 | 0.984 |
| active | normal_s2 | 2.50 | 30 | 0.000 | — | 0.441 | 0.971 |
| active | normal_s4 | 0.50 | 30 | 0.000 | — | 0.459 | 0.945 |
| active | normal_s4 | 1.00 | 30 | 0.000 | — | 0.454 | 0.979 |
| active | normal_s4 | 1.50 | 30 | 0.000 | — | 0.425 | 0.984 |
| active | normal_s4 | 2.50 | 30 | 0.000 | — | 0.455 | 0.972 |
| full | fixed | 0.50 | 15 | 0.000 | — | 0.465 | 0.986 |
| full | fixed | 1.00 | 15 | 0.000 | — | 0.470 | 0.994 |
| full | fixed | 1.50 | 15 | 0.000 | — | 0.454 | 0.995 |
| full | fixed | 2.50 | 15 | 0.000 | — | 0.466 | 0.992 |
| full | normal_s2 | 0.50 | 15 | 0.000 | — | 0.466 | 0.985 |
| full | normal_s2 | 1.00 | 15 | 0.000 | — | 0.465 | 0.994 |
| full | normal_s2 | 1.50 | 15 | 0.000 | — | 0.461 | 0.995 |
| full | normal_s2 | 2.50 | 15 | 0.000 | — | 0.454 | 0.991 |
| full | normal_s4 | 0.50 | 15 | 0.000 | — | 0.469 | 0.985 |
| full | normal_s4 | 1.00 | 15 | 0.000 | — | 0.460 | 0.993 |
| full | normal_s4 | 1.50 | 15 | 0.000 | — | 0.454 | 0.994 |
| full | normal_s4 | 2.50 | 15 | 0.000 | — | 0.467 | 0.991 |
| lcc | fixed | 0.50 | 15 | 0.067 | ∞ | 0.388 | 0.000 |
| lcc | fixed | 1.00 | 15 | 0.400 | ∞ | 0.304 | 0.000 |
| lcc | fixed | 1.50 | 15 | 0.267 | ∞ | 0.310 | 0.000 |
| lcc | fixed | 2.50 | 15 | 0.333 | ∞ | 0.308 | 0.000 |
| lcc | normal_s2 | 0.50 | 15 | 0.000 | — | 0.433 | 0.000 |
| lcc | normal_s2 | 1.00 | 15 | 0.133 | ∞ | 0.368 | 0.000 |
| lcc | normal_s2 | 1.50 | 15 | 0.400 | ∞ | 0.332 | 0.000 |
| lcc | normal_s2 | 2.50 | 15 | 0.467 | ∞ | 0.266 | 0.000 |
| lcc | normal_s4 | 0.50 | 15 | 0.000 | — | 0.439 | 0.000 |
| lcc | normal_s4 | 1.00 | 15 | 0.267 | ∞ | 0.344 | 0.000 |
| lcc | normal_s4 | 1.50 | 15 | 0.333 | ∞ | 0.316 | 0.000 |
| lcc | normal_s4 | 2.50 | 15 | 0.333 | ∞ | 0.306 | 0.000 |
| active_data_init | fixed | 0.50 | 15 | 0.000 | — | 0.440 | 0.979 |
| active_data_init | fixed | 1.00 | 15 | 0.000 | — | 0.450 | 0.986 |
| active_data_init | fixed | 1.50 | 15 | 0.000 | — | 0.367 | 0.992 |
| active_data_init | fixed | 2.50 | 15 | 0.000 | — | 0.312 | 0.984 |
| active_data_init | normal_s4 | 0.50 | 15 | 0.000 | — | 0.428 | 0.969 |
| active_data_init | normal_s4 | 1.00 | 15 | 0.000 | — | 0.392 | 0.985 |
| active_data_init | normal_s4 | 1.50 | 15 | 0.000 | — | 0.362 | 0.990 |
| active_data_init | normal_s4 | 2.50 | 15 | 0.000 | — | 0.313 | 0.984 |

## 6. Dataset hypergraph: paired q contrasts

| scope | latency | metric | level | n | mean diff | 95% CI low | 95% CI high | p (Holm) | Cliff δ |
|---|---|---|---|---|---|---|---|---|---|
| active | exponential | final_diversity | 1.00 | 30 | -0.011 | -0.031 | 0.010 | 0.661 | -0.158 |
| active | exponential | final_diversity | 1.50 | 30 | -0.012 | -0.029 | 0.006 | 0.661 | -0.253 |
| active | exponential | final_diversity | 2.50 | 30 | -0.011 | -0.028 | 0.007 | 0.661 | -0.239 |
| active | fixed | final_diversity | 1.00 | 30 | 0.009 | -0.006 | 0.025 | 0.582 | 0.133 |
| active | fixed | final_diversity | 1.50 | 30 | -0.003 | -0.024 | 0.016 | 0.747 | 0.016 |
| active | fixed | final_diversity | 2.50 | 30 | -0.018 | -0.040 | 0.002 | 0.373 | -0.193 |
| active | normal_s1 | final_diversity | 1.00 | 30 | 0.008 | -0.008 | 0.024 | 0.347 | 0.109 |
| active | normal_s1 | final_diversity | 1.50 | 30 | -0.019 | -0.043 | 0.003 | 0.204 | -0.180 |
| active | normal_s1 | final_diversity | 2.50 | 30 | -0.023 | -0.046 | -0.004 | 0.155 | -0.274 |
| active | normal_s2 | final_diversity | 1.00 | 30 | 0.006 | -0.014 | 0.024 | 1.000 | 0.116 |
| active | normal_s2 | final_diversity | 1.50 | 30 | -0.011 | -0.034 | 0.012 | 1.000 | -0.027 |
| active | normal_s2 | final_diversity | 2.50 | 30 | -0.008 | -0.029 | 0.014 | 1.000 | -0.078 |
| active | normal_s4 | final_diversity | 1.00 | 30 | -0.005 | -0.024 | 0.011 | 1.000 | 0.038 |
| active | normal_s4 | final_diversity | 1.50 | 30 | -0.034 | -0.059 | -0.011 | 0.040 | -0.298 |
| active | normal_s4 | final_diversity | 2.50 | 30 | -0.004 | -0.018 | 0.009 | 1.000 | -0.031 |
| active | exponential | oscillation_index | 1.00 | 30 | 0.008 | 0.004 | 0.012 | 0.002 | 0.518 |
| active | exponential | oscillation_index | 1.50 | 30 | 0.005 | -0.000 | 0.010 | 0.070 | 0.182 |
| active | exponential | oscillation_index | 2.50 | 30 | -0.010 | -0.013 | -0.007 | 0.000 | -0.667 |
| active | fixed | oscillation_index | 1.00 | 30 | -0.002 | -0.005 | 0.002 | 0.356 | -0.093 |
| active | fixed | oscillation_index | 1.50 | 30 | 0.003 | -0.000 | 0.007 | 0.237 | 0.162 |
| active | fixed | oscillation_index | 2.50 | 30 | -0.010 | -0.015 | -0.005 | 0.001 | -0.647 |
| active | normal_s1 | oscillation_index | 1.00 | 30 | 0.002 | -0.001 | 0.006 | 0.144 | 0.244 |
| active | normal_s1 | oscillation_index | 1.50 | 30 | 0.007 | 0.002 | 0.011 | 0.018 | 0.360 |
| active | normal_s1 | oscillation_index | 2.50 | 30 | -0.009 | -0.013 | -0.005 | 0.000 | -0.713 |
| active | normal_s2 | oscillation_index | 1.00 | 30 | 0.003 | -0.000 | 0.007 | 0.089 | 0.280 |
| active | normal_s2 | oscillation_index | 1.50 | 30 | 0.008 | 0.002 | 0.014 | 0.045 | 0.278 |
| active | normal_s2 | oscillation_index | 2.50 | 30 | -0.008 | -0.012 | -0.004 | 0.003 | -0.589 |
| active | normal_s4 | oscillation_index | 1.00 | 30 | 0.008 | 0.003 | 0.012 | 0.003 | 0.427 |
| active | normal_s4 | oscillation_index | 1.50 | 30 | 0.010 | 0.006 | 0.015 | 0.000 | 0.622 |
| active | normal_s4 | oscillation_index | 2.50 | 30 | -0.008 | -0.012 | -0.005 | 0.000 | -0.633 |
| full | fixed | final_diversity | 1.00 | 15 | 0.004 | -0.011 | 0.020 | 1.000 | 0.089 |
| full | fixed | final_diversity | 1.50 | 15 | -0.012 | -0.032 | 0.008 | 0.855 | -0.209 |
| full | fixed | final_diversity | 2.50 | 15 | 0.001 | -0.023 | 0.024 | 1.000 | 0.084 |
| full | normal_s2 | final_diversity | 1.00 | 15 | -0.001 | -0.016 | 0.015 | 1.000 | 0.000 |
| full | normal_s2 | final_diversity | 1.50 | 15 | -0.005 | -0.026 | 0.018 | 1.000 | -0.004 |
| full | normal_s2 | final_diversity | 2.50 | 15 | -0.012 | -0.028 | 0.003 | 0.465 | -0.200 |
| full | normal_s4 | final_diversity | 1.00 | 15 | -0.009 | -0.025 | 0.007 | 0.689 | -0.240 |
| full | normal_s4 | final_diversity | 1.50 | 15 | -0.015 | -0.039 | 0.008 | 0.689 | -0.244 |
| full | normal_s4 | final_diversity | 2.50 | 15 | -0.002 | -0.018 | 0.013 | 0.773 | 0.218 |
| full | fixed | oscillation_index | 1.00 | 15 | 0.005 | 0.002 | 0.009 | 0.040 | 0.502 |
| full | fixed | oscillation_index | 1.50 | 15 | 0.003 | 0.001 | 0.006 | 0.040 | 0.387 |
| full | fixed | oscillation_index | 2.50 | 15 | -0.008 | -0.012 | -0.004 | 0.009 | -0.600 |
| full | normal_s2 | oscillation_index | 1.00 | 15 | 0.006 | 0.003 | 0.010 | 0.012 | 0.644 |
| full | normal_s2 | oscillation_index | 1.50 | 15 | 0.006 | 0.002 | 0.009 | 0.019 | 0.547 |
| full | normal_s2 | oscillation_index | 2.50 | 15 | -0.005 | -0.008 | -0.001 | 0.038 | -0.493 |
| full | normal_s4 | oscillation_index | 1.00 | 15 | 0.004 | 0.000 | 0.007 | 0.127 | 0.360 |
| full | normal_s4 | oscillation_index | 1.50 | 15 | 0.006 | -0.001 | 0.012 | 0.127 | 0.378 |
| full | normal_s4 | oscillation_index | 2.50 | 15 | -0.006 | -0.010 | -0.002 | 0.038 | -0.600 |
| lcc | fixed | final_diversity | 1.00 | 15 | -0.084 | -0.150 | -0.020 | 0.094 | -0.484 |
| lcc | fixed | final_diversity | 1.50 | 15 | -0.078 | -0.142 | -0.019 | 0.094 | -0.444 |
| lcc | fixed | final_diversity | 2.50 | 15 | -0.080 | -0.146 | -0.011 | 0.094 | -0.480 |
| lcc | normal_s2 | final_diversity | 1.00 | 15 | -0.065 | -0.133 | 0.003 | 0.097 | -0.320 |
| lcc | normal_s2 | final_diversity | 1.50 | 15 | -0.101 | -0.167 | -0.033 | 0.026 | -0.409 |
| lcc | normal_s2 | final_diversity | 2.50 | 15 | -0.167 | -0.203 | -0.132 | 0.000 | -0.911 |
| lcc | normal_s4 | final_diversity | 1.00 | 15 | -0.095 | -0.151 | -0.040 | 0.009 | -0.600 |
| lcc | normal_s4 | final_diversity | 1.50 | 15 | -0.123 | -0.181 | -0.064 | 0.003 | -0.702 |
| lcc | normal_s4 | final_diversity | 2.50 | 15 | -0.133 | -0.180 | -0.081 | 0.002 | -0.711 |
| lcc | fixed | oscillation_index | 1.00 | 15 | 0.029 | 0.018 | 0.040 | 0.001 | 0.813 |
| lcc | fixed | oscillation_index | 1.50 | 15 | 0.021 | 0.009 | 0.034 | 0.013 | 0.600 |
| lcc | fixed | oscillation_index | 2.50 | 15 | -0.010 | -0.021 | 0.001 | 0.124 | -0.236 |
| lcc | normal_s2 | oscillation_index | 1.00 | 15 | 0.027 | 0.016 | 0.039 | 0.002 | 0.751 |
| lcc | normal_s2 | oscillation_index | 1.50 | 15 | 0.029 | 0.017 | 0.041 | 0.002 | 0.742 |
| lcc | normal_s2 | oscillation_index | 2.50 | 15 | -0.005 | -0.015 | 0.005 | 0.355 | -0.164 |
| lcc | normal_s4 | oscillation_index | 1.00 | 15 | 0.022 | 0.007 | 0.040 | 0.026 | 0.529 |
| lcc | normal_s4 | oscillation_index | 1.50 | 15 | 0.011 | -0.005 | 0.026 | 0.194 | 0.289 |
| lcc | normal_s4 | oscillation_index | 2.50 | 15 | -0.020 | -0.033 | -0.008 | 0.022 | -0.547 |
| active_data_init | fixed | final_diversity | 1.00 | 15 | 0.010 | -0.029 | 0.050 | 0.626 | 0.124 |
| active_data_init | fixed | final_diversity | 1.50 | 15 | -0.073 | -0.096 | -0.050 | 0.000 | -0.653 |
| active_data_init | fixed | final_diversity | 2.50 | 15 | -0.128 | -0.161 | -0.096 | 0.000 | -0.929 |
| active_data_init | normal_s4 | final_diversity | 1.00 | 15 | -0.035 | -0.072 | 0.000 | 0.076 | -0.378 |
| active_data_init | normal_s4 | final_diversity | 1.50 | 15 | -0.066 | -0.096 | -0.033 | 0.006 | -0.538 |
| active_data_init | normal_s4 | final_diversity | 2.50 | 15 | -0.115 | -0.142 | -0.084 | 0.001 | -0.840 |
| active_data_init | fixed | oscillation_index | 1.00 | 15 | 0.006 | -0.005 | 0.016 | 0.308 | 0.253 |
| active_data_init | fixed | oscillation_index | 1.50 | 15 | -0.013 | -0.020 | -0.006 | 0.011 | -0.538 |
| active_data_init | fixed | oscillation_index | 2.50 | 15 | -0.029 | -0.034 | -0.022 | 0.000 | -0.893 |
| active_data_init | normal_s4 | oscillation_index | 1.00 | 15 | -0.003 | -0.014 | 0.006 | 0.551 | -0.147 |
| active_data_init | normal_s4 | oscillation_index | 1.50 | 15 | -0.014 | -0.026 | -0.000 | 0.136 | -0.493 |
| active_data_init | normal_s4 | oscillation_index | 2.50 | 15 | -0.032 | -0.038 | -0.026 | 0.000 | -0.982 |

## 7. Update-rule sensitivity

| update rule | q | n | P(consensus) | RMST(T_c) | final D | O |
|---|---|---|---|---|---|---|
| edge_sweep | 0.50 | 8 | 0.000 | 100.000 | 0.450 | 0.054 |
| edge_sweep | 1.00 | 8 | 0.000 | 100.000 | 0.457 | 0.054 |
| edge_sweep | 1.50 | 8 | 0.000 | 100.000 | 0.456 | 0.055 |
| edge_sweep | 2.50 | 8 | 0.000 | 100.000 | 0.455 | 0.063 |
| node_mean | 0.50 | 8 | 0.000 | 100.000 | 0.438 | 0.091 |
| node_mean | 1.00 | 8 | 0.250 | 82.758 | 0.370 | 0.105 |
| node_mean | 1.50 | 8 | 0.750 | 52.833 | 0.257 | 0.143 |
| node_mean | 2.50 | 8 | 1.000 | 34.067 | 0.179 | 0.120 |
| node_sample | 0.50 | 8 | 0.000 | 100.000 | 0.431 | 0.078 |
| node_sample | 1.00 | 8 | 0.125 | 97.003 | 0.414 | 0.110 |
| node_sample | 1.50 | 8 | 0.750 | 63.374 | 0.254 | 0.127 |
| node_sample | 2.50 | 8 | 1.000 | 43.846 | 0.184 | 0.141 |
