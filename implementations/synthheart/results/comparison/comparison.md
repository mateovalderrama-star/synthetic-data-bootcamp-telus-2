# TabDDPM vs NeMo Safe Synthesizer · measured results

Built by `06_compare_generators.py`. Same train split, same medical-rule filter, same scorecard (03b),
same locked real test set for every column. Method notes and caveats: `../../NSS_COMPARISON.md`.


## Fidelity

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| α-precision (synthetic rows look real) | 0.998 | 0.962 | 0.958 | ≥ 0.90 |
| β-recall (real variety is covered) | 0.499 | 0.514 | 0.521 | higher |
| Columns whose distribution differs (KS / TVD) | 0 | 5 | 5 | ≤ 1 |
| Mean KS / TVD distance | 0.0047 | 0.0221 | 0.0227 | lower |
| Confidence-interval overlap | 0.882 | 0.096 | 0.272 | higher |
| Correlation-matrix gap | 0.041 | 0.072 | 0.094 | ≤ 0.10 |
| Mutual-information gap | 0.035 | 0.033 | 0.049 | lower |
| Rows passing the medical rules | 99.99% | 99.99% | 100.00% | ≥ 98% |

## Utility

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| TSTR AUC, 1× synthetic (real test set) | 0.7981 | 0.7952 | 0.8012 | ≥ real − 0.02 |
| TSTR AUC, 2× synthetic | 0.7999 | 0.7969 | 0.8016 | higher |
| Real-data baseline AUC (TRTR) | 0.8014 | 0.8014 | 0.8014 | reference |
| AUC lost vs real data | +0.0033 | +0.0062 | +0.0002 | ≤ 0.02 |
| Mean F1 difference (holdout) | -0.0001 | -0.0016 | -0.0029 | near 0 |

## Privacy

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| Membership-inference AUC | 0.5081 | 0.5044 | 0.5069 | ≤ 0.55 (0.50 = coin flip) |
| MIA true-positive rate at 10% false positives | 0.100 | 0.099 | 0.097 | ≈ 0.10 |
| DCR score | 0.7574 | 0.7359 | 0.7387 | ≈ ideal |
| DCR ideal (train share) | 0.7500 | 0.7405 | 0.7405 | reference |
| Exact copies: train sample ÷ holdout | 0.97 | 1.11 | 1.16 | ≤ 1.2 |
| Hitting rate, train sample | 0.371 | 0.387 | 0.389 | ≈ holdout |
| Hitting rate, holdout | 0.424 | 0.426 | 0.427 | reference |
| ε-identifiability risk | 0.503 | 0.499 | 0.507 | lower |
| NNDR | 0.762 | 0.754 | 0.749 | higher |

## Relationships

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| BP × age risk table, max difference | 0.016 | 0.049 | 0.032 | lower |
| Feature-importance agreement (Spearman) | 0.993 | 0.790 | 0.972 | higher |

## Overall

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| Acceptance targets passed | 8/8 | 7/8 | 7/8 | 8/8 |

## Runtime

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| Training (min) | 6.8 | 10.0 | 46.5 | lower |
| Generating one train-size table (min) | 0.60 | 4.92 | 4.68 | lower |
| Generation speed (rows / s) | 1,143 | 139 | 146 | higher |
| Whole run, wall clock (min) | 8.9 | 20.0 | 56.0 | lower |
| Peak GPU memory (GB) | 1.3 | 114.1 | 114.1 | lower |
| GPU energy (Wh) | 32 | 81 | 214 | lower |
| Mean GPU utilisation (%) | 31 | 27 | 29 | - |
| Trained model on disk (MB) | 21 | 61 | 61 | lower |

## Data

| Metric | TabDDPM | NSS default | NSS 3pass | Good is |
|---|---:|---:|---:|---|
| Train rows the generator saw | 41,135 | 39,135 | 39,135 | - |
| Synthetic rows scored (1×) | 41,131 | 41,132 | 41,134 | - |

## NSS's own report (as NSS computes it, on its own 5% holdout)

Scores are out of 10. Full report: `nss_native/<run>_evaluation_report.html`.

| NSS summary field | NSS default | NSS 3pass |
|---|---:|---:|
| `attribute_inference_protection_score` | 9.1 | 8.8 |
| `column_correlation_stability_score` | 9.6 | 9.5 |
| `column_distribution_stability_score` | 9.7 | 9.6 |
| `data_privacy_score` | 9.6 | 9.4 |
| `deep_structure_stability_score` | 9.2 | 9.1 |
| `membership_inference_protection_score` | 10 | 10 |
| `num_completion_tokens` | 5.417e+06 | 5.412e+06 |
| `num_invalid_record_tokens` | 9517 | 5661 |
| `num_invalid_records` | 156 | 93 |
| `num_non_record_tokens` | 3.811e+05 | 3.806e+05 |
| `num_prompts` | 6351 | 6345 |
| `num_valid_record_tokens` | 5.027e+06 | 5.026e+06 |
| `num_valid_records` | 8.241e+04 | 8.239e+04 |
| `synthetic_data_quality_score` | 9.5 | 9.4 |
| `timing.evaluation_time_sec` | 6.849 | 6.703 |
| `timing.generation_time_sec` | 462.3 | 467.2 |
| `timing.total_time_sec` | 1199 | 3359 |
| `timing.training_time_sec` | 569.3 | 2771 |
| `tokenization_overhead_sec` | 8.92 | 9.214 |
| `tokens_per_prompt` | 853 | 853 |
| `tokens_per_second` | 1.172e+04 | 1.158e+04 |
| `valid_record_fraction` | 0.9981 | 0.9989 |
| `valid_record_token_fraction` | 0.9279 | 0.9286 |
| `valid_tokens_per_second` | 1.087e+04 | 1.076e+04 |

## Reading notes

- **Train rows seen.** NSS keeps 5% of its input (max 2,000 rows) out of training for its own privacy tests. The scorecard counts only the rows a generator trained on as `train`, so privacy metrics are exact.
- **Peak GPU memory** is the whole device. NSS's generation engine (vLLM) reserves about 80% of the card up front, so its peak shows a reservation, not a need. NVIDIA lists an A100 as the minimum.
- **Generation time** is the time of the sampling call divided by the rows it returned, scaled to one train-size table. It includes loading the model.
- **One run per generator, so no error bars.** Treat gaps of a few thousandths of AUC as ties until the runs are repeated with other seeds.
