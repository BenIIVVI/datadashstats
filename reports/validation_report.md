# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 9.44% (9.28 to 9.61) | 7.31% (7.14 to 7.50) | +1.38% |
| list_aware | 10,069 | 5.33% (5.24 to 5.42) | 4.19% (4.09 to 4.31) | -0.62% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)


## Seasonal backtests (train before June 1, predict June to September)

| Season | Track | Homes | MdAPE | MAPE | Median signed error |
|---|---|---|---|---|---|
| 2023 | list_aware | 8,302 | 5.50% | 6.66% | +1.17% |
| 2023 | off_market | 8,302 | 6.62% | 8.88% | +0.93% |
| 2024 | list_aware | 7,630 | 5.06% | 6.21% | +1.41% |
| 2024 | off_market | 7,630 | 6.64% | 8.96% | +1.54% |
| 2025 (held back) | list_aware | 8,052 | 4.30% | 5.37% | +0.84% |
| 2025 (held back) | off_market | 8,052 | 6.84% | 9.08% | +0.91% |

### off_market, 2025 season

80% range from earlier seasons covers 79.5% of homes (target 80%); median width 22.6% of the price.

| Property type | Homes | MdAPE | Median signed error | In 80% range |
|---|---|---|---|---|
| Co-Op Apt | 50 | 8.46% | +2.85% | 80.0% |
| Condo Apt | 3,623 | 5.54% | +1.45% | 80.4% |
| Condo Townhouse | 554 | 5.63% | -0.02% | 79.1% |
| Detached | 2,683 | 8.73% | +1.00% | 78.8% |
| Freehold Townhouse | 292 | 8.42% | +2.09% | 76.4% |
| Semi-Detached | 850 | 8.39% | -2.12% | 79.1% |

MdAPE by predicted-price tenth (1 = cheapest; grouped by prediction, not sale price, so noise in the sale does not exaggerate the pattern): 1: 5.1%, 2: 5.3%, 3: 5.2%, 4: 5.5%, 5: 6.6%, 6: 7.8%, 7: 7.4%, 8: 8.2%, 9: 9.1%, 10: 10.5%

### list_aware, 2025 season

80% range from earlier seasons covers 87.9% of homes (target 80%); median width 20.7% of the price.

| Property type | Homes | MdAPE | Median signed error | In 80% range |
|---|---|---|---|---|
| Co-Op Apt | 50 | 3.16% | -0.36% | 92.0% |
| Condo Apt | 3,623 | 3.43% | -0.27% | 93.1% |
| Condo Townhouse | 554 | 4.57% | +1.58% | 88.3% |
| Detached | 2,683 | 5.19% | +2.34% | 85.0% |
| Freehold Townhouse | 292 | 5.27% | +2.09% | 81.5% |
| Semi-Detached | 850 | 6.40% | +2.10% | 76.4% |

MdAPE by predicted-price tenth (1 = cheapest; grouped by prediction, not sale price, so noise in the sale does not exaggerate the pattern): 1: 3.4%, 2: 3.4%, 3: 3.7%, 4: 3.7%, 5: 4.4%, 6: 5.1%, 7: 5.2%, 8: 4.8%, 9: 5.0%, 10: 4.7%