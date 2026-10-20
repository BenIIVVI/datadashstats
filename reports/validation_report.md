# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 9.57% (9.40 to 9.74) | 7.38% (7.24 to 7.58) | +1.32% |
| list_aware | 10,069 | 4.88% (4.81 to 4.97) | 3.83% (3.73 to 3.91) | -0.11% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)


## Seasonal backtests (train before June 1, predict June to September)

| Season | Track | Homes | MdAPE | MAPE | Median signed error |
|---|---|---|---|---|---|
| 2023 | list_aware | 8,302 | 4.52% | 5.66% | +0.52% |
| 2023 | off_market | 8,302 | 6.74% | 8.96% | +0.83% |
| 2024 | list_aware | 7,630 | 4.24% | 5.35% | +0.27% |
| 2024 | off_market | 7,630 | 6.65% | 9.00% | +1.38% |
| 2025 (held back) | list_aware | 8,052 | 3.86% | 4.88% | -0.05% |
| 2025 (held back) | off_market | 8,052 | 6.86% | 9.20% | +0.91% |

### off_market, 2025 season

80% range from earlier seasons covers 78.9% of homes (target 80%); median width 23.3% of the price.

| Property type | Homes | MdAPE | Median signed error | In 80% range |
|---|---|---|---|---|
| Co-Op Apt | 50 | 7.52% | +2.03% | 78.0% |
| Condo Apt | 3,623 | 5.55% | +1.36% | 79.9% |
| Condo Townhouse | 554 | 5.48% | +0.10% | 79.4% |
| Detached | 2,683 | 8.82% | +1.12% | 78.2% |
| Freehold Townhouse | 292 | 8.70% | +2.64% | 73.3% |
| Semi-Detached | 850 | 8.75% | -2.31% | 78.8% |

MdAPE by predicted-price tenth (1 = cheapest; grouped by prediction, not sale price, so noise in the sale does not exaggerate the pattern): 1: 5.1%, 2: 5.3%, 3: 5.2%, 4: 5.6%, 5: 6.5%, 6: 8.2%, 7: 7.9%, 8: 8.4%, 9: 9.2%, 10: 10.9%

### list_aware, 2025 season

80% range from earlier seasons covers 85.4% of homes (target 80%); median width 16.5% of the price.

| Property type | Homes | MdAPE | Median signed error | In 80% range |
|---|---|---|---|---|
| Co-Op Apt | 50 | 3.70% | +1.14% | 86.0% |
| Condo Apt | 3,623 | 3.14% | -0.61% | 88.0% |
| Condo Townhouse | 554 | 3.59% | +0.20% | 85.0% |
| Detached | 2,683 | 4.59% | +1.00% | 84.9% |
| Freehold Townhouse | 292 | 4.78% | +0.61% | 80.5% |
| Semi-Detached | 850 | 5.32% | +0.16% | 77.4% |

MdAPE by predicted-price tenth (1 = cheapest; grouped by prediction, not sale price, so noise in the sale does not exaggerate the pattern): 1: 3.1%, 2: 3.1%, 3: 3.2%, 4: 3.3%, 5: 3.9%, 6: 4.2%, 7: 4.4%, 8: 4.4%, 9: 4.8%, 10: 4.9%