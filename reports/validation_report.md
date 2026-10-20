# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 9.77% (9.60 to 9.95) | 7.57% (7.41 to 7.73) | +1.32% |
| list_aware | 10,069 | 4.92% (4.84 to 5.00) | 3.85% (3.76 to 3.94) | -0.09% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)


## Seasonal backtests (train before June 1, predict June to September)

| Season | Track | Homes | MdAPE | MAPE | Median signed error |
|---|---|---|---|---|---|
| 2023 | list_aware | 8,302 | 4.56% | 5.69% | +0.55% |
| 2023 | off_market | 8,302 | 6.81% | 9.16% | +0.86% |
| 2024 | list_aware | 7,630 | 4.25% | 5.38% | +0.31% |
| 2024 | off_market | 7,630 | 6.87% | 9.24% | +1.32% |
| 2025 (held back) | list_aware | 8,052 | 3.88% | 4.89% | -0.02% |
| 2025 (held back) | off_market | 8,052 | 6.97% | 9.40% | +0.99% |

### off_market, 2025 season

80% range from earlier seasons covers 79.4% of homes (target 80%); median width 23.3% of the price.

| Property type | Homes | MdAPE | Median signed error | In 80% range |
|---|---|---|---|---|
| Co-Op Apt | 50 | 8.67% | +3.65% | 80.0% |
| Condo Apt | 3,623 | 5.66% | +1.36% | 80.2% |
| Condo Townhouse | 554 | 5.69% | -0.18% | 79.2% |
| Detached | 2,683 | 9.05% | +1.33% | 79.8% |
| Freehold Townhouse | 292 | 9.45% | +2.62% | 72.6% |
| Semi-Detached | 850 | 8.83% | -2.10% | 77.5% |

MdAPE by predicted-price tenth (1 = cheapest; grouped by prediction, not sale price, so noise in the sale does not exaggerate the pattern): 1: 5.3%, 2: 5.1%, 3: 5.3%, 4: 5.8%, 5: 6.4%, 6: 8.2%, 7: 8.1%, 8: 8.3%, 9: 9.6%, 10: 11.1%

### list_aware, 2025 season

80% range from earlier seasons covers 85.4% of homes (target 80%); median width 16.7% of the price.

| Property type | Homes | MdAPE | Median signed error | In 80% range |
|---|---|---|---|---|
| Co-Op Apt | 50 | 3.58% | +0.76% | 90.0% |
| Condo Apt | 3,623 | 3.15% | -0.64% | 87.9% |
| Condo Townhouse | 554 | 3.70% | +0.17% | 84.8% |
| Detached | 2,683 | 4.72% | +0.87% | 84.9% |
| Freehold Townhouse | 292 | 4.66% | +0.52% | 80.8% |
| Semi-Detached | 850 | 5.42% | +0.24% | 77.8% |

MdAPE by predicted-price tenth (1 = cheapest; grouped by prediction, not sale price, so noise in the sale does not exaggerate the pattern): 1: 3.0%, 2: 3.2%, 3: 3.3%, 4: 3.3%, 5: 4.0%, 6: 4.1%, 7: 4.6%, 8: 4.4%, 9: 4.8%, 10: 4.8%