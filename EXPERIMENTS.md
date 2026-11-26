# Modeling Experiments (Nov 2026)

To bound what public team-level data can predict, four model classes
were trained on a leak-free 3-season dataset (3,360 games, 2,189 train /
1,171 test via walk-forward split):

| Model | Margin MAE | Total MAE | Brier | Accuracy |
|---|---|---|---|---|
| EWMA + linear (production) | **11.57** | 14.75 | **0.202** | **68.3%** |
| Ridge on 39 engineered features | 11.61 | **14.61** | 0.204 | 67.6% |
| NumPy GBDT (OOF) | 11.84 | 15.04 | 0.208 | 65.8% |
| Optimal ensemble | 11.61 | 14.62 | 0.204 | 67.8% |

Grid search over 81 EWMA hyperparameter combinations finds no
improvement beyond noise (best margin 11.568 vs current 11.572).

**Conclusion:** At team-level + last-game-player granularity from
`nba_api`, no model class beats MAE ~11.5 on margin. The gap to
professional models (~10.5 MAE) is data (live injury reports, EPM),
not algorithm.

See `ml-model/` for reference implementations.
