# Reference implementations

These scripts document the experiments in `../EXPERIMENTS.md`. They are
not part of the production pipeline — `build_model.js` is.

Kept as starting points if external data (injury reports, EPM, line
movement) is ever integrated.

- `extract_features.py` — leak-free team + schedule features (numpy)
- `fetch_player_logs.py` — bulk per-player logs via nba_api
- `extract_lineup_features.py` — last-game availability proxies
- `gbdt_numpy.py` — pure-numpy gradient boosting (no sklearn)
- `ridge_check.py`, `train.py`, `ensemble.py`, `tune_ewma.py` — evaluations
