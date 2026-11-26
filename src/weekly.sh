#!/data/data/com.termux/files/usr/bin/bash
set -e
cd "$(dirname "$0")"
echo "== 1/5 results =="   && python refresh.py
echo "== 2/5 rest =="     && python add_features.py
echo "== 3/5 model =="    && node build_model.js
echo "== 4/5 fixtures ==" && python fetch_fixtures.py --days "${1:-7}"
echo "== 5/5 predict ==" && node predict_week.js
echo "done: data/predictions.json"
