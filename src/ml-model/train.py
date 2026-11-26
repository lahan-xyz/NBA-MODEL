"""Walk-forward train + evaluate: NumpyGBDT vs EWMA+linear baseline.
No sklearn required — uses ml-model/gbdt_numpy.py."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, 'ml-model')
from gbdt_numpy import NumpyGBDT

df = pd.read_csv('ml-model/features_v2.csv', dtype={'game_id': str})
TRAIN = {'2023-24', '2024-25'}
TEST  = {'2025-26'}
train = df[df.season.isin(TRAIN)].reset_index(drop=True)
test  = df[df.season.isin(TEST)].reset_index(drop=True)

drop = ['game_id', 'date', 'season', 'home', 'away',
        'margin', 'total', 'home_won']
feats = [c for c in df.columns if c not in drop]
print(f"Train: {len(train)}  Test: {len(test)}  Features: {len(feats)}")

# internal validation split from train (chronological — last 20%)
split = int(len(train) * 0.8)
tr, va = train.iloc[:split], train.iloc[split:]

X_tr = tr[feats].fillna(0).values
X_va = va[feats].fillna(0).values
X_te = test[feats].fillna(0).values

ym_tr, ym_va, ym_te = tr.margin.values, va.margin.values, test.margin.values
yt_tr, yt_va, yt_te = tr.total.values,  va.total.values,  test.total.values
yw_te = test.home_won.values

# baseline linear fit on full train (matches build_model.js)
m_b, m_a = np.polyfit(train.raw_margin, train.margin, 1)
t_b, t_a = np.polyfit(train.raw_total,  train.total,  1)
bm_tr = m_a + m_b * train.raw_margin.values
bt_tr = t_a + t_b * train.raw_total.values
bm_te = m_a + m_b * test.raw_margin.values
bt_te = t_a + t_b * test.raw_total.values
b_sig_m = float(np.std(train.margin.values - bm_tr, ddof=1))
b_sig_t = float(np.std(train.total.values  - bt_tr, ddof=1))

# ---- ML heads ----
print("\nFitting margin head...")
ml_m = NumpyGBDT(n_estimators=400, learning_rate=0.05, max_depth=4,
                 min_samples_leaf=25, subsample=0.8, random_state=42, verbose=True)
ml_m.fit(X_tr, ym_tr, X_val=X_va, y_val=ym_va, early_stopping_rounds=40)

print("Fitting total head...")
ml_t = NumpyGBDT(n_estimators=400, learning_rate=0.05, max_depth=4,
                 min_samples_leaf=25, subsample=0.8, random_state=42, verbose=True)
ml_t.fit(X_tr, yt_tr, X_val=X_va, y_val=yt_va, early_stopping_rounds=40)

pm_te = ml_m.predict(X_te)
pt_te = ml_t.predict(X_te)
pm_tr = ml_m.predict(X_tr)
ml_sig_m = float(np.std(ym_tr - pm_tr, ddof=1))

# ---- metrics ----
def mae(p, y):  return float(np.mean(np.abs(p - y)))
def brier(pm, s, y): return float(np.mean((norm.cdf(pm/s) - y) ** 2))
def acc(pm, y):  return float(np.mean((pm > 0).astype(int) == y))

print(f"\n{'='*62}")
print(f"{'Metric':<26} {'Baseline':>12} {'NumPy GBDT':>12} {'Δ':>10}")
print('-'*62)
for name, b, m in [
    ('MAE margin',       mae(bm_te, ym_te), mae(pm_te, ym_te)),
    ('MAE total',        mae(bt_te, yt_te), mae(pt_te, yt_te)),
    ('Brier (win prob)', brier(bm_te, b_sig_m, yw_te),
                         brier(pm_te, ml_sig_m, yw_te)),
    ('Winner accuracy',  acc(bm_te, yw_te), acc(pm_te, yw_te)),
]:
    print(f"{name:<26} {b:>12.4f} {m:>12.4f} {m-b:>+10.4f}")
print('='*62)

# ---- calibration ----
print("\nGBDT calibration (margin head):")
p = norm.cdf(pm_te / ml_sig_m)
for lo, hi, lbl in [(0,.3,'<30%'),(.3,.4,'30-40%'),(.4,.5,'40-50%'),
                    (.5,.6,'50-60%'),(.6,.7,'60-70%'),(.7,1,'>70%')]:
    m = (p >= lo) & (p < hi)
    if m.sum():
        print(f"  {lbl}: n={m.sum():4d}  pred={p[m].mean():.3f}  actual={yw_te[m].mean():.3f}")

# ---- feature importance ----
print("\nTop features (margin, split-count importance):")
for j in np.argsort(-ml_m.feature_importances_)[:10]:
    print(f"  {feats[j]:<16} {ml_m.feature_importances_[j]:.4f}")

pd.DataFrame({
    'game_id': test.game_id, 'date': test.date,
    'home': test.home, 'away': test.away,
    'actual_margin': test.margin, 'actual_total': test.total,
    'base_m': bm_te, 'ml_m': pm_te,
    'base_t': bt_te, 'ml_t': pt_te,
    'base_p': norm.cdf(bm_te / b_sig_m), 'ml_p': p,
}).to_csv('ml-model/test_predictions.csv', index=False)
print("\nWrote ml-model/test_predictions.csv")
