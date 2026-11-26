"""Quick numpy-only sanity check: do the engineered features beat raw_margin alone?"""
import numpy as np
import pandas as pd
from scipy.stats import norm

df = pd.read_csv('ml-model/features_v2.csv', dtype={'game_id': str})
drop = ['game_id','date','season','home','away','margin','total','home_won']
feats = [c for c in df.columns if c not in drop]

def ridge(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(Xs),1)), Xs])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[0,0] -= lam  # don't regularize intercept
    w = np.linalg.solve(A, Xb.T @ y)
    return lambda Xn: np.hstack([np.ones((len(Xn),1)), (Xn-mu)/sd]) @ w

tr = df[df.season.isin(['2023-24','2024-25'])]
te = df[df.season == '2025-26']

Xtr_full = tr[feats].fillna(0).values
Xte_full = te[feats].fillna(0).values
Xtr_raw  = tr[['raw_margin']].values
Xte_raw  = te[['raw_margin']].values

ytr_m = tr.margin.values; yte_m = te.margin.values
ytr_t = tr.total.values;  yte_t = te.total.values

def mae(p, y): return float(np.mean(np.abs(p-y)))
def brier(pm, sig, y): return float(np.mean((norm.cdf(pm/sig)-y)**2))

for name, xtr, xte in [('raw_margin only', Xtr_raw, Xte_raw),
                       ('all 33 features', Xtr_full, Xte_full)]:
    fm = ridge(xtr, ytr_m); ft = ridge(xtr, ytr_t)
    pm_tr = fm(xtr); pt_tr = ft(xtr)
    pm_te = fm(xte); pt_te = ft(xte)
    sig_m = float(np.std(ytr_m - pm_tr, ddof=1))
    print(f"\n[{name}]")
    print(f"  MAE margin {mae(pm_te, yte_m):.3f}   MAE total {mae(pt_te, yte_t):.3f}")
    print(f"  Brier     {brier(pm_te, sig_m, te.home_won.values):.4f}")
