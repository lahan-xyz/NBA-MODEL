"""Ensemble baseline + ridge + GBDT with OOF weight fitting.
Fast version: max_features subsampling, 3-fold OOF, 100 trees."""
import sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, 'ml-model')
from gbdt_numpy import NumpyGBDT

t0 = time.time()
df = pd.read_csv('ml-model/features_v2.csv', dtype={'game_id': str})
train = df[df.season.isin(['2023-24','2024-25'])].reset_index(drop=True)
test  = df[df.season == '2025-26'].reset_index(drop=True)
drop = ['game_id','date','season','home','away','margin','total','home_won']
feats = [c for c in df.columns if c not in drop]
X_tr, X_te = train[feats].fillna(0).values, test[feats].fillna(0).values
ym_tr, ym_te = train.margin.values, test.margin.values
yt_tr, yt_te = train.total.values,  test.total.values
yw_te = test.home_won.values
print(f"Loaded: train={len(train)} test={len(test)} feats={len(feats)}  [{time.time()-t0:.1f}s]")

# --- baseline ---
m_b, m_a = np.polyfit(train.raw_margin, train.margin, 1)
t_b, t_a = np.polyfit(train.raw_total,  train.total,  1)
base_m_tr = m_a + m_b * train.raw_margin.values
base_m_te = m_a + m_b * test.raw_margin.values
base_t_tr = t_a + t_b * train.raw_total.values
base_t_te = t_a + t_b * test.raw_total.values
base_sig  = float(np.std(ym_tr - base_m_tr, ddof=1))
print(f"Baseline fit  [{time.time()-t0:.1f}s]")

# --- ridge ---
def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xb = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam
    w = np.linalg.solve(A, Xb.T @ y)
    return lambda Xn: np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd]) @ w
ridge_m_tr = ridge_fit(X_tr, ym_tr)(X_tr); ridge_m_te = ridge_fit(X_tr, ym_tr)(X_te)
ridge_t_tr = ridge_fit(X_tr, yt_tr)(X_tr); ridge_t_te = ridge_fit(X_tr, yt_tr)(X_te)
ridge_sig  = float(np.std(ym_tr - ridge_m_tr, ddof=1))
print(f"Ridge fit  [{time.time()-t0:.1f}s]")

# --- GBDT via 3-fold OOF, max_features=15 ---
def kfold(n, k, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]

gbdt_m_oof = np.zeros(len(train))
gbdt_t_oof = np.zeros(len(train))
folds = list(kfold(len(train), 3))
for i, (tr_idx, va_idx) in enumerate(folds):
    ti = time.time()
    print(f"  [fold {i+1}/3] training margin head...")
    gm = NumpyGBDT(n_estimators=100, learning_rate=0.06, max_depth=3,
                   min_samples_leaf=40, max_features=15, subsample=0.8,
                   random_state=100+i)
    gm.fit(X_tr[tr_idx], ym_tr[tr_idx], X_val=X_tr[va_idx],
           y_val=ym_tr[va_idx], early_stopping_rounds=20)
    gbdt_m_oof[va_idx] = gm.predict(X_tr[va_idx])
    print(f"  [fold {i+1}/3] total head...")
    gt = NumpyGBDT(n_estimators=100, learning_rate=0.06, max_depth=3,
                   min_samples_leaf=40, max_features=15, subsample=0.8,
                   random_state=200+i)
    gt.fit(X_tr[tr_idx], yt_tr[tr_idx], X_val=X_tr[va_idx],
           y_val=yt_tr[va_idx], early_stopping_rounds=20)
    gbdt_t_oof[va_idx] = gt.predict(X_tr[va_idx])
    print(f"  [fold {i+1}/3] done in {time.time()-ti:.1f}s  [{time.time()-t0:.1f}s total]")

print(f"Training final full-fit GBDTs...")
gbdt_m_full = NumpyGBDT(n_estimators=150, learning_rate=0.06, max_depth=3,
                        min_samples_leaf=40, max_features=15, subsample=0.8,
                        random_state=42)
gbdt_m_full.fit(X_tr, ym_tr, X_val=X_tr[:300], y_val=ym_tr[:300],
                early_stopping_rounds=30)
gbdt_t_full = NumpyGBDT(n_estimators=150, learning_rate=0.06, max_depth=3,
                        min_samples_leaf=40, max_features=15, subsample=0.8,
                        random_state=42)
gbdt_t_full.fit(X_tr, yt_tr, X_val=X_tr[:300], y_val=yt_tr[:300],
                early_stopping_rounds=30)
gbdt_m_te = gbdt_m_full.predict(X_te)
gbdt_t_te = gbdt_t_full.predict(X_te)
gbdt_sig = float(np.std(ym_tr - gbdt_m_oof, ddof=1))
print(f"GBDT done  [{time.time()-t0:.1f}s]")

# --- ensemble weights (OOF least squares, no intercept) ---
def fit_w(preds, y):
    A = np.column_stack(preds)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w
w_m = fit_w([base_m_tr, ridge_m_tr, gbdt_m_oof], ym_tr)
w_t = fit_w([base_t_tr, ridge_t_tr, gbdt_t_oof], yt_tr)
print(f"\nEnsemble weights (margin): base={w_m[0]:+.3f} ridge={w_m[1]:+.3f} gbdt={w_m[2]:+.3f}")
print(f"Ensemble weights (total):  base={w_t[0]:+.3f} ridge={w_t[1]:+.3f} gbdt={w_t[2]:+.3f}\n")

ens_m_te = w_m[0]*base_m_te + w_m[1]*ridge_m_te + w_m[2]*gbdt_m_te
ens_t_te = w_t[0]*base_t_te + w_t[1]*ridge_t_te + w_t[2]*gbdt_t_te
ens_sig  = float(np.std(ym_tr - (w_m[0]*base_m_tr + w_m[1]*ridge_m_tr + w_m[2]*gbdt_m_oof), ddof=1))

# --- metrics ---
def mae(p, y): return float(np.mean(np.abs(p-y)))
def brier(pm, s, y): return float(np.mean((norm.cdf(pm/s)-y)**2))
def acc(pm, y): return float(np.mean((pm>0).astype(int)==y))

print(f"{'Model':<14} {'MAE m':>8} {'MAE t':>8} {'Brier':>8} {'Acc':>8}")
print('-'*50)
for name, pm, pt, sig in [
    ('Baseline', base_m_te,  base_t_te,  base_sig),
    ('Ridge',    ridge_m_te, ridge_t_te, ridge_sig),
    ('GBDT',     gbdt_m_te,  gbdt_t_te,  gbdt_sig),
    ('Ensemble', ens_m_te,   ens_t_te,   ens_sig),
]:
    print(f"{name:<14} {mae(pm,ym_te):>8.3f} {mae(pt,yt_te):>8.3f} "
          f"{brier(pm,sig,yw_te):>8.4f} {acc(pm,yw_te):>8.3f}")

print("\nEnsemble calibration (95% CI on actual):")
p_ens = norm.cdf(ens_m_te / ens_sig)
for lo, hi, lbl in [(0,.3,'<30%'),(.3,.4,'30-40%'),(.4,.5,'40-50%'),
                    (.5,.6,'50-60%'),(.6,.7,'60-70%'),(.7,1,'>70%')]:
    m = (p_ens>=lo)&(p_ens<hi); n=m.sum()
    if n==0: continue
    p_hat = float(yw_te[m].mean()); se = np.sqrt(p_hat*(1-p_hat)/n)
    print(f"  {lbl}: n={n:4d}  pred={p_ens[m].mean():.3f}  actual={p_hat:.3f} ± {1.96*se:.3f}")

out = test[['game_id','date','home','away','margin','total','home_won']].copy()
out['base_m']=base_m_te; out['ridge_m']=ridge_m_te; out['gbdt_m']=gbdt_m_te
out['ens_m']=ens_m_te;   out['ens_t']=ens_t_te;    out['ens_p']=p_ens
out.to_csv('ml-model/ensemble_predictions.csv', index=False)
print(f"\nWrote ml-model/ensemble_predictions.csv  [total {time.time()-t0:.1f}s]")
