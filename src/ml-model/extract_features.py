"""
Extract a leak-free feature matrix for the ML model.

Mirrors the EWMA logic in build_model.js exactly, so the baseline
(HistGradientBoosting trained on these features) is directly comparable
to the existing EWMA + linear-calibration model.

Output: ml-model/features.csv — one row per in-scope game, all features
        computed as-of tip-off (no future info).
"""
import json
from datetime import date
import numpy as np
import pandas as pd

# --- hyperparameters: mirrored from build_model.js ---
LAM_O = 0.50
LAM_D = 0.25
ALPHA = 0.06
PACE_A = 0.15
INIT = 110.0
IPACE = 99.0
DONE = {'2023-24', '2024-25', '2025-26'}   # seasons that feed calibration

# --- load raw data ---
games = json.load(open('data/games.json'))
factors_list = json.load(open('data/factors.json'))
F = {f['game_id']: f for f in factors_list}

# League 3P% per season (mirrors build_model.js)
LG3 = {}
for s in {f['season'] for f in factors_list}:
    m3 = sum(f['h_FG3M'] + f['a_FG3M'] for f in factors_list if f['season'] == s)
    a3 = sum(f['h_FG3A'] + f['a_FG3A'] for f in factors_list if f['season'] == s)
    LG3[s] = m3 / a3 if a3 else 0.36


def adj_pts(f, side, lam, p3):
    m3 = lam * f[f'{side}_FG3M'] + (1 - lam) * f[f'{side}_FG3A'] * p3
    return 2 * (f[f'{side}_FGM'] - f[f'{side}_FG3M']) + 3 * m3 + f[f'{side}_FTM']


# Ensure chronological order (defensive — should already be sorted)
games = sorted(games, key=lambda g: (g['date'], g['game_id']))

# --- ratings state, one entry per team ---
R = {}
def get(t):
    if t not in R:
        R[t] = {
            'o': INIT, 'd': INIT, 'p': IPACE, 'n': 0,
            'last_date': None,
            'hist_o': [], 'hist_d': [], 'hist_p': [],
        }
    return R[t]


def roll(vals, n):
    """Mean of last n values; NaN if fewer than n available."""
    if len(vals) < n:
        return float('nan')
    return float(sum(vals[-n:]) / n)


def safe_days(d1, d2):
    try:
        return (date.fromisoformat(d1) - date.fromisoformat(d2)).days
    except Exception:
        return float('nan')


league = INIT
rows = []

for g in games:
    h = get(g['home'])
    a = get(g['away'])

    # ---- PRE-TIP FEATURES ----
    pace = (h['p'] + a['p']) / 2
    hRaw = (h['o'] + a['d'] - league) * pace / 100
    aRaw = (a['o'] + h['d'] - league) * pace / 100
    rawM = hRaw - aRaw
    rawT = hRaw + aRaw

    # rest / b2b — only meaningful if team has already played this season
    if h['n'] > 0 and h['last_date']:
        h_rest = safe_days(g['date'], h['last_date'])
    else:
        h_rest = float('nan')
    if a['n'] > 0 and a['last_date']:
        a_rest = safe_days(g['date'], a['last_date'])
    else:
        a_rest = float('nan')

    h_b2b = 1 if (not np.isnan(h_rest) and h_rest <= 1) else 0
    a_b2b = 1 if (not np.isnan(a_rest) and a_rest <= 1) else 0
    h_3in4 = 1 if (not np.isnan(h_rest) and h_rest <= 2 and h['n'] >= 2) else 0
    a_3in4 = 1 if (not np.isnan(a_rest) and a_rest <= 2 and a['n'] >= 2) else 0

    feat = {
        'game_id': g['game_id'],
        'date': g['date'],
        'season': g['season'],
        'home': g['home'],
        'away': g['away'],
        # EWMA ratings as-of tip
        'h_o': h['o'], 'h_d': h['d'], 'h_p': h['p'],
        'a_o': a['o'], 'a_d': a['d'], 'a_p': a['p'],
        'h_n': h['n'], 'a_n': a['n'],
        'league': league,
        'pace_avg': pace,
        # baseline linear prediction (diagnostic + useful tree feature)
        'raw_margin': rawM,
        'raw_total': rawT,
        # rest / schedule context
        'h_rest': h_rest, 'a_rest': a_rest,
        'h_b2b': h_b2b, 'a_b2b': a_b2b,
        'h_3in4': h_3in4, 'a_3in4': a_3in4,
        'rest_diff': h_rest - a_rest if not (np.isnan(h_rest) or np.isnan(a_rest)) else float('nan'),
        # rolling windows of past games only
        'h_o_roll5':  roll(h['hist_o'], 5),
        'h_o_roll10': roll(h['hist_o'], 10),
        'h_d_roll5':  roll(h['hist_d'], 5),
        'h_d_roll10': roll(h['hist_d'], 10),
        'h_p_roll5':  roll(h['hist_p'], 5),
        'a_o_roll5':  roll(a['hist_o'], 5),
        'a_o_roll10': roll(a['hist_o'], 10),
        'a_d_roll5':  roll(a['hist_d'], 5),
        'a_d_roll10': roll(a['hist_d'], 10),
        'a_p_roll5':  roll(a['hist_p'], 5),
    }

    # ---- TARGET GATE (mirror build_model.js: non-OT, in DONE, both n>=10) ----
    is_completed = g.get('home_pts') is not None and g.get('away_pts') is not None
    if is_completed and g['season'] in DONE and not g.get('ot', False) and h['n'] >= 10 and a['n'] >= 10:
        feat['margin'] = g['home_pts'] - g['away_pts']
        feat['total'] = g['home_pts'] + g['away_pts']
        feat['home_won'] = 1 if feat['margin'] > 0 else 0
        rows.append(feat)

    # ---- POST-GAME STATE UPDATE (only after feature extraction) ----
    if is_completed:
        f = F.get(g['game_id'])
        if f:
            p3 = LG3.get(g['season'], 0.36)
            hO = adj_pts(f, 'h', LAM_O, p3) / g['poss'] * 100
            aO = adj_pts(f, 'a', LAM_O, p3) / g['poss'] * 100
            hD = adj_pts(f, 'a', LAM_D, p3) / g['poss'] * 100
            aD = adj_pts(f, 'h', LAM_D, p3) / g['poss'] * 100
        else:
            hO = g['home_pts'] / g['poss'] * 100
            aO = g['away_pts'] / g['poss'] * 100
            hD = aO
            aD = hO

        h['o'] = ALPHA * hO + (1 - ALPHA) * h['o']
        h['d'] = ALPHA * hD + (1 - ALPHA) * h['d']
        h['p'] = PACE_A * g['pace'] + (1 - PACE_A) * h['p']
        a['o'] = ALPHA * aO + (1 - ALPHA) * a['o']
        a['d'] = ALPHA * aD + (1 - ALPHA) * a['d']
        a['p'] = PACE_A * g['pace'] + (1 - PACE_A) * a['p']
        h['n'] += 1
        a['n'] += 1
        league = ALPHA * ((hO + aO) / 2) + (1 - ALPHA) * league

        h['last_date'] = g['date']
        a['last_date'] = g['date']
        h['hist_o'].append(hO); h['hist_d'].append(hD); h['hist_p'].append(g['pace'])
        a['hist_o'].append(aO); a['hist_d'].append(aD); a['hist_p'].append(g['pace'])


df = pd.DataFrame(rows)
out = 'ml-model/features.csv'
df.to_csv(out, index=False)

print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {out}")
print(f"\nSeason breakdown:")
print(df['season'].value_counts().sort_index().to_string())
print(f"\nTarget summary:")
print(df[['margin', 'total']].describe().round(2).to_string())
print(f"\nNaN counts per column (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10).to_string())
print(f"\nSample row (first):")
print(df.iloc[0][['date','h_o','a_d','raw_margin','margin','h_rest','a_rest','h_o_roll5']].to_string())
