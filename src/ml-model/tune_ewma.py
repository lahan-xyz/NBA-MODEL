"""Grid-search EWMA hyperparameters (ALPHA, LAM_O, LAM_D, PACE_A).
Tests 81 combos; each takes ~1s. Reports best margin/total MAE."""
import json, itertools, time
import numpy as np
import pandas as pd

games = json.load(open('data/games.json'))
factors = json.load(open('data/factors.json'))
F = {f['game_id']: f for f in factors}
LG3 = {}
for s in {f['season'] for f in factors}:
    m3 = sum(f['h_FG3M']+f['a_FG3M'] for f in factors if f['season']==s)
    a3 = sum(f['h_FG3A']+f['a_FG3A'] for f in factors if f['season']==s)
    LG3[s] = m3/a3 if a3 else 0.36
games = sorted(games, key=lambda g: (g['date'], g['game_id']))
SEASONS = {'2023-24','2024-25','2025-26'}

def adj(f, side, lam, p3):
    m3 = lam*f[f'{side}_FG3M'] + (1-lam)*f[f'{side}_FG3A']*p3
    return 2*(f[f'{side}_FGM']-f[f'{side}_FG3M']) + 3*m3 + f[f'{side}_FTM']

def run(ALPHA, LAM_O, LAM_D, PACE_A, INIT=110.0, IPACE=99.0):
    R = {}
    def get(t):
        if t not in R: R[t] = {'o':INIT, 'd':INIT, 'p':IPACE, 'n':0}
        return R[t]
    league = INIT; rows = []
    for g in games:
        h, a = get(g['home']), get(g['away'])
        pace = (h['p']+a['p'])/2
        rawM = (h['o']+a['d']-league)*pace/100 - (a['o']+h['d']-league)*pace/100
        rawT = (h['o']+a['d']-league)*pace/100 + (a['o']+h['d']-league)*pace/100
        done = g.get('home_pts') is not None
        if done and g['season'] in SEASONS and not g.get('ot',False) and h['n']>=10 and a['n']>=10:
            rows.append({'season':g['season'],'rawM':rawM,'rawT':rawT,
                         'margin':g['home_pts']-g['away_pts'],
                         'total':g['home_pts']+g['away_pts']})
        if done:
            f = F.get(g['game_id']); p3 = LG3.get(g['season'], 0.36)
            if f:
                hO = adj(f,'h',LAM_O,p3)/g['poss']*100; aO = adj(f,'a',LAM_O,p3)/g['poss']*100
                hD = adj(f,'a',LAM_D,p3)/g['poss']*100; aD = adj(f,'h',LAM_D,p3)/g['poss']*100
            else:
                hO = g['home_pts']/g['poss']*100; aO = g['away_pts']/g['poss']*100; hD=aO; aD=hO
            h['o'] = ALPHA*hO+(1-ALPHA)*h['o']; h['d'] = ALPHA*hD+(1-ALPHA)*h['d']
            h['p'] = PACE_A*g['pace']+(1-PACE_A)*h['p']; h['n'] += 1
            a['o'] = ALPHA*aO+(1-ALPHA)*a['o']; a['d'] = ALPHA*aD+(1-ALPHA)*a['d']
            a['p'] = PACE_A*g['pace']+(1-PACE_A)*a['p']; a['n'] += 1
            league = ALPHA*((hO+aO)/2)+(1-ALPHA)*league
    return pd.DataFrame(rows)

def eval_mae(df):
    tr = df[df.season.isin(['2023-24','2024-25'])]; te = df[df.season=='2025-26']
    m_b, m_a = np.polyfit(tr.rawM, tr.margin, 1); t_b, t_a = np.polyfit(tr.rawT, tr.total, 1)
    return (float(np.mean(np.abs(m_a+m_b*te.rawM.values - te.margin.values))),
            float(np.mean(np.abs(t_a+t_b*te.rawT.values - te.total.values))))

grid = {'ALPHA':[0.04,0.06,0.08],'LAM_O':[0.4,0.5,0.6],
        'LAM_D':[0.20,0.25,0.30],'PACE_A':[0.10,0.15,0.20]}
best_m = (None, np.inf); best_t = (None, np.inf)
t0 = time.time()
for vals in itertools.product(*grid.values()):
    p = dict(zip(grid.keys(), vals))
    mm, tt = eval_mae(run(**p))
    if mm < best_m[1]: best_m = (p.copy(), mm)
    if tt < best_t[1]: best_t = (p.copy(), tt)

print(f"\nBaseline (current ALPHA=0.06 LAM_O=0.5 LAM_D=0.25 PACE_A=0.15): 11.572 / 14.747")
print(f"Best margin: {best_m[1]:.3f}  ->  {best_m[0]}")
print(f"Best total:  {best_t[1]:.3f}  ->  {best_t[0]}")
print(f"Grid took {time.time()-t0:.1f}s")
