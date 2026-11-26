import json, math
from collections import defaultdict

G = json.load(open('data/factors.json'))
SEASONS = ['2023-24', '2024-25']
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]

lg = {}
for s in SEASONS:
    rows = [r for r in G if r['season'] == s]
    m = sum(r['h_FG3M'] + r['a_FG3M'] for r in rows)
    a = sum(r['h_FG3A'] + r['a_FG3A'] for r in rows)
    lg[s] = m / a if a else 0.36
    print(f'{s} league 3P% = {lg[s]:.4f}')

series = defaultdict(lambda: defaultdict(list))
for rec in G:
    s = rec['season']
    if s not in lg: continue
    p3 = lg[s]
    poss = {}
    ok = True
    for side in ('h', 'a'):
        fga, fta, tov = rec[f'{side}_FGA'], rec[f'{side}_FTA'], rec[f'{side}_TOV']
        poss[side] = fga - rec[f'{side}_OREB'] + tov + 0.44 * fta
        if poss[side] <= 0 or fga <= 0: ok = False
    if not ok: continue
    pace = (poss['h'] + poss['a']) / 2
    for me, opp in (('h', 'a'), ('a', 'h')):
        name = rec['home'] if me == 'h' else rec['away']
        for lam in LAMS:
            m3 = lam*rec[f'{me}_FG3M'] + (1-lam)*rec[f'{me}_FG3A']*p3
            pts = 2*(rec[f'{me}_FGM'] - rec[f'{me}_FG3M']) + 3*m3 + rec[f'{me}_FTM']
            series[(s, name)][f'O{lam}'].append(pts / pace * 100)
            o3 = lam*rec[f'{opp}_FG3M'] + (1-lam)*rec[f'{opp}_FG3A']*p3
            opts = 2*(rec[f'{opp}_FGM'] - rec[f'{opp}_FG3M']) + 3*o3 + rec[f'{opp}_FTM']
            series[(s, name)][f'D{lam}'].append(opts / pace * 100)

def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((v-mx)**2 for v in x))
    sy = math.sqrt(sum((v-my)**2 for v in y))
    if sx == 0 or sy == 0: return 0.0
    return sum((x[i]-mx)*(y[i]-my) for i in range(n))/(sx*sy)

def rel(key):
    out = []
    for s in SEASONS:
        A, B = [], []
        for (ss, t), d in series.items():
            if ss != s: continue
            v = d[key]
            if len(v) < 40: continue
            A.append(sum(v[0::2])/len(v[0::2]))
            B.append(sum(v[1::2])/len(v[1::2]))
        if len(A) >= 20: out.append(pearson(A, B))
    return sum(out)/len(out) if out else 0.0

print('\nlambda = share of ACTUAL 3P makes retained (0 = all expectation)')
print('lambda    ORtg rel   DRtg rel')
best = {}
for lam in LAMS:
    ro, rd = rel(f'O{lam}'), rel(f'D{lam}')
    print(f' {lam:.2f}      {ro:+.3f}      {rd:+.3f}')
    best[lam] = (ro, rd)

bo = max(LAMS, key=lambda l: best[l][0])
bd = max(LAMS, key=lambda l: best[l][1])
sb = lambda r: 2*r/(1+r)
print(f'\nBest offense lambda {bo:.2f}  rel {best[bo][0]:+.3f}  (full-season {sb(best[bo][0]):.3f})')
print(f'Best defense lambda {bd:.2f}  rel {best[bd][1]:+.3f}  (full-season {sb(best[bd][1]):.3f})')
print(f'vs current (lambda 1.00): O {best[1.0][0]:+.3f}  D {best[1.0][1]:+.3f}')
