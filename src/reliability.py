import json, math
from collections import defaultdict

G = json.load(open('data/factors.json'))

def factors(rec):
    res = {}
    for me, opp in (('h','a'), ('a','h')):
        fga, fta, tov = rec[f'{me}_FGA'], rec[f'{me}_FTA'], rec[f'{me}_TOV']
        ofga, ofta, otov = rec[f'{opp}_FGA'], rec[f'{opp}_FTA'], rec[f'{opp}_TOV']
        poss  = fga  - rec[f'{me}_OREB']  + tov  + 0.44*fta
        oposs = ofga - rec[f'{opp}_OREB'] + otov + 0.44*ofta
        if min(poss, oposs, fga, ofga) <= 0: return None
        p = (poss + oposs) / 2
        name = rec['home'] if me == 'h' else rec['away']
        res[name] = {
            'ORtg':      rec[f'{me}_PTS'] / p * 100,
            'DRtg':      rec[f'{opp}_PTS'] / p * 100,
            'eFG%':      (rec[f'{me}_FGM'] + 0.5*rec[f'{me}_FG3M']) / fga,
            'opp_eFG%':  (rec[f'{opp}_FGM'] + 0.5*rec[f'{opp}_FG3M']) / ofga,
            'own_3P%':   rec[f'{me}_FG3M'] / max(rec[f'{me}_FG3A'], 1),
            'opp_3P%':   rec[f'{opp}_FG3M'] / max(rec[f'{opp}_FG3A'], 1),
            'TOV%':      tov / (fga + 0.44*fta + tov),
            'opp_TOV%':  otov / (ofga + 0.44*ofta + otov),
            'OREB%':     rec[f'{me}_OREB'] / max(rec[f'{me}_OREB'] + rec[f'{opp}_DREB'], 1),
            'FTr':       fta / fga,
            'pace':      p,
        }
    return res

def pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    sx = math.sqrt(sum((v-mx)**2 for v in x))
    sy = math.sqrt(sum((v-my)**2 for v in y))
    if sx == 0 or sy == 0: return 0.0
    return sum((x[i]-mx)*(y[i]-my) for i in range(n)) / (sx*sy)

series = defaultdict(lambda: defaultdict(list))
for rec in G:
    f = factors(rec)
    if f is None: continue
    for team, m in f.items():
        for k, v in m.items():
            series[(rec['season'], team)][k].append(v)

keys = list(next(iter(series.values())).keys())
acc = defaultdict(list)
for season in ['2023-24', '2024-25', '2025-26']:
    teams = [t for (s, t) in series if s == season]
    for k in keys:
        A, B = [], []
        for t in teams:
            v = series[(season, t)][k]
            if len(v) < 40: continue
            A.append(sum(v[0::2]) / len(v[0::2]))
            B.append(sum(v[1::2]) / len(v[1::2]))
        if len(A) >= 20:
            acc[k].append(pearson(A, B))

print('Split-half reliability by season (odd vs even games, 30 teams)')
print('metric        23-24   24-25   25-26    mean')
rows = sorted(keys, key=lambda k: -sum(acc[k])/max(len(acc[k]),1))
for k in rows:
    v = acc[k]
    mean = sum(v)/len(v) if v else 0
    cells = '  '.join(f'{x:+.2f}' for x in v)
    print(f'{k:12} {cells}   {mean:+.3f}')
