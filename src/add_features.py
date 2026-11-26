import json
from datetime import date
from collections import defaultdict

SRC = 'data/games.json'

def d(s):
    y, m, dd = s.split('-')
    return date(int(y), int(m), int(dd))

games = json.load(open(SRC))
games.sort(key=lambda g: (g['date'], g['game_id']))

hist, season = {}, None
for g in games:
    if g['season'] != season:
        season, hist = g['season'], {}
    today = d(g['date'])
    for side in ('home', 'away'):
        past = hist.setdefault(g[side], [])
        if not past:
            rest, b2b, t34 = 7, 0, 0
        else:
            rest = max(0, min((today - past[-1]).days - 1, 7))
            b2b = 1 if rest == 0 else 0
            t34 = 1 if sum(1 for p in past if (today - p).days <= 3) >= 2 else 0
        g[f'{side}_rest'] = rest
        g[f'{side}_b2b'] = b2b
        g[f'{side}_3in4'] = t34
        past.append(today)
        if len(past) > 12:
            past.pop(0)

with open(SRC, 'w') as f:
    json.dump(games, f)

b = defaultdict(list)
for g in games:
    if g.get('ot'):
        continue
    k = max(-3, min(3, g['home_rest'] - g['away_rest']))
    b[k].append(g['home_pts'] - g['away_pts'])

print('Rest advantage -> mean home margin (raw signal, all seasons)')
print('restDiff    n    meanMargin')
for k in sorted(b):
    v = b[k]
    print(f'{k:+3d}     {len(v):5d}    {sum(v)/len(v):+6.2f}')
print(f'\nEnriched {len(games)} games.')
