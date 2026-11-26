import json, os
from datetime import date
from fetch_games import fetch_season, build

KEEP = ['FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','TOV','PTS']

def cur_season():
    d = date.today()
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{str(y+1)[2:]}"

def num(x):
    try:
        f = float(x); return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0

def factors_from(rows, season):
    byg = {}
    for r in rows: byg.setdefault(str(r['GAME_ID']), []).append(r)
    out = []
    for gid, rs in byg.items():
        if len(rs) != 2: continue
        h = a = None
        for r in rs:
            m = str(r.get('MATCHUP',''))
            if 'vs.' in m: h = r
            elif '@' in m: a = r
        if h is None or a is None: continue
        rec = {'game_id': gid, 'season': season, 'date': str(h['GAME_DATE'])[:10],
               'home': h['TEAM_NAME'], 'away': a['TEAM_NAME']}
        for side, r in (('h',h), ('a',a)):
            for k in KEEP: rec[f'{side}_{k}'] = num(r.get(k))
        out.append(rec)
    return out

REST = ('_rest','_b2b','_3in4')
def core(r): return {k:v for k,v in r.items() if not k.endswith(REST)}

def merge(path, new):
    old = json.load(open(path)) if os.path.exists(path) else []
    idx = {r['game_id']: r for r in old}
    added = updated = 0
    for r in new:
        prev = idx.get(r['game_id'])
        if prev is None: added += 1
        elif core(prev) != core(r): updated += 1
        idx[r['game_id']] = r
    allr = sorted(idx.values(), key=lambda x: (x['date'], x['game_id']))
    json.dump(allr, open(path,'w'))
    return added, updated, len(allr)

s = cur_season()
print('refreshing', s, flush=True)
rows = fetch_season(s)
g, f = build(rows, s), factors_from(rows, s)
a1,u1,t1 = merge('data/games.json', g)
a2,u2,t2 = merge('data/factors.json', f)
print(f'  games.json   +{a1} new, {u1} revised, {t1} total')
print(f'  factors.json +{a2} new, {u2} revised, {t2} total')
