import json, time, inspect
from nba_api.stats.endpoints import leaguegamelog

SEASONS = ['2023-24', '2024-25', '2025-26', '2026-27']
KEEP = ['FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','TOV','PTS']

def num(x):
    try:
        f = float(x); return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0

def fetch(season):
    p = inspect.signature(leaguegamelog.LeagueGameLog.__init__).parameters
    kw = {'season': season, 'timeout': 60}
    for k, v in [('season_type_all_star','Regular Season'),
                 ('player_or_team_abbreviation','T')]:
        if k in p: kw[k] = v
    return leaguegamelog.LeagueGameLog(**kw).get_data_frames()[0].to_dict('records')

out = []
for s in SEASONS:
    print('fetching', s, flush=True)
    try:
        rows = fetch(s)
    except Exception as e:
        print('  skipped:', e); continue
    byg = {}
    for r in rows:
        byg.setdefault(str(r['GAME_ID']), []).append(r)
    n = 0
    for gid, rs in byg.items():
        if len(rs) != 2: continue
        h = a = None
        for r in rs:
            m = str(r.get('MATCHUP',''))
            if 'vs.' in m: h = r
            elif '@' in m: a = r
        if h is None or a is None: continue
        rec = {'game_id': gid, 'season': s, 'date': str(h['GAME_DATE'])[:10],
               'home': h['TEAM_NAME'], 'away': a['TEAM_NAME']}
        for side, r in (('h', h), ('a', a)):
            for k in KEEP:
                rec[f'{side}_{k}'] = num(r.get(k))
        out.append(rec); n += 1
    print(' ', n, 'games')
    time.sleep(1.5)

out.sort(key=lambda x: (x['date'], x['game_id']))
json.dump(out, open('data/factors.json','w'))
print('saved', len(out), 'games to data/factors.json')
