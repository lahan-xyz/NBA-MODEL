import json, argparse, inspect, time
from datetime import date, datetime, timedelta
from collections import defaultdict
from nba_api.stats.static import teams as static_teams

FMTS = ['%m/%d/%Y %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%dT%H:%M:%SZ','%Y-%m-%d','%m/%d/%Y']

def pdate(v):
    if not v: return None
    s = str(v).strip()
    for f in FMTS:
        try: return datetime.strptime(s, f).date()
        except ValueError: pass
    try: return datetime.fromisoformat(s.replace('Z','+00:00')).date()
    except Exception: return None

def cur_season():
    d = date.today(); y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{str(y+1)[2:]}"

ap = argparse.ArgumentParser()
ap.add_argument('--days', type=int, default=7)
ap.add_argument('--start', default=None)
ap.add_argument('--source', default='auto', choices=['auto','schedule','scoreboard'])
A = ap.parse_args()
start = datetime.strptime(A.start,'%Y-%m-%d').date() if A.start else date.today()
end = start + timedelta(days=A.days)

ratings = json.load(open('data/model.json'))['ratings']
by_name = {}
for full in ratings:
    by_name[full.lower()] = full
    by_name[full.split()[-1].lower()] = full
ID2NAME = {int(t['id']): t['full_name'] for t in static_teams.get_teams()}

def resolve(city, name, tid):
    c = [f'{city} {name}'.strip().lower(), str(name).lower()]
    try:
        full = ID2NAME.get(int(tid))
    except (TypeError, ValueError):
        full = None
    if full: c += [full.lower(), full.split()[-1].lower()]
    for k in c:
        if k in by_name: return by_name[k]
    return None

def from_schedule(season):
    from nba_api.stats.endpoints import scheduleleaguev2 as mod
    p = inspect.signature(mod.ScheduleLeagueV2.__init__).parameters
    kw = {'timeout': 60}
    for k, v in [('season', season), ('league_id','00')]:
        if k in p: kw[k] = v
    ep = mod.ScheduleLeagueV2(**kw)
    try: data = ep.get_dict()
    except Exception: data = json.loads(ep.get_json())
    sched = data.get('leagueSchedule', data)
    out = []
    for b in sched.get('gameDates', []):
        for g in b.get('games', []):
            gid = str(g.get('gameId',''))
            if gid.startswith('001'): continue          # preseason
            ht, at = g.get('homeTeam') or {}, g.get('awayTeam') or {}
            hid = ht.get('teamId', g.get('homeTeamId'))
            aid = at.get('teamId', g.get('awayTeamId'))
            if not hid or not aid: continue             # Cup slots with TBD teams
            h = resolve(ht.get('teamCity',''), ht.get('teamName',''), hid)
            a = resolve(at.get('teamCity',''), at.get('teamName',''), aid)
            if not h or not a: continue
            d = None
            for k in ('gameDateEst','gameDate','gameDateTimeEst','gameDateTimeUTC'):
                d = pdate(g.get(k))
                if d: break
            if d is None: d = pdate(b.get('gameDate'))
            if d is None: continue
            out.append({'game_id':gid,'date':d.isoformat(),'home':h,'away':a,
                        'status':g.get('gameStatus',1),
                        'neutral':bool(g.get('isNeutral',False))})
    return out

def from_scoreboard(s, e):
    from nba_api.stats.endpoints import scoreboardv2 as mod
    p = inspect.signature(mod.ScoreboardV2.__init__).parameters
    out, d = [], s
    while d <= e:
        kw = {'timeout': 60}
        for k, v in [('game_date', d.isoformat()), ('day_offset',0), ('league_id','00')]:
            if k in p: kw[k] = v
        df = mod.ScoreboardV2(**kw).game_header.get_data_frame()
        for _, r in df.iterrows():
            h, a = resolve('','',r['HOME_TEAM_ID']), resolve('','',r['VISITOR_TEAM_ID'])
            if not h or not a: continue
            out.append({'game_id':str(r['GAME_ID']),'date':d.isoformat(),'home':h,'away':a,
                        'status':int(r['GAME_STATUS_ID']),'neutral':False})
        d += timedelta(days=1); time.sleep(1.0)
    return out

season, parsed = cur_season(), []
if A.source in ('auto','schedule'):
    try:
        parsed = from_schedule(season)
        print(f'scheduleleaguev2: {len(parsed)} games in {season}')
    except Exception as ex:
        print('scheduleleaguev2 failed:', type(ex).__name__, ex)
if not parsed and A.source in ('auto','scoreboard'):
    print('falling back to scoreboardv2 (one request per date)...')
    parsed = from_scoreboard(start, end)
    print(f'scoreboardv2: {len(parsed)} games')
if not parsed:
    raise SystemExit('no fixtures retrieved from either source')

tl = defaultdict(set)
for g in json.load(open('data/games.json')):
    tl[g['home']].add(g['date']); tl[g['away']].add(g['date'])
for g in parsed:
    tl[g['home']].add(g['date']); tl[g['away']].add(g['date'])

def rest(team, d):
    past = sorted(x for x in tl[team] if x < d)
    if not past: return 7, 0
    n = max(0, min((date.fromisoformat(d)-date.fromisoformat(past[-1])).days-1, 7))
    return n, (1 if n == 0 else 0)

out = []
for g in parsed:
    if g['status'] == 3: continue
    if not (start.isoformat() <= g['date'] <= end.isoformat()): continue
    hr, hb = rest(g['home'], g['date']); ar, ab = rest(g['away'], g['date'])
    out.append({**g, 'home_rest':hr, 'away_rest':ar, 'home_b2b':hb, 'away_b2b':ab})
out.sort(key=lambda x: x['date'])

json.dump({'generated_at':datetime.now().isoformat(timespec='seconds'),
           'window':{'from':start.isoformat(),'to':end.isoformat()},
           'games':out}, open('data/fixtures.json','w'), indent=2)
print(f'{len(out)} fixtures {start} -> {end} -> data/fixtures.json')
print(f'  B2B: home {sum(g["home_b2b"] for g in out)}, away {sum(g["away_b2b"] for g in out)}'
      f'   neutral-site {sum(1 for g in out if g["neutral"])}')
