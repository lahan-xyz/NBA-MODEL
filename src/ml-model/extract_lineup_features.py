"""Walk-forward lineup features from player_logs.json.

Per team, per game, computes (all as-of tip, past games only):
  player_strength    Σ (ewma_min × ewma_ppm)      expected points from available players
  min_coverage       Σ ewma_min                   total projected minutes (~240 if healthy)
  top_player_min     max ewma_min                 is the #1 option playing?
  rotation_depth     count(ewma_min > 15)         is the bench thin?

Output: ml-model/lineup_features.csv
"""
import json
import numpy as np
import pandas as pd

DECAY_MIN = 0.15   # faster — availability reflects recent injury/return
DECAY_PPM = 0.06   # slower — player quality is stable

def parse_min(s):
    if isinstance(s, (int, float)): return float(s)
    if not s: return 0.0
    s = str(s)
    if ':' in s:
        mm, ss = s.split(':')[:2]
        return float(mm) + float(ss) / 60.0
    try: return float(s)
    except: return 0.0

ABBR = {
    'Atlanta Hawks':'ATL','Boston Celtics':'BOS','Brooklyn Nets':'BKN',
    'Charlotte Hornets':'CHA','Chicago Bulls':'CHI','Cleveland Cavaliers':'CLE',
    'Dallas Mavericks':'DAL','Denver Nuggets':'DEN','Detroit Pistons':'DET',
    'Golden State Warriors':'GSW','Houston Rockets':'HOU','Indiana Pacers':'IND',
    'LA Clippers':'LAC','Los Angeles Clippers':'LAC',
    'Los Angeles Lakers':'LAL','Memphis Grizzlies':'MEM','Miami Heat':'MIA',
    'Milwaukee Bucks':'MIL','Minnesota Timberwolves':'MIN',
    'New Orleans Pelicans':'NOP','New York Knicks':'NYK',
    'Oklahoma City Thunder':'OKC','Orlando Magic':'ORL',
    'Philadelphia 76ers':'PHI','Phoenix Suns':'PHX',
    'Portland Trail Blazers':'POR','Sacramento Kings':'SAC',
    'San Antonio Spurs':'SAS','Toronto Raptors':'TOR',
    'Utah Jazz':'UTA','Washington Wizards':'WAS',
}

rows = json.load(open('ml-model/player_logs.json'))
print(f"Loaded {len(rows)} player-game rows")

by_game = {}
for r in rows:
    gid = r['GAME_ID']
    team = r['TEAM_ABBREVIATION']
    pid = r['PLAYER_ID']
    m = parse_min(r.get('MIN'))
    pts = float(r.get('PTS') or 0)
    by_game.setdefault(gid, {}).setdefault(team, []).append((pid, m, pts))

games = json.load(open('data/games.json'))
games_sorted = sorted(games, key=lambda g: (g['date'], g['game_id']))
existing = set(pd.read_csv('ml-model/features.csv', dtype={'game_id': str})['game_id'])

state = {}  # team_abbr -> pid -> {ewma_min, ewma_ppm, n}
def getp(t, pid):
    return state.setdefault(t, {}).setdefault(pid, {'ewma_min':0.0,'ewma_ppm':0.0,'n':0})

out = []
for g in games_sorted:
    gid = g['game_id']
    h, a = ABBR.get(g['home']), ABBR.get(g['away'])
    if not (h and a): continue

    if gid in existing:
        def team_feats(t):
            ps = state.get(t, {})
            return {
                'strength': sum(p['ewma_min'] * p['ewma_ppm'] for p in ps.values()),
                'min_cov':  sum(p['ewma_min'] for p in ps.values()),
                'top_min':  max((p['ewma_min'] for p in ps.values()), default=0.0),
                'depth':    sum(1 for p in ps.values() if p['ewma_min'] > 15),
            }
        hf, af = team_feats(h), team_feats(a)
        out.append({
            'game_id': gid,
            'h_player_strength': hf['strength'], 'a_player_strength': af['strength'],
            'h_min_coverage':    hf['min_cov'],  'a_min_coverage':    af['min_cov'],
            'h_top_player_min':  hf['top_min'],  'a_top_player_min':  af['top_min'],
            'h_rotation_depth':  hf['depth'],    'a_rotation_depth':  af['depth'],
            'player_strength_diff': hf['strength'] - af['strength'],
            'min_coverage_diff':    hf['min_cov']  - af['min_cov'],
        })

    # update AFTER feature extraction
    if gid in by_game:
        for t in (h, a):
            players = by_game[gid].get(t, [])
            appeared = set()
            for pid, m, pts in players:
                ps = getp(t, pid)
                ppm = pts / m if m > 0 else 0.0
                if ps['n'] == 0:
                    ps['ewma_min'], ps['ewma_ppm'] = m, ppm
                else:
                    ps['ewma_min'] = DECAY_MIN * m + (1 - DECAY_MIN) * ps['ewma_min']
                    if m > 0:
                        ps['ewma_ppm'] = DECAY_PPM * ppm + (1 - DECAY_PPM) * ps['ewma_ppm']
                ps['n'] += 1
                appeared.add(pid)
            # decay non-appearers so injured players fade out
            for pid, ps in state.get(t, {}).items():
                if pid not in appeared and ps['n'] > 0:
                    ps['ewma_min'] *= (1 - DECAY_MIN)

df = pd.DataFrame(out)
df.to_csv('ml-model/lineup_features.csv', index=False)
print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> ml-model/lineup_features.csv")
print(df.head(3).to_string())
