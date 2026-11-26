import json, os, time, inspect
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog

SEASONS = ['2023-24', '2024-25', '2025-26', '2026-27']
OUT = 'data/games.json'

def fetch_season(season):
    # Only pass kwargs this nba_api version actually supports
    params = inspect.signature(leaguegamelog.LeagueGameLog.__init__).parameters
    kw = {'season': season, 'timeout': 60}
    for k, v in [('season_type_all_star', 'Regular Season'),
                 ('player_or_team_abbreviation', 'T'),
                 ('direction', 'ASC')]:
        if k in params:
            kw[k] = v
    return leaguegamelog.LeagueGameLog(**kw).get_data_frames()[0].to_dict('records')

def num(x):
    try:
        f = float(x)
        return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0

def norm_date(s):
    s = str(s).strip()
    if 'T' in s:
        return s.split('T')[0]
    try:
        return datetime.strptime(s, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        pass
    try:
        return datetime.strptime(s, '%b %d, %Y').strftime('%Y-%m-%d')
    except ValueError:
        return s

def build(rows, season):
    by_game = {}
    for r in rows:
        by_game.setdefault(str(r['GAME_ID']), []).append(r)
    out = []
    for gid, rs in by_game.items():
        if len(rs) != 2:
            continue
        home = away = None
        for r in rs:
            m = str(r.get('MATCHUP', ''))
            if 'vs.' in m:
                home = r
            elif '@' in m:
                away = r
        if home is None or away is None:
            continue

        def poss(r):
            return num(r['FGA']) - num(r['OREB']) + num(r['TOV']) + 0.44 * num(r['FTA'])

        minutes = num(home.get('MIN')) or 240.0
        if minutes < 100:      # some versions report 48 instead of 240
            minutes *= 5.0
        p = (poss(home) + poss(away)) / 2.0
        if p <= 0:
            continue
        out.append({
            'game_id': gid,
            'date': norm_date(home['GAME_DATE']),
            'season': season,
            'home': home['TEAM_NAME'],
            'away': away['TEAM_NAME'],
            'home_pts': num(home['PTS']),
            'away_pts': num(away['PTS']),
            'poss': round(p, 2),
            'pace': round(p * 240.0 / minutes, 2),
            'ot': minutes > 240
        })
    return out

def main():
    os.makedirs('data', exist_ok=True)
    all_games = []
    for s in SEASONS:
        print(f'Fetching {s} ...', flush=True)
        try:
            g = build(fetch_season(s), s)
            print(f'  {len(g)} games')
            all_games += g
        except Exception as e:
            print(f'  skipped {s}: {e}')
        time.sleep(1.5)
    all_games.sort(key=lambda x: (x['date'], x['game_id']))
    with open(OUT, 'w') as f:
        json.dump(all_games, f)
    print(f'\nSaved {len(all_games)} games to {OUT}')

if __name__ == '__main__':
    main()
