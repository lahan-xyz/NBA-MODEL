"""Fetch per-player per-game stats using PlayerGameLogs (bulk, paginated).
Much faster than per-game box scores. Output: ml-model/player_logs.json"""
import json, time
from pathlib import Path
from nba_api.stats.endpoints import playergamelogs

OUT = Path('ml-model/player_logs.json')
CACHE = Path('ml-model/player_logs_cache')
SEASONS = ['2023-24', '2024-25', '2025-26']
CACHE.mkdir(exist_ok=True)

all_rows = []
for season in SEASONS:
    cache = CACHE / f'{season}.json'
    if cache.exists():
        rows = json.load(open(cache))
        print(f"{season}: cached ({len(rows)} rows)")
    else:
        print(f"{season}: fetching (may take 60-120s)...")
        try:
            resp = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable='Regular Season',
                timeout=120,
            )
            df = resp.get_data_frames()[0]
            cols = ['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION',
                    'GAME_DATE', 'MIN', 'PTS', 'REB', 'AST']
            rows = df[cols].to_dict('records')
            json.dump(rows, open(cache, 'w'))
            print(f"{season}: {len(rows)} rows cached")
        except Exception as e:
            print(f"{season}: FAILED — {type(e).__name__}: {e}")
            continue
    all_rows.extend(rows)

json.dump(all_rows, open(OUT, 'w'))
print(f"\nWrote {len(all_rows)} total rows -> {OUT}")
