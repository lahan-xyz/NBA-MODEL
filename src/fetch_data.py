import json
import time
import math
from nba_api.stats.endpoints import leaguedashteamstats

def fetch_team_stats():
    print("Fetching NBA team stats for the last 10 seasons... (This will take a few minutes)")
    
    # 2016-17 to 2025-26
    seasons = [f"{year}-{str(year+1)[2:]}" for year in range(2016, 2026)]
    team_history = {}
    
    for season in seasons:
        print(f"Fetching {season}...")
        try:
            # CHANGE: We use 'Advanced' to get ORtg, DRtg, and Pace
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season, 
                measure_type_detailed_defense='Advanced', 
                timeout=30
            )
            stats_df = team_stats.get_data_frames()[0]
            
            # Print columns on the first run just to verify we got the right data
            if season == seasons[0]:
                print(f"✅ Verified columns found: {list(stats_df.columns)}")
            
            for _, row in stats_df.iterrows():
                team_name = row['TEAM_NAME']
                if team_name not in team_history:
                    team_history[team_name] = {'off': [], 'def': [], 'pace': []}
                
                # Extract metrics
                off = row.get('OFF_RATING', 100)
                defr = row.get('DEF_RATING', 100)
                pace = row.get('PACE', 100)
                
                # Properly check for NaNs and ensure they are valid numbers
                if not math.isnan(off) and off > 0: team_history[team_name]['off'].append(float(off))
                if not math.isnan(defr) and defr > 0: team_history[team_name]['def'].append(float(defr))
                if not math.isnan(pace) and pace > 0: team_history[team_name]['pace'].append(float(pace))
                
        except Exception as e:
            print(f"Error fetching {season}: {e}")
            continue
            
        # Sleep to avoid hitting NBA API rate limits
        time.sleep(1.5) 
        
    # Calculate 10-year averages for each team
    data = {}
    for team, stats in team_history.items():
        if stats['off'] and stats['def'] and stats['pace']:
            data[team] = {
                'off_rating': sum(stats['off']) / len(stats['off']),
                'def_rating': sum(stats['def']) / len(stats['def']),
                'pace': sum(stats['pace']) / len(stats['pace']),
                'seasons_played': len(stats['off'])
            }
            
    with open('data/team_stats.json', 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"\n✅ Successfully saved 10-year historical averages for {len(data)} teams.")

if __name__ == "__main__":
    fetch_team_stats()
