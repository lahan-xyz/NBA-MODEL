import pandas as pd
base = pd.read_csv('ml-model/features.csv', dtype={'game_id': str})
lineup = pd.read_csv('ml-model/lineup_features.csv', dtype={'game_id': str})
merged = base.merge(lineup, on='game_id', how='left')
merged.to_csv('ml-model/features_v2.csv', index=False)
print(f"Merged: {len(merged)} rows x {len(merged.columns)} cols -> ml-model/features_v2.csv")
print(f"Games missing lineup data: {merged['h_player_strength'].isna().sum()}")
