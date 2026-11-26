import json, statistics as st
from collections import Counter

g = json.load(open('data/games.json'))
te = [x for x in g if x['season'] == '2025-26' and not x['ot']]
m = [x['home_pts'] - x['away_pts'] for x in te]

print(f'n={len(te)}  mean={st.mean(m):+.2f}  sd={st.pstdev(m):.2f}')
print(f'mean |margin| = {st.mean([abs(v) for v in m]):.2f}')
print(f'MAE of always +2.5 = {st.mean([abs(2.5-v) for v in m]):.2f}')

ids = Counter(x['game_id'] for x in g)
print(f'\nduplicate game_ids: {sum(1 for v in ids.values() if v > 1)}')

bad = [x for x in g if min(x['home_pts'], x['away_pts']) < 60
                    or max(x['home_pts'], x['away_pts']) > 200
                    or abs(x['home_pts'] - x['away_pts']) > 60]
print(f'suspicious scores : {len(bad)}')
for x in bad[:12]:
    print(f"  {x['date']} {x['home'][:18]:18} {x['home_pts']:.0f} - "
          f"{x['away_pts']:.0f} {x['away'][:18]:18} poss={x['poss']}")

cnt = Counter()
for x in te:
    cnt[x['home']] += 1; cnt[x['away']] += 1
print(f'\nteam game counts: min={min(cnt.values())} max={max(cnt.values())} (expect ~82)')
print('outliers:', [(t, c) for t, c in cnt.items() if not 78 <= c <= 86])
