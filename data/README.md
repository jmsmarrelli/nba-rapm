# Data

Two files per season in `data/raw/`:

* `play_by_play_<season>.csv.gz` — every logged event, including substitutions.
* `player_box_<season>.csv.gz` — box scores, used for rosters, starters, and the
  minutes check that validates the reconstruction.

`<season>` is the year the season ended: 2023 is 2022-23.

## Getting the files

Both come from [sportsdataverse/hoopR-data](https://github.com/sportsdataverse/hoopR-data),
a mirror of ESPN's public feeds.

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/sportsdataverse/hoopR-data.git /tmp/hoopR-data
cd /tmp/hoopR-data
git sparse-checkout set --no-cone "nba/pbp/csv/*" "nba/player_box/csv/*"
git checkout
cd -
for y in $(seq 2014 2023); do
  cp /tmp/hoopR-data/nba/pbp/csv/play_by_play_${y}.csv.gz data/raw/
  cp /tmp/hoopR-data/nba/player_box/csv/player_box_${y}.csv.gz data/raw/
done
```

About 250 MB for ten seasons. `data/raw/` and `data/processed/` are gitignored;
everything the analysis cites is written to `output/` and committed.

## Columns used

From the play-by-play:

| Column | Used for |
|---|---|
| `game_id`, `game_play_number` | ordering, and the cross-validation folds |
| `period_number` | period-start lineup inference |
| `type_text` | substitutions, shot and rebound types, turnovers |
| `team_id`, `home_team_id`, `away_team_id` | possession attribution |
| `athlete_id_1`, `athlete_id_2` | on a substitution, who entered and who left |
| `athlete_id_1..3` | evidence that a player was on the floor |
| `start_game_seconds_remaining` | floor time, for the box score check |
| `scoring_play`, `score_value`, `shooting_play` | points and possession markers |

From the box score: `athlete_id`, `team_id`, `starter`, `minutes`,
`did_not_play`, and the counting stats used in stage 4.

## Coverage

The play-by-play mirror is not complete. It carries 11,913 regular season games
across these ten seasons before any of this project's own filters run — short of
the scheduled total in most seasons — and `output/reconstruction_audit.csv`
records what happens to them after. Seasons 2017-18 through 2019-20 have noticeably thinner substitution
logging than the rest.
