"""Paths, constants and the handful of knobs the pipeline exposes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
OUT = ROOT / "output"

for d in (RAW, PROC, FIG, OUT):
    d.mkdir(parents=True, exist_ok=True)

# Season label is the year the season ends: 2023 == 2022-23. The public
# play-by-play mirror this project reads stops there.
FIRST_SEASON = 2014
LAST_SEASON = 2023
SEASONS = list(range(FIRST_SEASON, LAST_SEASON + 1))

# The headline model pools three seasons. One season of stints is not enough to
# separate teammates who never play apart; three is the usual compromise between
# sample and staleness.
WINDOW = 3
HEADLINE_END = LAST_SEASON

# A game is kept only if reconstructed floor time agrees with the box score.
MAX_MEAN_MINUTE_ERROR = 1.5     # minutes, averaged over players in the game
MAX_PLAYER_MINUTE_ERROR = 6.0   # minutes, worst single player

MIN_POSS_PLAYER = 500           # possessions to appear in a leaderboard
SEED = 20240117
