"""Counting possessions from the log rather than estimating them.

The usual shortcut is FGA + 0.44*FTA - OREB + TOV. It is a decent season-level
approximation and a poor stint-level one, because the 0.44 is a league-average
guess about which free throws end a trip and stints are short enough for that
guess to matter.

Since the log is right there, possessions are counted directly. Every event that
unambiguously identifies which team has the ball is marked with that team; the
sequence of marks is then compressed into runs, and each run is one possession.
Ambiguous events - fouls, timeouts, jump balls, substitutions - are skipped
rather than guessed at, which is safe: dropping a marker can only merge two
possessions that were already adjacent and identical, and adjacent identical
marks collapse anyway.

This handles the case that trips up the formula, an and-one: made field goal and
free throw are both marked to the same team, they are adjacent, and they
collapse into the single possession they actually were.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TURNOVER_PAT = (r"Turnover|Traveling|Double Dribble|Palming|Discontinue Dribble|"
                r"Second Violation|Step Out of Bounds|Offensive Foul")


def offensive_team(pbp: pd.DataFrame) -> pd.Series:
    """Team in possession, where the event says so, NaN where it does not."""
    tt = pbp["type_text"].fillna("")
    team = pbp["team_id"]
    other = np.where(team == pbp["home_team_id"], pbp["away_team_id"], pbp["home_team_id"])

    off = pd.Series(np.nan, index=pbp.index, dtype="float64")
    # The offence shoots, and shoots free throws.
    shoot = pbp["shooting_play"].fillna(False).astype(bool) | tt.str.startswith("Free Throw")
    off = off.mask(shoot, team)
    # The offence rebounds its own miss; the defence rebounding ends the trip.
    off = off.mask(tt.eq("Offensive Rebound"), team)
    off = off.mask(tt.eq("Defensive Rebound"), pd.Series(other, index=pbp.index))
    # The offence turns it over.
    off = off.mask(tt.str.contains(TURNOVER_PAT, regex=True, na=False), team)
    return off


def possession_runs(off: pd.Series) -> pd.Series:
    """Label each marked event with a possession number.

    Unmarked events inherit the possession they sit inside, so a foul in the
    middle of a trip lands in the right stint bucket.
    """
    marked = off.dropna()
    change = marked.ne(marked.shift())
    poss_id = change.cumsum()
    return poss_id.reindex(off.index).ffill()
