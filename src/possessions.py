"""Event-marked offensive sequences: an approximation, not certified possessions.

Periods reset the sequence. Special retained-ball free throws do not identify
control and their points are excluded from the modeled scoring outcome. Ambiguous
jump balls and missing events can still merge sequences; results use proxy units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TURNOVER_PAT = (r"Turnover|Traveling|Double Dribble|Palming|Discontinue Dribble|"
                r"Second Violation|Step Out of Bounds|Offensive Foul")


def special_free_throw(pbp: pd.DataFrame) -> pd.Series:
    text = pbp["type_text"].fillna("")
    if "text" in pbp:
        text = text + " " + pbp["text"].fillna("")
    return text.str.contains(r"free throw", case=False) & text.str.contains(
        r"technical|flagrant|clear[ -]?path|away[ -]?from[ -]?play|take foul", case=False)


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
    return off.mask(special_free_throw(pbp))


def possession_runs(off: pd.Series, period: pd.Series) -> pd.Series:
    """Number marked runs, resetting at period boundaries without cross-period fill."""
    marked = off.dropna()
    p = period.reindex(marked.index)
    change = marked.ne(marked.shift()) | p.ne(p.shift())
    ids = change.cumsum().reindex(off.index)
    return ids.groupby(period).ffill()


def scoring_points(pbp: pd.DataFrame) -> pd.Series:
    """Made free throws are one point even when the mirror's score_value is zero."""
    made = pbp["scoring_play"].fillna(False).astype(bool)
    ft = pbp["type_text"].fillna("").str.contains("free throw", case=False)
    points = pd.to_numeric(pbp["score_value"], errors="coerce").fillna(0).where(made, 0)
    return points.mask(made & ft, 1)
