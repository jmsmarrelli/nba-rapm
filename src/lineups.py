"""Reconstructing who was actually on the floor.

The play-by-play files record substitutions but never state a lineup. To get one
you have to replay the game: start each period with the right five and apply
every substitution in order.

The hard part is the start of each period, because periods two, three and four
have no equivalent of a box score's starters column. The inference used here is
the standard one and it is exact whenever the log is complete:

    a player was on the floor at the start of a period if, during that period,
    he did something - took a shot, grabbed a rebound, committed a foul, went to
    the bench - before he ever entered the game.

Concretely, for each player in each period, compare the index of his first
substitution *in* against the index of his first appearance in any other event,
including being substituted *out*. If anything comes first, he started the
period. That returns exactly five players per team in the overwhelming majority
of periods; the rest are dropped and counted, because a lineup that is silently
wrong is far worse than one that is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Events that prove presence on the floor. Deliberately excludes the "enters the
# game" side of a substitution, which proves the opposite.
ATHLETE_COLS = ["athlete_id_1", "athlete_id_2", "athlete_id_3"]


def period_starters(period: pd.DataFrame, team_players: dict) -> dict | None:
    """Return {team_id: frozenset of five player ids} or None if it is ambiguous."""
    subs = period[period["type_text"] == "Substitution"]
    other = period[period["type_text"] != "Substitution"]

    first_in, first_evidence = {}, {}
    for i, row in zip(subs.index, subs.itertuples(index=False)):
        pin, pout = row.athlete_id_1, row.athlete_id_2
        if pd.notna(pin):
            first_in.setdefault(int(pin), i)
        if pd.notna(pout):
            first_evidence.setdefault(int(pout), i)
    for col in ATHLETE_COLS:
        vals = other[col]
        for i, p in zip(vals.index[vals.notna()], vals[vals.notna()]):
            first_evidence.setdefault(int(p), min(first_evidence.get(int(p), i), i))

    out = {}
    for team, roster in team_players.items():
        on = set()
        for p in roster:
            ev, sin = first_evidence.get(p), first_in.get(p)
            if ev is None:
                continue
            if sin is None or ev < sin:
                on.add(p)
        if len(on) != 5:
            return None
        out[team] = frozenset(on)
    return out


def walk_game(game: pd.DataFrame, team_players: dict, box_starters: dict | None = None):
    """Yield one row per event with the ten players on the floor attached.

    Returns (events, seconds_by_player, ok). `ok` is False when any period
    failed to resolve to five a side, in which case the game should be dropped.
    """
    game = game.sort_values("game_play_number")
    teams = sorted(team_players)
    on_court = {t: None for t in teams}
    seconds = {t: {} for t in teams}
    rows = []

    for period, block in game.groupby("period_number", sort=True):
        starters = period_starters(block, team_players)
        if starters is None and period == 1 and box_starters is not None:
            starters = box_starters
        if starters is None:
            return None, None, False
        on_court = {t: set(starters[t]) for t in teams}

        prev_clock = block["start_game_seconds_remaining"].iloc[0]
        for i, row in zip(block.index, block.itertuples(index=False)):
            clock = row.start_game_seconds_remaining
            elapsed = max(prev_clock - clock, 0)
            if elapsed:
                for t in teams:
                    for p in on_court[t]:
                        seconds[t][p] = seconds[t].get(p, 0) + elapsed
            prev_clock = clock

            if row.type_text == "Substitution":
                team = int(row.team_id) if pd.notna(row.team_id) else None
                pin = int(row.athlete_id_1) if pd.notna(row.athlete_id_1) else None
                pout = int(row.athlete_id_2) if pd.notna(row.athlete_id_2) else None
                if team in on_court and pin is not None and pout is not None:
                    on_court[team].discard(pout)
                    on_court[team].add(pin)
                    if len(on_court[team]) != 5:
                        return None, None, False
                continue

            rows.append((i, period,
                         tuple(sorted(on_court[teams[0]])),
                         tuple(sorted(on_court[teams[1]]))))

    events = pd.DataFrame(rows, columns=["idx", "period", "lineup_a", "lineup_b"])
    events = events.set_index("idx")
    return events, seconds, True


def minutes_check(seconds: dict, box: pd.DataFrame) -> pd.DataFrame:
    """Compare reconstructed floor time against the box score, player by player."""
    recon = {p: s / 60 for team in seconds.values() for p, s in team.items()}
    b = box[["athlete_id", "minutes"]].dropna()
    b = b[b["minutes"] > 0]
    b["recon"] = b["athlete_id"].map(recon).fillna(0.0)
    b["err"] = b["recon"] - b["minutes"]
    return b


def resolve_team_players(game: pd.DataFrame, box: pd.DataFrame) -> dict:
    """Roster on each side, taken from the box score rather than the log.

    Using the box score matters: a player who checks in, plays two minutes and
    never touches the ball appears in the log exactly twice, and both are
    substitutions.
    """
    b = box[(box["did_not_play"] != True) & box["athlete_id"].notna()]  # noqa: E712
    return {int(t): set(g["athlete_id"].astype(int))
            for t, g in b.groupby("team_id")}


def box_starters(box: pd.DataFrame) -> dict | None:
    b = box[box["starter"] == True]  # noqa: E712
    out = {int(t): frozenset(g["athlete_id"].astype(int)) for t, g in b.groupby("team_id")}
    if len(out) != 2 or any(len(v) != 5 for v in out.values()):
        return None
    return out
