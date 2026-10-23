"""Stage 1 - turn play-by-play into stints with ten named players on the floor.

    python src/run_01_stints.py [first_season] [last_season]

For every regular season game: rebuild the lineups, check the reconstruction
against the box score, split the game into stints (a maximal run with the same
ten players), and count possessions and points on each side of each stint.

Writes data/processed/stints_<season>.csv.gz and output/reconstruction_audit.csv.
The audit is the interesting output: it says how much of the schedule survived
and how close the surviving games came to the box score.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (MAX_MEAN_MINUTE_ERROR, MAX_PLAYER_MINUTE_ERROR, OUT, PROC,  # noqa: E402
                    RAW, SEASONS)
import lineups as L  # noqa: E402
import possessions as P  # noqa: E402

PBP_COLS = ["game_id", "game_play_number", "period_number", "type_text", "team_id",
            "athlete_id_1", "athlete_id_2", "athlete_id_3", "scoring_play",
            "score_value", "shooting_play", "start_game_seconds_remaining",
            "home_team_id", "away_team_id", "season", "season_type", "game_date", "text"]
BOX_COLS = ["game_id", "athlete_id", "team_id", "team_abbreviation", "starter", "minutes",
            "did_not_play", "season_type", "home_away", "team_score"]


FRANCHISES = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NJ", "NO",
    "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "SEA", "TOR",
    "UTAH", "WSH",
}


def game_stints(g: pd.DataFrame, box_g: pd.DataFrame) -> tuple:
    g = g.copy()
    g["score_value"] = P.scoring_points(g)
    roster = L.resolve_team_players(g, box_g)
    if len(roster) != 2 or any(len(v) < 5 for v in roster.values()):
        return None, "roster"

    events, seconds, ok = L.walk_game(g, roster, L.box_starters(box_g))
    if not ok:
        return None, "lineup"

    chk = L.minutes_check(seconds, box_g)
    if chk.empty:
        return None, "no_box_minutes"
    if chk["err"].abs().mean() > MAX_MEAN_MINUTE_ERROR:
        return None, "minutes_mean"
    if chk["err"].abs().max() > MAX_PLAYER_MINUTE_ERROR:
        return None, "minutes_max"

    e = g.loc[events.index].copy()
    e["lineup_a"] = events["lineup_a"].to_numpy()
    e["lineup_b"] = events["lineup_b"].to_numpy()

    off = P.offensive_team(e)
    e["poss_id"] = P.possession_runs(off, e["period_number"])
    e = e[e["poss_id"].notna()]
    if e.empty:
        return None, "no_possessions"
    e["off_team"] = off.reindex(e.index).groupby(e["period_number"]).ffill()

    change = (e["lineup_a"] != e["lineup_a"].shift()) | (e["lineup_b"] != e["lineup_b"].shift())
    change |= e["period_number"].ne(e["period_number"].shift())
    e["stint"] = change.cumsum()

    home = int(g["home_team_id"].iloc[0])
    teams = sorted(roster)
    a_is_home = teams[0] == home

    # A possession belongs to the stint it began in, and so do the points it
    # produced. Splitting those two apart - counting the possession in one stint
    # and its basket in the next because someone substituted mid-trip - would put
    # credit on the wrong five players.
    first = e.groupby("poss_id").first()
    poss = (first.groupby(["stint", "off_team"]).size()
            .rename("poss").reset_index())

    scored = e[e["scoring_play"].fillna(False).astype(bool) & ~P.special_free_throw(e)].copy()
    scored["stint"] = scored["poss_id"].map(first["stint"])
    pts = (scored.groupby(["stint", "team_id"])["score_value"].sum()
           .rename("pts").reset_index().rename(columns={"team_id": "off_team"}))

    lineup = e.groupby("stint").agg(
        lineup_a=("lineup_a", "first"), lineup_b=("lineup_b", "first"),
        secs=("start_game_seconds_remaining", lambda s: s.iloc[0] - s.iloc[-1]))

    tab = poss.merge(pts, on=["stint", "off_team"], how="outer").fillna({"poss": 0, "pts": 0})
    if ((tab["poss"] == 0) & (tab["pts"] != 0)).any():
        return None, "unassigned_scoring"
    tab = tab[tab["poss"] > 0]
    source_score = g[g["scoring_play"].fillna(False).astype(bool)]
    recorded = source_score.groupby("team_id")["score_value"].sum()
    official = box_g.groupby("team_id")["team_score"].first()
    if not np.allclose(recorded.reindex(official.index, fill_value=0), official):
        return None, "score_box_mismatch"
    ordinary = source_score[~P.special_free_throw(source_score)]
    expected = ordinary.groupby("team_id")["score_value"].sum()
    modeled = tab.groupby("off_team")["pts"].sum()
    if not np.allclose(modeled.reindex(official.index, fill_value=0),
                       expected.reindex(official.index, fill_value=0)):
        return None, "score_not_conserved"
    tab = tab.join(lineup, on="stint")
    tab["def_team"] = np.where(tab["off_team"] == home,
                               int(g["away_team_id"].iloc[0]), home)
    tab["off_lineup"] = np.where(
        tab["off_team"] == teams[0], tab["lineup_a"], tab["lineup_b"])
    tab["def_lineup"] = np.where(
        tab["off_team"] == teams[0], tab["lineup_b"], tab["lineup_a"])
    tab["off_home"] = (tab["off_team"] == home).astype(int)
    tab["game_id"] = int(g["game_id"].iloc[0])
    tab["season"] = int(g["season"].iloc[0])
    tab["a_is_home"] = int(a_is_home)

    keep = ["game_id", "season", "stint", "off_team", "def_team", "off_home",
            "off_lineup", "def_lineup", "poss", "pts", "secs"]
    return tab[keep], chk["err"].abs().mean()


def run_season(season: int) -> dict:
    pbp_path = RAW / f"play_by_play_{season}.csv.gz"
    box_path = RAW / f"player_box_{season}.csv.gz"
    if not (pbp_path.exists() and box_path.exists()):
        raise FileNotFoundError(f"Missing raw inputs for {season}")

    pbp = pd.read_csv(pbp_path, low_memory=False, usecols=PBP_COLS, compression="gzip")
    box = pd.read_csv(box_path, low_memory=False, usecols=BOX_COLS, compression="gzip")
    pbp = pbp[pbp["season_type"] == 2]
    box = box[box["season_type"] == 2]
    box = box[box["athlete_id"].notna()]

    real = set(box.loc[box["team_abbreviation"].isin(FRANCHISES), "team_id"]
               .dropna().astype("int64"))
    keep_games = (box.groupby("game_id")["team_id"]
                  .apply(lambda t: set(t.dropna().astype("int64")) <= real))
    exhibition = int((~keep_games).sum())
    box = box[box["game_id"].isin(keep_games[keep_games].index)]
    pbp = pbp[pbp["game_id"].isin(keep_games[keep_games].index)]

    repaired = P.scoring_points(pbp)
    corrected_free_throws = int(repaired.ne(pbp["score_value"]).sum())
    pbp["score_value"] = repaired
    boxes = dict(list(box.groupby("game_id")))
    out, reasons, errs = [], {}, []
    t0 = time.time()
    for gid, g in pbp.groupby("game_id", sort=True):
        bg = boxes.get(gid)
        if bg is None:
            reasons["no_box"] = reasons.get("no_box", 0) + 1
            continue
        res, info = game_stints(g, bg)
        if res is None:
            reasons[info] = reasons.get(info, 0) + 1
            continue
        out.append(res)
        errs.append(info)

    n_games = pbp["game_id"].nunique()
    if out:
        stints = pd.concat(out, ignore_index=True)
        stints.to_csv(PROC / f"stints_{season}.csv.gz", index=False, compression="gzip")
    else:
        raise RuntimeError(f"No games passed reconstruction and scoring checks for {season}")

    kept = pbp[pbp["game_id"].isin(stints["game_id"].unique())] if len(stints) else pbp.iloc[:0]
    scored_kept = kept[kept["scoring_play"].fillna(False).astype(bool)]
    excluded_points = scored_kept.loc[P.special_free_throw(scored_kept), "score_value"].sum()
    audit = {
        "corrected_scoring_rows": corrected_free_throws,
        "recorded_points_kept": float(scored_kept["score_value"].sum()),
        "excluded_special_ft_points": float(excluded_points),
        "modeled_points": float(stints["pts"].sum()) if len(stints) else 0,
        "season": season,
        "games_in_pbp": n_games,
        "dropped_exhibition": exhibition,
        "games_kept": len(out),
        "keep_rate": round(len(out) / n_games, 4) if n_games else 0.0,
        "mean_minute_error": round(float(np.mean(errs)), 3) if errs else np.nan,
        "stints": len(stints),
        "possessions": int(stints["poss"].sum()) if len(stints) else 0,
        "seconds": round(time.time() - t0, 1),
    }
    audit.update({f"drop_{k}": v for k, v in sorted(reasons.items())})
    print("  " + "  ".join(f"{k}={v}" for k, v in audit.items()))
    return audit


def main() -> None:
    seasons = SEASONS
    if len(sys.argv) == 3:
        seasons = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
    rows = [run_season(s) for s in seasons]
    rows = [r for r in rows if r]
    audit = pd.DataFrame(rows)
    path = OUT / "reconstruction_audit.csv"
    if path.exists():
        old = pd.read_csv(path)
        audit = (pd.concat([old[~old["season"].isin(audit["season"])], audit])
                 .sort_values("season"))
    audit.to_csv(path, index=False)
    print("\n" + audit.to_string(index=False))


if __name__ == "__main__":
    main()
