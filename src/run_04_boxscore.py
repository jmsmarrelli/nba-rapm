"""Stage 4 - what the box score can see, and what it misses.

    python src/run_04_boxscore.py

Two questions a front office actually asks.

**How much of a player's plus-minus impact is visible in his counting stats?**
Regressing RAPM on per-100 box rates answers it directly, and the residuals name
the players the box score is worst at.

**Does any of this predict next season?** For each anchor season, ratings are
built from a three-season window and used to forecast the *following* season's
team net rating, against two honest competitors: the team's own net rating last
year, and the box-score estimate over the same window. Nothing in the forecast
uses target-season inputs: player usage and team assignments are carried over
from the prior season. Target-season data are used only for evaluation.

Writes output/box_model.csv, output/box_vs_rapm.csv, output/forecast.csv
and figures 06-07.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (FIG, HEADLINE_END, MIN_POSS_PLAYER, OUT, PROC, RAW,  # noqa: E402
                    SEASONS, SEED, WINDOW)
import labels as LB  # noqa: E402
import rapm as R  # noqa: E402
import style as S  # noqa: E402
from run_02_rapm import ALPHAS, load_window, names_for  # noqa: E402

LABEL_POSS = 6000   # roughly a starter's three seasons

BOX_STATS = ["points", "assists", "offensive_rebounds", "defensive_rebounds",
             "steals", "blocks", "turnovers", "fouls", "field_goals_attempted",
             "three_point_field_goals_made", "three_point_field_goals_attempted",
             "free_throws_attempted", "field_goals_made"]
PRETTY = {
    "points": "Points", "assists": "Assists", "offensive_rebounds": "Offensive rebounds",
    "defensive_rebounds": "Defensive rebounds", "steals": "Steals", "blocks": "Blocks",
    "turnovers": "Turnovers", "fouls": "Fouls", "field_goals_attempted": "Shot attempts",
    "three_point_field_goals_made": "Threes made",
    "three_point_field_goals_attempted": "Threes attempted",
    "free_throws_attempted": "Free throw attempts", "field_goals_made": "Field goals made",
}


def box_totals(seasons, eligible_games) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = RAW / f"player_box_{s}.csv.gz"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False, compression="gzip",
                                      usecols=["game_id", "athlete_id", "season", "season_type",
                                               "team_abbreviation", *BOX_STATS]))
    box = pd.concat(frames, ignore_index=True)
    box = box[(box["season_type"] == 2) & box["game_id"].isin(eligible_games)]
    return box.groupby("athlete_id")[BOX_STATS].sum()


def rates(totals: pd.DataFrame, poss: pd.Series) -> pd.DataFrame:
    p = poss.reindex(totals.index) / 2  # Counts include offensive and defensive sequences.
    return totals.div(p, axis=0).mul(100.0).dropna()


def fit_window(end: int, window: int, alpha: float):
    stints = load_window(end, window)
    design = R.Design(stints)
    coef, _ = R.fit(design, alpha)
    ratings = design.coefficients(coef)
    poss = design.possessions_by_player()
    ratings["poss"] = ratings["player_id"].map(poss)
    return ratings, poss, stints


def team_net(stints: pd.DataFrame) -> pd.DataFrame:
    off = stints.groupby(["season", "off_team"]).agg(
        pts=("pts", "sum"), poss=("poss", "sum")).reset_index()
    off = off.rename(columns={"off_team": "team"})
    dfn = stints.groupby(["season", "def_team"]).agg(
        pts_a=("pts", "sum"), poss_a=("poss", "sum")).reset_index()
    dfn = dfn.rename(columns={"def_team": "team"})
    t = off.merge(dfn, on=["season", "team"])
    t["net"] = 100 * (t["pts"] / t["poss"] - t["pts_a"] / t["poss_a"])
    return t[["season", "team", "net", "poss"]]


def player_team_possessions(stints: pd.DataFrame) -> pd.DataFrame:
    """Possessions each player spent with each team in each season."""
    off = R.parse_lineups(stints["off_lineup"])
    rows = np.repeat(np.arange(len(stints)), 5)
    long = pd.DataFrame({
        "season": stints["season"].to_numpy()[rows],
        "team": stints["off_team"].to_numpy()[rows],
        "player_id": off.ravel(),
        "poss": stints["poss"].to_numpy()[rows],
    })
    return long.groupby(["season", "team", "player_id"], as_index=False)["poss"].sum()


def forecast(window: int = WINDOW) -> pd.DataFrame:
    """Roll the window forward and predict the season after it, three ways."""
    rows = []
    anchors = [t for t in SEASONS if t - WINDOW + 1 >= SEASONS[0] and t + 1 <= HEADLINE_END]
    for t in anchors:
        history = load_window(t, window)
        cv = R.cross_validate(R.Design(history), ALPHAS, seed=SEED)
        cv.to_csv(OUT / f"forecast_training_cv_{t}_{window}.csv", index=False)
        alpha = float(cv.loc[cv["cv_mse"].idxmin(), "alpha"])
        ratings, poss, stints = fit_window(t, window, alpha)
        totals = box_totals(range(t - window + 1, t + 1), stints["game_id"].unique())
        rate = rates(totals, poss)
        keep = ratings.set_index("player_id").reindex(rate.index).dropna(subset=["rapm"])
        keep = keep[keep["poss"] >= 250]
        X, y, w = rate.loc[keep.index], keep["rapm"], keep["poss"]
        est = Ridge(alpha=25.0).fit(X, y, sample_weight=w)
        box_est = pd.Series(est.predict(rate), index=rate.index)

        nxt = pd.read_csv(PROC / f"stints_{t + 1}.csv.gz", low_memory=False)
        pt = player_team_possessions(stints[stints["season"] == t])
        actual = team_net(nxt).set_index("team")["net"]
        last = team_net(stints)
        last = last[last["season"] == t].set_index("team")["net"]

        pt["rapm"] = pt["player_id"].map(ratings.set_index("player_id")["rapm"])
        pt["box"] = pt["player_id"].map(box_est)
        for col in ("rapm", "box"):
            pt[col] = pt[col].fillna(-1.5)   # unseen player: a replacement-level prior
        agg = pt.groupby("team").apply(
            lambda g: pd.Series({
                "pred_rapm": 5 * np.average(g["rapm"], weights=g["poss"]),
                "pred_box": 5 * np.average(g["box"], weights=g["poss"]),
            }), include_groups=False)
        agg["pred_last"] = last.reindex(agg.index)
        agg["actual"] = actual.reindex(agg.index)
        agg["target_season"] = t + 1
        agg["usage_season"] = t
        agg["training_end"] = t
        agg["selected_alpha"] = alpha
        rows.append(agg.dropna().reset_index())
    return pd.concat(rows, ignore_index=True)


def fig_box_gap(tab: pd.DataFrame) -> None:
    d = tab.copy()
    d["gap"] = d["rapm"] - d["box_est"]
    # Label only well-observed players. At the 500-possession floor the extremes
    # are mostly players the model has barely seen, which is not the point here.
    heavy = d[d["poss"] >= LABEL_POSS]
    flagged = pd.concat([heavy.nlargest(7, "gap"), heavy.nsmallest(7, "gap")])

    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.scatter(d["box_est"], d["rapm"], s=13, c=S.NEUTRAL, lw=0, alpha=0.7, zorder=2)
    lim = [min(d["box_est"].min(), d["rapm"].min()) - 0.4,
           max(d["box_est"].max(), d["rapm"].max()) + 0.4]
    ax.plot(lim, lim, color=S.INK_3, lw=1.0, zorder=3)
    ax.scatter(flagged["box_est"], flagged["rapm"], s=34, c=S.SERIES[0],
               edgecolors=S.SURFACE, lw=1.4, zorder=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    S.strip(ax)
    ax.set_xlabel("What the box score predicts, points per 100 marked sequences")
    ax.set_ylabel("RAPM, points per 100 marked sequences")
    r2 = d["box_est"].corr(d["rapm"]) ** 2
    ax.set_title("How closely do box-score rates fit the adjusted ratings?",
                 loc="left")
    ax.text(0.03, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=10,
            color=S.INK, fontweight="semibold", va="top")
    fig.tight_layout()
    LB.place(ax, flagged["box_est"], flagged["rapm"], flagged["player"], colour=S.INK)
    S.caption(fig, f"Players with {MIN_POSS_PLAYER:,}+ possessions in the window. Above the "
                   "line, the box score undersells the player; below it, it oversells him. "
                   f"Named: the seven largest misses in each direction among players with "
                   f"{LABEL_POSS:,}+ possessions.")
    fig.savefig(FIG / "06_boxscore_gap.png")
    plt.close(fig)


def fig_forecast(fc: pd.DataFrame, corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    d = corr.iloc[::-1].reset_index(drop=True)
    colours = [S.SERIES[1] if "RAPM" in m else S.NEUTRAL for m in d["model"]]
    ax.barh(d.index, d["r"], color=colours, height=0.55, zorder=2)
    for i, row in d.iterrows():
        ax.text(row["r"] + 0.012, i, f"{row['r']:.2f}", va="center", fontsize=9,
                color=S.INK_2, fontweight="semibold")
    ax.set_yticks(d.index, d["model"])
    ax.set_xlim(0, max(d["r"]) * 1.18)
    ax.grid(axis="y", visible=False)
    S.strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Correlation with next season's team net rating")
    ax.set_title("Forecasting the season after", loc="left")

    ax = axes[1]
    ax.scatter(fc["pred_rapm"], fc["actual"], s=20, c=S.SERIES[0], lw=0, alpha=0.75, zorder=2)
    lo = min(fc["pred_rapm"].min(), fc["actual"].min()) - 1
    hi = max(fc["pred_rapm"].max(), fc["actual"].max()) + 1
    ax.plot([lo, hi], [lo, hi], color=S.INK_3, lw=1.0, zorder=3)
    b, a = np.polyfit(fc["pred_rapm"], fc["actual"], 1)
    xs = np.linspace(lo, hi, 40)
    ax.plot(xs, a + b * xs, color=S.INK, lw=1.5, zorder=4)
    S.strip(ax)
    ax.set_xlabel("Predicted from prior RAPM, points per 100 marked sequences")
    ax.set_ylabel("Actual net rating, points per 100 marked sequences")
    ax.set_title("Team by team", loc="left")

    S.caption(fig, f"{len(fc):,} team-seasons. Each forecast uses a three-season RAPM window "
                   "ending the year before, using that prior season’s team assignments and usage. "
                   "This carry-over baseline ignores future trades, injuries and rotation changes. "
                   "Grey is perfect calibration; black is a descriptive fitted line.")
    fig.tight_layout()
    fig.savefig(FIG / "07_forecast.png")
    plt.close(fig)


def main() -> None:
    cv = pd.read_csv(OUT / "ridge_cv.csv")
    alpha = float(cv.loc[cv["cv_mse"].idxmin(), "alpha"])
    seasons = list(range(HEADLINE_END - WINDOW + 1, HEADLINE_END + 1))

    ratings, poss, stints = fit_window(HEADLINE_END, WINDOW, alpha)
    totals = box_totals(seasons, stints["game_id"].unique())
    rate = rates(totals, poss)
    tab = ratings.set_index("player_id").reindex(rate.index).dropna(subset=["rapm"])
    tab = tab[tab["poss"] >= MIN_POSS_PLAYER]
    X, y, w = rate.loc[tab.index], tab["rapm"], tab["poss"]

    est = Ridge(alpha=25.0).fit(X, y, sample_weight=w)
    tab["box_est"] = est.predict(rate.loc[tab.index])
    meta = names_for(tab.index, seasons)
    tab["player"] = meta["athlete_display_name"].to_numpy()
    tab["team"] = meta["team_abbreviation"].to_numpy()
    tab.reset_index().round(4).to_csv(OUT / "box_vs_rapm.csv", index=False)

    sd = X.std()
    model = pd.DataFrame({
        "stat": [PRETTY[c] for c in X.columns],
        "per_100_coef": est.coef_,
        "per_sd": est.coef_ * sd.to_numpy(),
    }).sort_values("per_sd", key=np.abs, ascending=False)
    model.round(4).to_csv(OUT / "box_model.csv", index=False)

    r2 = tab["box_est"].corr(tab["rapm"]) ** 2
    print(f"box score explains R² = {r2:.3f} of RAPM across {len(tab):,} players")
    print("\nbox model, ranked by effect of a one standard deviation change")
    print(model.round(3).to_string(index=False))

    tab["gap"] = tab["rapm"] - tab["box_est"]
    heavy = tab[tab["poss"] >= LABEL_POSS]
    print(f"\nmost undersold by the box score ({LABEL_POSS:,}+ possessions)")
    print(heavy.nlargest(8, "gap")[["player", "team", "poss", "rapm", "box_est", "gap"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nmost oversold")
    print(heavy.nsmallest(8, "gap")[["player", "team", "poss", "rapm", "box_est", "gap"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig_box_gap(tab)

    fc = forecast()
    fc.round(4).to_csv(OUT / "forecast.csv", index=False)
    fc1 = forecast(window=1)
    fc1.round(4).to_csv(OUT / "forecast_1season.csv", index=False)
    corr = pd.DataFrame([
        {"model": f"RAPM, {WINDOW}-season window", "r": fc["pred_rapm"].corr(fc["actual"])},
        {"model": "RAPM, one season only", "r": fc1["pred_rapm"].corr(fc1["actual"])},
        {"model": "Box-score estimate", "r": fc["pred_box"].corr(fc["actual"])},
        {"model": "Team's own net rating last year", "r": fc["pred_last"].corr(fc["actual"])},
    ])
    # Correlation alone misses calibration and the size of forecast errors.
    for i, (data, col) in enumerate([(fc,"pred_rapm"),(fc1,"pred_rapm"),
                                     (fc,"pred_box"),(fc,"pred_last")]):
        error = data[col] - data["actual"]
        corr.loc[i,"mae"] = error.abs().mean()
        corr.loc[i,"rmse"] = np.sqrt((error ** 2).mean())
        corr.loc[i,"bias"] = error.mean()
        corr.loc[i,"team_seasons"] = len(data)
    corr.round(4).to_csv(OUT / "forecast_correlations.csv", index=False)
    fig_forecast(fc, corr)
    print(f"\nforecasting next season's team net rating, {len(fc):,} team-seasons")
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
