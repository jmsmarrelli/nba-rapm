"""Stage 2 - fit RAPM, and show what the penalty is actually doing.

    python src/run_02_rapm.py

Fits the headline three-season window, picks the ridge penalty by cross-
validation on held-out games, and writes the ratings. Along the way it fits the
same model with the penalty turned almost all the way off, which is adjusted
plus-minus, so that the difference is visible rather than asserted.

Writes output/rapm_<window>.csv, output/ridge_cv.csv, and figures 01-03.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIG, HEADLINE_END, MIN_POSS_PLAYER, OUT, PROC, SEED, WINDOW  # noqa: E402
import rapm as R  # noqa: E402
import style as S  # noqa: E402

ALPHAS = np.array([1, 10, 30, 100, 300, 1000, 2000, 4000, 8000, 16000, 32000, 64000])
NEAR_ZERO = 1e-6   # ridge with the penalty off, i.e. plain adjusted plus-minus


def load_window(end: int, window: int) -> pd.DataFrame:
    frames = []
    for s in range(end - window + 1, end + 1):
        path = PROC / f"stints_{s}.csv.gz"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise SystemExit("No stint files. Run src/run_01_stints.py first.")
    return pd.concat(frames, ignore_index=True)


def names_for(player_ids, seasons) -> pd.Series:
    from config import RAW
    frames = []
    for s in seasons:
        p = RAW / f"player_box_{s}.csv.gz"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False, compression="gzip",
                                      usecols=["athlete_id", "athlete_display_name",
                                               "team_abbreviation", "game_date"]))
    box = pd.concat(frames, ignore_index=True).sort_values("game_date")
    last = box.groupby("athlete_id").last()
    return last.reindex(player_ids)


def fig_cv(cv: pd.DataFrame, best: float) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(cv["alpha"], cv["cv_mse"], color=S.SERIES[0], zorder=3)
    ax.scatter(cv["alpha"], cv["cv_mse"], s=22, color=S.SERIES[0],
               edgecolors=S.SURFACE, lw=1.5, zorder=4)
    ax.axvline(best, color=S.SERIES[1], lw=1.4, ls=(0, (3, 3)), zorder=2)
    ax.annotate(f"chosen penalty: {best:,.0f}", (best, cv["cv_mse"].max()),
                xytext=(8, -4), textcoords="offset points", fontsize=9,
                color=S.SERIES[1], fontweight="semibold")
    ax.set_xscale("log")
    S.strip(ax)
    ax.set_xlabel("Ridge penalty")
    ax.set_ylabel("Held-out weighted mean squared error")
    ax.set_title("The penalty is chosen, not assumed", loc="left")
    S.caption(fig, "Five-fold cross-validation with folds split by game, weighted by "
                   "possessions. The left end of this curve is adjusted plus-minus with no "
                   "regularisation at all; it is the worst model on the chart.")
    fig.tight_layout()
    fig.savefig(FIG / "01_ridge_cv.png")
    plt.close(fig)


def fig_shrinkage(apm: pd.DataFrame, rapm: pd.DataFrame, poss: pd.Series) -> None:
    m = apm.merge(rapm, on="player_id", suffixes=("_apm", "_rapm"))
    m["poss"] = m["player_id"].map(poss)
    m = m[m["poss"] >= 100]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    bins = np.linspace(-40, 40, 61)
    ax.hist(m["rapm_apm"].clip(-40, 40), bins=bins, color=S.NEUTRAL, zorder=2,
            label="No penalty (adjusted plus-minus)")
    ax.hist(m["rapm_rapm"].clip(-40, 40), bins=bins, color=S.SERIES[0], zorder=3,
            label="Ridge penalty (RAPM)")
    ax.legend(loc="upper left", fontsize=8.6)
    S.strip(ax)
    ax.set_xlabel("Rating, points per 100 marked sequences")
    ax.set_ylabel("Players")
    ax.set_title("What regularisation removes", loc="left")

    ax = axes[1]
    ax.scatter(m["poss"], m["rapm_apm"], s=8, c=S.NEUTRAL, lw=0, alpha=0.6, zorder=2)
    ax.scatter(m["poss"], m["rapm_rapm"], s=8, c=S.SERIES[0], lw=0, alpha=0.7, zorder=3)
    ax.axhline(0, color=S.INK_3, lw=0.9, zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(-45, 45)
    S.strip(ax)
    ax.set_xlabel("Marked sequences on the floor")
    ax.set_ylabel("Rating, points per 100 marked sequences")
    ax.set_title("Where it removes it", loc="left")
    ax.text(0.03, 0.05, "Grey: no penalty.   Blue: RAPM.", transform=ax.transAxes,
            fontsize=8.6, color=S.INK_2)

    S.caption(fig, f"Players with 100+ possessions in the window. Unpenalised ratings run "
                   f"from {m['rapm_apm'].min():.0f} to {m['rapm_apm'].max():+.0f} points per "
                   f"100 and are worst exactly where the sample is thinnest. The penalty "
                   f"leaves the well-observed players almost untouched.")
    fig.tight_layout()
    fig.savefig(FIG / "02_shrinkage.png")
    plt.close(fig)


def fig_leaderboard(tab: pd.DataFrame, window_label: str) -> None:
    d = tab.nlargest(20, "rapm").iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    # Paired bars rather than a stack: a stacked bar cannot show a negative half.
    h = 0.36
    ax.barh(y + h / 2 + 0.02, d["o_rapm"], height=h, color=S.SERIES[1], zorder=2)
    ax.barh(y - h / 2 - 0.02, d["d_rapm"], height=h, color=S.SERIES[0], zorder=2)
    right = max(d["o_rapm"].max(), d["d_rapm"].max()) * 1.06
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(right, i, f"{r['rapm']:+.2f}", va="center", ha="left",
                fontsize=8.8, color=S.INK, fontweight="semibold")
    ax.set_xlim(min(d["d_rapm"].min(), d["o_rapm"].min(), 0) * 1.15, right * 1.14)
    ax.axvline(0, color=S.INK_3, lw=0.9, zorder=3)
    ax.set_yticks(y, d["player"])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis="y", visible=False)
    S.strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Points per 100 marked sequences")
    ax.set_title(f"Sequence-proxy adjusted ratings, {window_label}", loc="left", pad=30)
    ax.text(0.0, 1.028, "Orange: offence.   Blue: defence.   Number: the two added together.",
            transform=ax.transAxes, fontsize=9, color=S.INK_2)
    S.caption(fig, f"Players with {MIN_POSS_PLAYER:,}+ possessions across the window. Ratings "
                   "are relative to a replacement-free league average, so they sum to roughly "
                   "zero across the league and are not a measure of value above a bench player.")
    fig.tight_layout()
    fig.savefig(FIG / "03_leaderboard.png")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    seasons = list(range(HEADLINE_END - WINDOW + 1, HEADLINE_END + 1))
    label = f"{seasons[0] - 1}-{str(seasons[0])[2:]} to {seasons[-1] - 1}-{str(seasons[-1])[2:]}"
    stints = load_window(HEADLINE_END, WINDOW)
    print(f"{len(stints):,} stint-sides, {stints['game_id'].nunique():,} games, "
          f"{stints['poss'].sum():,.0f} possessions  [{label}]")

    design = R.Design(stints)
    print(f"{design.n_players:,} players, design {design.X.shape[0]:,} x {design.X.shape[1]:,}")

    cv = R.cross_validate(design, ALPHAS, seed=SEED)
    cv.to_csv(OUT / "ridge_cv.csv", index=False)
    best = float(cv.loc[cv["cv_mse"].idxmin(), "alpha"])
    print("\n" + cv.round(4).to_string(index=False))
    print(f"\nchosen penalty {best:,.0f}")

    coef, b0 = R.fit(design, best)
    ratings = design.coefficients(coef)
    coef0, _ = R.fit(design, NEAR_ZERO)
    apm = design.coefficients(coef0)

    poss = design.possessions_by_player()
    ratings["poss"] = ratings["player_id"].map(poss)
    meta = names_for(ratings["player_id"], seasons)
    ratings["player"] = meta["athlete_display_name"].to_numpy()
    ratings["team"] = meta["team_abbreviation"].to_numpy()

    qualified = ratings[ratings["poss"] >= MIN_POSS_PLAYER].copy()
    qualified.round(4).to_csv(OUT / f"rapm_{seasons[0]}_{seasons[-1]}.csv", index=False)
    apm.round(4).to_csv(OUT / f"apm_{seasons[0]}_{seasons[-1]}.csv", index=False)

    fig_cv(cv, best)
    fig_shrinkage(apm, ratings[["player_id", "rapm"]], poss)
    fig_leaderboard(qualified, label)

    print(f"\nhome court, estimated inside the model: {coef[-1]:+.2f} points per 100 marked sequences")
    print(f"\n--- top 15 by RAPM, {label} ---")
    print(qualified.nlargest(15, "rapm")[
        ["player", "team", "poss", "o_rapm", "d_rapm", "rapm"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n--- top 8 defence ---")
    print(qualified.nlargest(8, "d_rapm")[
        ["player", "team", "poss", "o_rapm", "d_rapm"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    thin = apm.merge(poss.rename("poss"), left_on="player_id", right_index=True)
    thin = thin[thin["poss"].between(100, 400)]
    print(f"\nunregularised ratings for players with 100-400 possessions: "
          f"SD {thin['rapm'].std():.1f}, range {thin['rapm'].min():.0f} to "
          f"{thin['rapm'].max():+.0f} points per 100 marked sequences")
    reg = ratings.merge(poss.rename("p2"), left_on="player_id", right_index=True)
    reg = reg[reg["p2"].between(100, 400)]
    print(f"the same players after regularisation:                      "
          f"SD {reg['rapm'].std():.1f}, range {reg['rapm'].min():.1f} to "
          f"{reg['rapm'].max():+.1f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
