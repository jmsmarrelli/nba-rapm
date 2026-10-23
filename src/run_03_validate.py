"""Stage 3 - check the lineups, then check whether the ratings predict anything.

    python src/run_03_validate.py [season]

Two independent tests, because a plus-minus model can fail in two very different
places and the failures look nothing alike.

**Does the reconstruction match reality?** Every player's floor time, rebuilt
from substitutions, against the minutes in the box score. If those disagree, the
lineups are wrong and nothing downstream is worth reading.

**Do the ratings beat the obvious alternatives out of sample?** Held-out games,
scored four ways: a league average, team-level offence and defence effects, plain
adjusted plus-minus, and RAPM. Improvement over the team-level model is the only
one that means anything - it is the difference between knowing which team was on
the floor and knowing which five players were.

Writes output/minute_errors.csv, output/out_of_sample.csv and figures 04-05.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIG, HEADLINE_END, OUT, PROC, RAW, SEED, WINDOW  # noqa: E402
import lineups as L  # noqa: E402
import rapm as R  # noqa: E402
import style as S  # noqa: E402
from run_01_stints import BOX_COLS, PBP_COLS  # noqa: E402
from run_02_rapm import ALPHAS, NEAR_ZERO, load_window  # noqa: E402


def minute_errors(season: int) -> pd.DataFrame:
    pbp = pd.read_csv(RAW / f"play_by_play_{season}.csv.gz", low_memory=False,
                      usecols=PBP_COLS, compression="gzip")
    box = pd.read_csv(RAW / f"player_box_{season}.csv.gz", low_memory=False,
                      usecols=BOX_COLS, compression="gzip")
    pbp, box = pbp[pbp["season_type"] == 2], box[box["season_type"] == 2]
    box = box[box["athlete_id"].notna()]
    boxes = dict(list(box.groupby("game_id")))

    rows, failed = [], 0
    for gid, g in pbp.groupby("game_id", sort=True):
        bg = boxes.get(gid)
        if bg is None:
            continue
        roster = L.resolve_team_players(g, bg)
        if len(roster) != 2:
            failed += 1
            continue
        _, seconds, ok = L.walk_game(g, roster, L.box_starters(bg))
        if not ok:
            failed += 1
            continue
        chk = L.minutes_check(seconds, bg)
        chk["game_id"] = gid
        rows.append(chk)
    out = pd.concat(rows, ignore_index=True)
    out.attrs["failed"] = failed
    out.attrs["games"] = pbp["game_id"].nunique()
    return out


def team_baseline_design(stints: pd.DataFrame):
    """Offence-team and defence-team effects, plus home. The bar RAPM has to clear."""
    off = stints["off_team"].to_numpy()
    dfn = stints["def_team"].to_numpy()
    teams = np.unique(np.concatenate([off, dfn]))
    lookup = {t: i for i, t in enumerate(teams)}
    n, k = len(stints), len(teams)
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack([[lookup[t] for t in off],
                            [lookup[t] + k for t in dfn]]).ravel()
    body = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, 2 * k))
    home = sparse.csr_matrix(stints["off_home"].to_numpy(dtype=float).reshape(-1, 1))
    return sparse.hstack([body, home], format="csr")


def out_of_sample(stints: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    games = np.unique(stints["game_id"].to_numpy())
    test_games = set(rng.choice(games, size=len(games) // 5, replace=False).tolist())
    is_test = stints["game_id"].isin(test_games).to_numpy()
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)

    # Tune only on training games; outer test outcomes never choose alpha.
    cv = R.cross_validate(R.Design(stints.iloc[tr]), ALPHAS, seed=seed)
    alpha = float(cv.loc[cv["cv_mse"].idxmin(), "alpha"])
    cv.to_csv(OUT / "holdout_training_cv.csv", index=False)
    split = stints[["game_id"]].assign(is_test=is_test).drop_duplicates()
    split.to_csv(OUT / "holdout_split.csv", index=False)
    design = R.Design(stints)
    y, w = design.y, design.poss
    Xt = team_baseline_design(stints)

    def score(pred):
        err = y[te] - pred
        return float((w[te] * err ** 2).sum() / w[te].sum())

    base = np.average(y[tr], weights=w[tr])
    results = [{"model": "League average", "mse": score(np.full(len(te), base))}]

    for name, X, a in [("Team offence and defence", Xt, 1.0),
                       ("Adjusted plus-minus (no penalty)", design.X, NEAR_ZERO),
                       ("RAPM", design.X, alpha)]:
        est = Ridge(alpha=a, fit_intercept=True, solver="sparse_cg", tol=1e-6, max_iter=3000)
        est.fit(X[tr], y[tr], sample_weight=w[tr])
        results.append({"model": name, "mse": score(X[te] @ est.coef_ + est.intercept_)})

    out = pd.DataFrame(results)
    out["r2_vs_average"] = 1 - out["mse"] / out.loc[0, "mse"]
    out["gain_over_team"] = out.loc[1, "mse"] - out["mse"]
    out["training_selected_alpha"] = alpha
    out["test_games"] = len(test_games)
    out["test_possessions"] = int(w[te].sum())
    return out


def fig_reconstruction(err: pd.DataFrame, season: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.scatter(err["minutes"], err["recon"], s=4, c=S.SERIES[0], alpha=0.10, lw=0, zorder=2)
    lim = [0, err["minutes"].max() * 1.02]
    ax.plot(lim, lim, color=S.INK_3, lw=1.0, zorder=3)
    S.strip(ax)
    ax.set_xlabel("Minutes in the box score")
    ax.set_ylabel("Minutes from the rebuilt lineups")
    ax.set_title("Rebuilt floor time against the box score", loc="left")

    ax = axes[1]
    e = err["err"].clip(-3, 3)
    ax.hist(e, bins=np.linspace(-3, 3, 61), color=S.SERIES[0], zorder=2)
    ax.axvline(0, color=S.INK_3, lw=0.9, zorder=3)
    S.strip(ax)
    ax.set_xlabel("Rebuilt minus box score, minutes")
    ax.set_ylabel("Player-games")
    ax.set_title("Where it disagrees", loc="left")
    within = (err["err"].abs() <= 1).mean()
    ax.text(0.03, 0.94, f"median error {err['err'].abs().median():.2f} min\n"
                        f"{100 * within:.1f}% within one minute",
            transform=ax.transAxes, va="top", fontsize=9, color=S.INK_2, linespacing=1.6)

    S.caption(fig, 'Reconstructed and box-score minutes for games with recoverable lineups. Differences may reflect timestamp, substitution, starter and feed errors. Agreement in minutes is necessary but does not prove every event lineup is correct.')
    fig.tight_layout()
    fig.savefig(FIG / "04_reconstruction.png")
    plt.close(fig)


def fig_out_of_sample(res: pd.DataFrame) -> None:
    d = res.iloc[::-1].reset_index(drop=True)
    colours = [S.SERIES[1] if m == "RAPM" else S.NEUTRAL for m in d["model"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.barh(d.index, d["mse"], color=colours, height=0.58, zorder=2)
    for i, row in d.iterrows():
        ax.text(row["mse"] + 8, i, f"{row['mse']:,.0f}", va="center",
                fontsize=9, color=S.INK_2, fontweight="semibold")
    ax.set_yticks(d.index, d["model"])
    ax.set_xlim(0, d["mse"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    S.strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", labelsize=9.2)
    ax.set_xlabel("Held-out weighted mean squared error (lower is better)")
    ax.set_title("Models evaluated on held-out games", loc="left")
    S.caption(fig, 'An outer fifth of games is held out before alpha is tuned on training games. Weighted mean squared errors use the marked-sequence scoring outcome. This is a same-period held-out test, not a future-season forecast.')
    fig.tight_layout()
    fig.savefig(FIG / "05_out_of_sample.png")
    plt.close(fig)


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else HEADLINE_END
    err = minute_errors(season)
    err.to_csv(OUT / "minute_errors.csv", index=False)
    fig_reconstruction(err, season)
    print(f"{season}: {len(err):,} player-games across "
          f"{err['game_id'].nunique():,} reconstructed games "
          f"({err.attrs['failed']} games could not be rebuilt)")
    print(f"  median absolute error {err['err'].abs().median():.3f} minutes")
    print(f"  mean absolute error   {err['err'].abs().mean():.3f} minutes")
    print(f"  within one minute     {100 * (err['err'].abs() <= 1).mean():.1f}%")
    print(f"  correlation           {err['recon'].corr(err['minutes']):.5f}")

    stints = load_window(HEADLINE_END, WINDOW)
    res = out_of_sample(stints)
    res.round(4).to_csv(OUT / "out_of_sample.csv", index=False)
    fig_out_of_sample(res)
    print("\nheld-out prediction of stint scoring")
    print(res[["model", "mse", "r2_vs_average"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
