"""Regularised adjusted plus-minus.

The idea is old and simple. Every stint is an observation of ten players and a
result. Regress the result on who was out there and the coefficients are each
player's effect, adjusted for everyone else on the floor.

Done as plain least squares this is adjusted plus-minus, and it does not work.
Teammates share almost all of their minutes, so the design matrix is close to
singular and the fit answers with enormous, unstable, obviously wrong numbers -
a backup centre with 300 possessions ends up at plus forty per hundred because
he happened to share them with a good starting five. The problem is not the idea.
It is that the data cannot separate players who are never apart, and least
squares does not know that it is guessing.

Ridge fixes it by refusing to make a large claim without a large amount of
evidence. The penalty pulls every coefficient toward zero, and pulls hardest on
exactly the players the data cannot pin down. The penalty strength is not a taste
parameter - it is chosen by cross-validation on held-out games, and
`run_02_rapm.py` shows the curve.

Two columns per player: one for what happens when he is on offence, one for when
he is on defence. Defensive coefficients are reported with the sign flipped so
that positive is good on both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


def parse_lineups(col: pd.Series) -> np.ndarray:
    """'(1, 2, 3, 4, 5)' -> an (n, 5) integer array, without literal_eval."""
    stripped = col.str.strip("()")
    return np.array([[int(v) for v in s.split(",")] for s in stripped], dtype=np.int64)


class Design:
    """Sparse design matrix plus the bookkeeping to read coefficients back."""

    def __init__(self, stints: pd.DataFrame):
        off = parse_lineups(stints["off_lineup"])
        dfn = parse_lineups(stints["def_lineup"])
        self.players = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
        self.index = {p: i for i, p in enumerate(self.players)}
        n_p = len(self.players)

        lookup = np.full(self.players.max() + 1, -1, dtype=np.int64)
        lookup[self.players] = np.arange(n_p)
        oi, di = lookup[off], lookup[dfn]

        n = len(stints)
        rows = np.repeat(np.arange(n), 10)
        cols = np.concatenate([oi, di + n_p], axis=1).ravel()
        data = np.ones(len(rows))

        home = sparse.csr_matrix(stints["off_home"].to_numpy(dtype=float).reshape(-1, 1))
        body = sparse.csr_matrix((data, (rows, cols)), shape=(n, 2 * n_p))
        self.X = sparse.hstack([body, home], format="csr")

        self.poss = stints["poss"].to_numpy(dtype=float)
        self.y = 100.0 * stints["pts"].to_numpy(dtype=float) / self.poss
        self.games = stints["game_id"].to_numpy()
        self.n_players = n_p

    def coefficients(self, beta: np.ndarray) -> pd.DataFrame:
        n_p = self.n_players
        return pd.DataFrame({
            "player_id": self.players,
            "o_rapm": beta[:n_p],
            "d_rapm": -beta[n_p:2 * n_p],   # sign flipped: positive is good defence
        }).assign(rapm=lambda d: d["o_rapm"] + d["d_rapm"])

    def possessions_by_player(self) -> pd.Series:
        n_p = self.n_players
        counts = self.X[:, :2 * n_p].T @ self.poss
        return pd.Series(counts[:n_p] + counts[n_p:], index=self.players, name="poss")


def fit(design: Design, alpha: float, rows: np.ndarray | None = None) -> np.ndarray:
    X = design.X if rows is None else design.X[rows]
    y = design.y if rows is None else design.y[rows]
    w = design.poss if rows is None else design.poss[rows]
    est = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", tol=1e-6, max_iter=3000)
    est.fit(X, y, sample_weight=w)
    return est.coef_, est.intercept_


def weighted_mse(design: Design, coef, intercept, rows: np.ndarray) -> float:
    pred = design.X[rows] @ coef + intercept
    err = design.y[rows] - pred
    w = design.poss[rows]
    return float((w * err ** 2).sum() / w.sum())


def cross_validate(design: Design, alphas, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Fold by game, so a stint is never scored by a model that saw its game."""
    rng = np.random.default_rng(seed)
    games = np.unique(design.games)
    fold_of_game = pd.Series(rng.integers(0, n_folds, len(games)), index=games)
    fold = fold_of_game.reindex(design.games).to_numpy()

    rows = []
    for alpha in alphas:
        errs = []
        for k in range(n_folds):
            tr = np.flatnonzero(fold != k)
            te = np.flatnonzero(fold == k)
            coef, b0 = fit(design, alpha, tr)
            errs.append(weighted_mse(design, coef, b0, te))
        rows.append({"alpha": alpha, "cv_mse": float(np.mean(errs)),
                     "cv_sd": float(np.std(errs))})
    return pd.DataFrame(rows)
