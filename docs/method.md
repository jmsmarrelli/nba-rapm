# Method notes

The reconstruction rules in full, the estimator's choices, and where I think this
is weakest.

## Rebuilding a lineup

For each game:

1. **Roster.** Both teams' available players come from the box score, not the
   log. This matters more than it sounds: a player who checks in, plays two
   minutes and never touches the ball appears in the log exactly twice, and both
   appearances are substitutions.

2. **Period-start five.** Within each period, for every player, record the index
   of his first substitution *in* and the index of his first appearance in any
   other event, including being substituted *out*. He started the period if the
   second exists and comes first, or if it exists and he never entered.

   Being substituted out counts as evidence of presence, which is what makes the
   rule work for a player who starts a period and does nothing before sitting
   down.

3. **Fallback.** If period one does not resolve to five a side, the box score's
   `starter` flag is used. Periods two onward have no fallback.

4. **Walk.** Apply substitutions in order. If a team's on-court set ever leaves
   five, the game is dropped.

5. **Check.** Accumulate floor time per player from the clock, compare against
   box score minutes, and drop the game if the mean absolute error across players
   exceeds 1.5 minutes or any single player is off by more than 6.

Steps 4 and 5 are the ones that matter for trust. The check is not a formality —
it is what catches a period whose starting five resolved to five *wrong* players,
which the walk itself has no way to notice.

### What the failures look like

| Reason | Meaning |
|---|---|
| `lineup` | a period would not resolve to five a side, or a substitution broke the five |
| `minutes_mean` | the whole game's floor time drifted from the box score |
| `minutes_max` | one player's did |
| `no_box` / `no_box_minutes` | the box score for that game is missing or empty |

The 2017-18 through 2019-20 seasons carry far more `lineup` failures than the
rest, which points at the source feed rather than at the rule: the same code run
on 2021-22 keeps 98.9% of games.

## Counting possessions

The usual estimator is FGA + 0.44×FTA − OREB + TOV. It is a good season-level
approximation and a poor stint-level one, because the 0.44 is a league-average
guess about which free throws end a trip and stints are three or four
possessions long.

Since the log is available, possessions are counted instead. Every event that
unambiguously identifies who has the ball is marked with that team — shots,
free throws, rebounds (offensive to the shooting team, defensive to the other),
turnovers — and the sequence of marks is compressed into runs. Each run is a
possession.

Events that do not identify possession (fouls, timeouts, jump balls,
substitutions) are skipped rather than guessed at. That is safe: dropping a
marker can only merge two possessions that were already adjacent and identical,
and adjacent identical marks collapse anyway.

The case this handles that the formula does not is the and-one. Made field goal
and free throw are both marked to the same team, they are adjacent, and they
collapse into the single possession they actually were.

The resulting count comes out at 96.3 possessions per team per game across the
sample. Published pace figures for these seasons sit a shade higher, around 98 to
100, and the gap is the possessions that end without a marker the log identifies
— a period expiring on a missed shot with no rebound recorded, most often. It is
a level shift of two or three per cent that applies to every stint equally, so it
scales every rating by the same small factor and does not move anyone's rank.

## The design matrix

One row per stint-side. Ten ones: five players in the offense block, five in the
defense block, plus a home indicator.

* Outcome: 100 × points ÷ possessions for that stint-side.
* Weight: possessions.
* Defensive coefficients are reported sign-flipped, so positive is good on both
  sides.

Ridge, `sparse_cg`, intercept unpenalized. 188,329 rows by 1,563 columns for the
three-season window, which solves in a couple of seconds.

### Why folds are split by game

Stints inside a game share a defense, a venue, a night's shooting, and mostly the
same twenty players. Randomly splitting rows lets the model score a stint using
another stint from the same game, which flatters every model in the comparison
and flatters the unregularised one most, because it is the one memorising.

### On the penalty

The cross-validation curve is shallow between roughly 1,000 and 8,000 — the
difference between the best and worst penalty in that range is under a tenth of a
percent of MSE. That is worth being honest about: the exact choice of 4,000 barely matters,
and the ranking of players is stable across the whole plateau. What matters
enormously is being somewhere on the plateau rather than at either end.

## Things I considered and did not do

**A box-score prior.** Every production metric starts players at a prior derived
from their counting stats, then lets the plus-minus data pull them away. It
improves the ratings. It also makes the comparison in `run_04_boxscore.py`
circular, because the thing being compared to the box score would already
contain it. Left out on purpose.

**Recency weighting.** A three-season window treats a possession from 2020 the
same as one from 2023. Real systems decay it. Adding decay would help the
forecast and cost the simplicity of "this window, these games, this fit".

**Filtering garbage time.** Every threshold is arbitrary, and possession
weighting already makes blowout minutes cheap relative to close ones. I would
revisit this if the ratings were going into a decision rather than a repository.

**Bootstrapped intervals.** Resampling games and refitting would give an honest
error bar per player. It is a few hours of compute and the right next step, and
its absence is the single biggest gap in what is here.

## Where I think this is weakest

**Collinearity is managed, not solved.** Ridge stops the model from making wild
claims about players it cannot separate. It does not let it separate them. Two
players who share 95% of their minutes get similar ratings because the data
genuinely cannot distinguish them, and the penalty makes that look like
agreement rather than ignorance. Nothing in the output flags which players are in
that position, and it should.

**The three thin seasons.** 2017-18 to 2019-20 keep 80-85% of games. If the games
that fail are not random — and there is no reason to assume they are — then those
seasons contribute a slightly biased sample, and the ten-season forecast exercise
runs across them.

**Ratings are league-relative.** They are not value over replacement and should
not be summed into a team total without care. The forecast in stage 4 does
possession-weight them into a team prediction, which is a legitimate use, but a
salary or trade valuation would need a replacement baseline this does not
provide.

**One number per player per window.** No role adjustment, no lineup interactions,
no split by opponent quality. A player whose value depends on fitting a specific
scheme gets one number that averages over the schemes he played in.
