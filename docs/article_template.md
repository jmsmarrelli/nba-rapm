# Regularized adjusted scoring: a public-data methods study

This project rebuilds lineups from public play-by-play and fits regularized offensive and defensive player effects. It is a methods demonstration using **marked offensive sequences**, an approximation to possessions. Its ratings should not be presented as certified NBA RAPM or a definitive player ranking.

## Reconstruction and scoring checks

{{table:reconstruction_audit.csv|season,games_in_pbp,games_kept,keep_rate,mean_minute_error,recorded_points_kept,excluded_special_ft_points,modeled_points}}

The source feed assigns zero `score_value` to many made free throws in older seasons. The corrected pipeline assigns one point to a marked made free throw and requires team totals to match the box score. The parser resets sequence IDs at period boundaries. Technical and other identified retained-ball free throws do not switch possession markers; their points are excluded from the modeled ordinary-scoring outcome and counted separately. Kept games reconcile recorded points to box scores and modeled points to the retained scoring events. Exhibition games are excluded.

Lineups must meet the published average and worst-player minute-error thresholds. Agreement in total minutes does not prove the lineup at every event is correct. Ambiguous jump balls and incomplete logs can still merge sequences. Credit goes to the lineup at the start of the marked sequence, which is a modeling convention when substitutions occur during a trip.

## A separate held-out test

{{table:out_of_sample.csv|model,mse,r2_vs_average,gain_over_team,test_games,training_selected_alpha}}

The outer test games are chosen first. Ridge tuning uses only the remaining training games; the split and training cross-validation scores are saved. A random held-out game from the same period measures a different task from forecasting a future season.

![Held-out evaluation](figures/05_out_of_sample.png)

MSE is measured in squared points per 100 marked sequences. These units use the sequence approximation described above.

## Prior-season carry-over baseline

{{table:forecast_correlations.csv}}

MAE, RMSE and bias are points per 100 marked sequences. Correlation is unitless.

For each forecast year, player ratings, tuning, team assignments and usage weights come from earlier seasons. The target season is used only for evaluation outcomes. This carry-over baseline deliberately cannot anticipate trades, injuries or new rotations. It is not a roster-aware preseason projection.

Box-score numerators are restricted to the same reconstructed games as their denominators. The published rates account for exposure being counted on both offense and defense.

## What the model supports

The useful contribution is the reconstruction audit, regularization comparison and separation of training from evaluation. Player coefficients remain conditional associations affected by teammates, omitted context, missing games and the sequence approximation. Before using the rankings for basketball decisions, replace the proxy with a possession parser validated against independent event annotations and evaluate sensitivity to game exclusions.

This study ends in 2022–23. It contains no current player projection.


## Reproduce

The historical analysis used Python 3.12 with the pinned dependencies below. From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make fetch-data PY=python
make all PY=python
make report site PY=python
```

`make all` verifies source hashes before running the analyses in order. A missing or changed input stops the build. Data downloads need network access; article rendering uses the included result tables. See [the source manifest](data/source_manifest.json), [method notes](docs/method.md) and [current validation record](VALIDATION_CURRENT.md) for the distinction between a source rebuild and checks of delivered outputs.

Season keys in CSV files use the **ending year**: `2025` means 2024–25. Display tables use full season labels. The MIT license covers code; third-party data retain their providers' rights and terms.

[Portfolio](https://github.com/jmsmarrelli/basketball-analytics) · [AI assistance and provenance](PROVENANCE.md)
