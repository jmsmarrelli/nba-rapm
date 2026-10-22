# Regularized adjusted scoring: a public-data methods study

This project rebuilds lineups from public play-by-play and fits regularized offensive and defensive player effects. It is a methods demonstration using **marked offensive sequences**, an approximation to possessions. Its ratings should not be presented as certified NBA RAPM or a definitive player ranking.

## Reconstruction and scoring checks

| Season | Games in source | Games kept | Retained share | Mean minute error (minutes) | Recorded points | Special FT points excluded | Modeled points |
|---|---|---|---|---|---|---|---|
| 2013–14 | 1,225 | 1,178 | 0.962 | 0.461 | 237,470 | 1,388 | 236,082 |
| 2014–15 | 1,226 | 1,157 | 0.944 | 0.459 | 231,040 | 1,139 | 229,901 |
| 2015–16 | 1,227 | 559 | 0.456 | 0.458 | 115,635 | 638 | 114,997 |
| 2016–17 | 1,230 | 1,176 | 0.956 | 0.452 | 247,958 | 1,323 | 246,635 |
| 2017–18 | 1,223 | 1,026 | 0.839 | 0.297 | 217,356 | 1,012 | 216,344 |
| 2018–19 | 1,229 | 981 | 0.798 | 0.308 | 217,827 | 1,106 | 216,721 |
| 2019–20 | 1,055 | 871 | 0.826 | 0.300 | 194,330 | 1,026 | 193,304 |
| 2020–21 | 1,080 | 1,065 | 0.986 | 0.260 | 238,589 | 1,154 | 237,435 |
| 2021–22 | 1,230 | 1,213 | 0.986 | 0.259 | 268,230 | 1,541 | 266,689 |
| 2022–23 | 1,175 | 1,149 | 0.978 | 0.263 | 263,125 | 1,433 | 261,692 |

Source: [reconstruction_audit.csv](output/reconstruction_audit.csv).

The source feed assigns zero `score_value` to many made free throws in older seasons. The corrected pipeline assigns one point to a marked made free throw and requires team totals to match the box score. The parser resets sequence IDs at period boundaries. Technical and other identified retained-ball free throws do not switch possession markers; their points are excluded from the modeled ordinary-scoring outcome and counted separately. Kept games reconcile recorded points to box scores and modeled points to the retained scoring events. Exhibition games are excluded.

Lineups must meet the published average and worst-player minute-error thresholds. Agreement in total minutes does not prove the lineup at every event is correct. Ambiguous jump balls and incomplete logs can still merge sequences. Credit goes to the lineup at the start of the marked sequence, which is a modeling convention when substitutions occur during a trip.

## A separate held-out test

| Model | MSE | R² vs league mean | MSE improvement vs team baseline | Test games | Training-selected penalty |
|---|---|---|---|---|---|
| League average | 3941.017 | 0.0000 | -7.087 | 685 | 4000.000 |
| Team offence and defence | 3933.931 | 0.0018 | 0.000 | 685 | 4000.000 |
| Adjusted plus-minus (no penalty) | 3948.006 | -0.0018 | -14.075 | 685 | 4000.000 |
| RAPM | 3915.011 | 0.0066 | 18.920 | 685 | 4000.000 |

Source: [out_of_sample.csv](output/out_of_sample.csv).

The outer test games are chosen first. Ridge tuning uses only the remaining training games; the split and training cross-validation scores are saved. A random held-out game from the same period measures a different task from forecasting a future season.

![Held-out evaluation](figures/05_out_of_sample.png)

MSE is measured in squared points per 100 marked sequences. These units use the sequence approximation described above.

## Prior-season carry-over baseline

| Model | Correlation | MAE | RMSE | Bias | Team-seasons |
|---|---|---|---|---|---|
| RAPM, 3-season window | 0.557 | 4.799 | 5.892 | 3.916 | 210 |
| RAPM, one season only | 0.577 | 3.405 | 4.401 | 1.669 | 210 |
| Box-score estimate | 0.475 | 4.547 | 5.767 | 3.873 | 210 |
| Team's own net rating last year | 0.575 | 3.444 | 4.480 | 0.000 | 210 |

Source: [forecast_correlations.csv](output/forecast_correlations.csv).

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
