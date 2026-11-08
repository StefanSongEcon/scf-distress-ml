# Household Delinquency Risk Score (v1.0)

A seven-question score that estimates how likely a household is to be behind on its debts, built from Federal
Reserve Survey of Consumer Finances data (2010–2025).

## What it answers

Given a few facts about a household, the score returns:

- the **probability** that a household with this profile was late or missed a loan payment in the past 12 months,
  and the probability that a payment was 60+ days late, at 2025 conditions;
- its **percentile** among U.S. households in 2025 and a **risk band** (Low < 50th, Moderate 50–80th,
  High 80–95th, Very high ≥ 95th percentile);
- the **main drivers** of its score relative to an average household.

## Inputs

| Input | Question | Unit |
|---|---|---|
| `age` | Age of the household head | years |
| `liquid_assets` | Checking, savings, money-market and prepaid balances | dollars |
| `total_debt` | All debt: mortgages, cards, student, auto and other loans | dollars |
| `home_value` | Value of the home you own (0 if renting) | dollars |
| `income_unusually_low` | Is this year's income unusually low compared with normal? | yes / no |
| `education_debt` | Student loan balance (part of total debt) | dollars |
| `education` | Highest education: no diploma / high school / some college / college degree | 1–4 or label |

Any input can be left out. A missing input is replaced by the 2025 survey-weighted sample mean of the model
features it feeds, so the household is treated as average on that dimension; the output lists which inputs were
filled in this way.

## How it was built

1. **Candidate inputs.** Seventeen values a household or adviser can supply, each mapped to its SCF variable.
2. **Selection.** Forward selection on the 2010–2019 validation years (train 2010–2013 → validate 2016; train
   2010–2016 → validate 2019), adding at each step the input that most improves validation log loss. The model
   keeps the smallest set within 0.005 validation AUC of the best: liquid assets and total debt alone take
   validation AUC from 0.63 (age only) to 0.79; five further inputs bring it to 0.81.
3. **Model.** L2-regularized logistic regression on smooth transforms of each input (dollars enter as
   asinh(dollars / 1,000)), plus indicators for having any debt and owning a home. Logistic regression matched
   gradient boosting throughout the underlying study, and each coefficient has the intuitive sign, so the score
   moves in one direction as any single input changes.
4. **Final fit.** Re-estimated on all five survey waves (2010–2025), with coefficients averaged across the SCF's
   five imputation replicates, and the intercept recalibrated so the average predicted rate equals the 2025 rate
   (19.6% any late payment; 8.2% 60+ days late).

## Performance

Trained on 2010–2019 and evaluated on the 2025 survey (which played no role in selection):

| Model | Inputs | AUC, any late | AUC, 60+ days | PR-AUC, any late | Top-10% capture |
|---|---|---|---|---|---|
| **Risk score (deployed)** | **7** | **0.851** | **0.855** | **0.55** | **31%** |
| All candidate inputs | 17 | 0.859 | 0.864 | 0.56 | 31% |
| Research model (paper) | 46 features | 0.858 | 0.862 | 0.56 | 32% |

Seven questions recover essentially all of the accuracy of the full research model.

**Effect of each input** (any late payment, log-odds): liquid assets −0.44 per unit of asinh($/1,000);
total debt +0.11, with +2.5 for having any debt; home value −0.06 and homeownership −0.13; income unusually low
+0.47; student loans +0.09; college degree −0.24 (relative to no diploma); risk peaks around age 45.

## Use

```python
from scf_distress.score import score, score_frame

score({"age": 29, "liquid_assets": 300, "total_debt": 38000, "education_debt": 30000,
       "income_unusually_low": "yes", "education": "some_college", "home_value": 0})
# any late payment: 74.1%, 100th percentile, Very high; drivers: liquid assets, total debt, low income

score_frame(households_df)   # batch scoring, one household per row
```

```bash
python -m scf_distress.score --age 52 --liquid-assets 45000 --total-debt 180000 --home-value 420000 --education college
python -m scf_distress.score --csv households.csv --out scores.csv
```

Example output:

```text
Any late or missed loan payment (past 12 months):
  probability 10.0%  (U.S. average 19.6%)
  percentile  41  ->  Low risk
Payment 60+ days late (past 12 months):
  probability 2.3%  (U.S. average 8.2%)
  percentile  38  ->  Low risk
Main drivers vs. an average household:
  total_debt             raises risk (+0.83 log-odds)
  liquid_assets          lowers risk (-0.73 log-odds)
  age                    raises risk (+0.26 log-odds)
Not provided (set to the 2025 average): income_unusually_low, education_debt
```

## Applications

- **Financial counseling and nonprofits:** prioritize outreach and emergency-savings support for households with
  thin buffers.
- **Policy analysis:** estimate the share of households under payment strain in surveys and administrative data
  that record finances but not delinquency.
- **Research:** a transparent, documented benchmark for household fragility.

The score describes risk for households with a given profile; it is not designed for individual credit
decisions, which are governed by fair-lending and credit-reporting rules.

## Notes

- Because risk is concentrated among a minority of households, a household that is average on every input scores
  below the national rate (10.6% versus 19.6%).
- The panel analysis in the paper shows that the same balance-sheet profile ranks households two years ahead
  (AUC 0.78 with past delinquency), but levels shift with economic conditions; the model should be recalibrated
  when new survey data are released.

## Files

| File | Content |
|---|---|
| `models/distress_score_v1.json` | coefficients, feature means, 2025 percentiles, training metadata |
| `src/scf_distress/deploy.py` | input definitions, feature builder, selection, fitting, export |
| `src/scf_distress/score.py` | scoring API and command line |
| `results/deploy/` | selection path and 2025 evaluation |
| `tests/test_deploy.py` | monotonicity, imputation, validation, train/score consistency |

Rebuild: `python -m scf_distress.pipeline --config config.yaml --deploy` (about 15 seconds).
