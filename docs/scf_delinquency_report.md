# Thin Buffers: Classifying and Predicting Household Delinquency in the Survey of Consumer Finances

**Yingrui Song**
November 2026

---

## Abstract

Which households fall behind on their debts, and can it be seen coming? Using Federal Reserve Survey of
Consumer Finances (SCF) microdata from 2007 to 2025, I train survey-weighted machine-learning models to identify
households that missed loan payments. In cross-sections, a household's balance sheet raises out-of-sample
discrimination from an AUC of 0.69 with demographics alone to 0.86 on the newly released 2025 survey. Liquid assets
carry most of the signal: among borrowers, detailed measures of debt and debt service add little once liquidity and
wealth are known, and regularized logistic regression performs as well as gradient boosting. Models trained on
2010–2019 rank 2025 households accurately but miss the level of delinquency, which rose from 12 to 20 percent. In the
2007–2009 SCF panel, balance sheets observed in 2007 predict delinquency during the financial crisis two years later
(AUC 0.78 with past delinquency), including among households that were current in 2007. Roughly half of the
balance-sheet signal is forward-looking; the remainder reflects distress that is already under way when finances are
measured.

---

## 1. Introduction

Missed debt payments are often the first observable sign of household financial distress, and they precede the more
costly outcomes—default, foreclosure, bankruptcy—that motivate much of household-finance policy. Understanding
which households fall behind, and whether this can be anticipated from information available beforehand, matters
for consumer-credit supervision, for the design of emergency-savings and forbearance policy, and for measuring
financial fragility in surveys that do not ask about payment problems directly.

This paper studies delinquency in the Survey of Consumer Finances, the most comprehensive source of household
balance-sheet data in the United States. It asks two questions. The first is a classification question: given a
household's demographics, income, assets and debts, how well can one identify households that were late on a loan
payment in the past year, and which parts of the balance sheet carry that information? The second is a prediction
question: does a household's balance sheet anticipate delinquency that occurs two years later?

The two questions call for different data. Classification uses five SCF cross-sections (2010, 2013, 2016, 2019 and
2025; 28,882 families), trained on the first four and evaluated once on 2025. Prediction uses the 2007–2009 SCF panel,
which re-interviewed 3,857 families during the financial crisis, so that every predictor is measured before the
outcome period.

Four findings emerge.

1. **Balance sheets identify delinquent households well, and liquidity is the core signal.** On the 2025 test, AUC
   rises from 0.69 with demographics to 0.86 with the full balance sheet. Liquid assets are the single most
   informative variable. Among borrowers, adding detailed debt amounts, composition and payment-to-income ratios to
   wealth and liquidity raises AUC by only about 0.02.
2. **Flexible models add nothing over a well-specified linear model.** Gradient boosting and L2-regularized
   logistic regression are statistically indistinguishable in every specification with the full feature set.
3. **Rankings transport across time; levels do not.** Models trained on 2010–2019 rank 2025 households as
   accurately as they ranked households in earlier years, yet delinquency rose sharply after a decade of decline and
   the models under-predict its 2025 level.
4. **About half of the balance-sheet signal is forward-looking.** In the panel, 2007 balance sheets predict 2009
   delinquency with AUC 0.76, or 0.78 adding 2007 delinquency, compared with 0.84 for the same families' 2009 balance
   sheets. They also predict the onset of delinquency among households that were current in 2007.

Methodologically, the paper builds a survey-correct machine-learning workflow for SCF microdata. The SCF's standard
variables are produced by Federal Reserve SAS programs; I port these programs to Python and reproduce the Fed's
published summary data exactly. Every model is estimated separately on each of the SCF's five multiple-imputation
implicates, survey weights enter both estimation and evaluation, and confidence intervals use the SCF's 999
replicate weights. The 2025 evaluation was specified and committed before the test data were scored.

The rest of the paper is organized as follows. Section 2 describes the data and measurement. Section 3 lays out the
empirical design. Sections 4 and 5 present the classification and prediction results. Section 6 discusses what the
two exercises imply together, and Section 7 concludes.

---

## 2. Data and Measurement

### 2.1 The Survey of Consumer Finances

The SCF is a triennial survey of U.S. families sponsored by the Federal Reserve Board. It combines an area-probability
sample with a list sample that over-samples wealthy households, so survey weights are essential for population
estimates. Missing responses are multiply imputed, and each family appears as five implicates. I use the full public
datasets for the 2010, 2013, 2016, 2019 and 2025 cross-sections and the public 2007–2009 panel. Table 1 summarizes the
samples.

**Table 1. Samples**

| | Cross-sections | Panel |
|---|---|---|
| Waves | 2010, 2013, 2016, 2019, 2025 | 2007 interview, 2009 re-interview |
| Families | 28,882 (4,360–6,482 per wave) | 3,857 |
| Records (5 implicates each) | 144,410 | 19,285 |
| Weights | wave survey weights | 2009 panel weights |
| Dollars | 2025 | 2009 |

### 2.2 Constructed variables

Income, assets, debts and payment burdens in the SCF are standardized by the Federal Reserve's summary-variable
programs, written in SAS. Rather than approximating them, I wrote an interpreter that executes the Fed's programs in
Python, resolving each year's branches of the code and reproducing SAS's treatment of missing values. Applied to the
raw survey responses, the port reproduces all 350 summary variables in the Fed's published extracts exactly, for every
family and implicate in all five cross-sections. The panel program is reproduced in the same way (Appendix A).

### 2.3 Outcomes

The SCF asks families with loan payments whether, in the past twelve months, all payments were made as scheduled and,
if not, whether any payment was 60 or more days late. I study two outcomes:

- `LATE`: any loan payment late or missed in the past twelve months;
- `LATE60`: any payment 60 or more days late.

Delinquency is common and has moved substantially over time (Figure 1). Among all families, the share with a late
payment fell from 17.3 percent in 2010 to 12.3 percent in 2019 and then rose to 19.6 percent in 2025; serious
delinquency followed the same path (8.1, 4.6 and 8.2 percent). Among families with debt, a quarter (25.3 percent)
reported a late payment in 2025.

![Figure 1](../results/figures/fig1_prevalence_by_year.png)

*Figure 1. Share of families with a late payment (blue) and a payment 60+ days late (orange) in the previous twelve
months, all families and families with debt. Survey-weighted; vertical ticks are 95% confidence intervals.*

### 2.4 Predictors

I organize 46 predictors into four nested blocks (Table 2). Dollar amounts enter through the inverse hyperbolic sine,
which preserves the sign of net worth and home equity; ratios use the Fed's definitions or floor denominators at
$1,000. Because families without loan payments cannot be delinquent, I also analyze the subsample of families with
debt, where debt variables cannot separate the outcome mechanically. The models exclude survey year, the outcome
questions, and realized hardship events such as bankruptcy, foreclosure or credit denial.

**Table 2. Feature blocks**

| Block | Variables added |
|---|---|
| M1 Demographics | age, age², marital status, children, education, family structure |
| M2 Income | income, wage income, normal income, zero/negative income, income unusually high or low |
| M3 Wealth and liquidity | total, financial, liquid and retirement assets; no liquid assets; liquid assets relative to income |
| M4 Debt and housing | net worth, debt by type, monthly payments, payment-to-income ratios, debt-to-income, debt-to-assets, liquid assets relative to debt, debt composition, homeownership, home value and equity |

---

## 3. Empirical Design

### 3.1 Models and estimation

I compare a constant-probability benchmark, L2-regularized logistic regression and gradient-boosted trees (XGBoost).
Each model is estimated separately on each of the five implicates, with survey weights in the loss function, and
results are averaged across implicates. Performance is measured by survey-weighted ROC AUC, precision-recall AUC
(PR-AUC), Brier score and calibration. Standard errors follow the Federal Reserve's method, combining variation across
999 replicate weights with variation across implicates; differences between models are tested on the same households
and replicate weights.

### 3.2 Classification: out-of-time evaluation

The cross-sectional models are tuned on the historical waves with forward-chaining validation—training on 2010 and
2013 to predict 2016, and on 2010–2016 to predict 2019—choosing hyperparameters by validation log loss. The final
models are re-estimated on 2010–2019 and evaluated once on 2025. The specification was fixed and committed to version
control before the 2025 data were scored, so no feature, transformation or hyperparameter was influenced by the test
wave.

### 3.3 Prediction: the 2007–2009 panel

In the panel, predictors come from the 2007 interview (May 2007–March 2008) and outcomes from the 2009 re-interview
(July 2009–January 2010), whose reference period covers the height of the financial crisis. The two reference periods
do not overlap. I compare four information sets: 2007 demographics, the full 2007 balance sheet, 2007 delinquency
alone, and the 2007 balance sheet together with 2007 delinquency. As a benchmark, the same families' 2009 balance
sheets are used to classify 2009 delinquency. With a single panel, models are evaluated by family-grouped five-fold
cross-validation, with hyperparameters tuned in an inner loop and the same folds used for every model so that all
comparisons are paired.

---

## 4. Classification Results

### 4.1 Out-of-sample performance in 2025

Table 3 reports performance on the 2025 test wave. Demographics alone achieve an AUC of 0.69. Each additional block
improves discrimination, and the full model reaches 0.86 for both outcomes. Precision-recall performance improves
proportionally more: PR-AUC rises from 0.33 to 0.57 for any late payment, against a base rate of 0.20, and from 0.17 to
0.36 for serious delinquency, against 0.08. The households in the top decile of predicted risk account for 32 percent
of late payers and 42 percent of serious delinquencies.

**Table 3. Classification performance on the 2025 test wave (all families)**

| Information | Model | `LATE` AUC | `LATE` PR-AUC | `LATE60` AUC | `LATE60` PR-AUC |
|---|---|---|---|---|---|
| Base rate | Constant | 0.500 | 0.196 | 0.500 | 0.082 |
| M1 Demographics | Logit | 0.690 [0.670, 0.709] | 0.33 | 0.686 [0.657, 0.715] | 0.17 |
| M2 + Income | XGBoost | 0.735 | 0.37 | 0.746 | 0.20 |
| M3 + Wealth and liquidity | XGBoost | 0.791 | 0.45 | 0.800 | 0.25 |
| M4 + Debt and housing | Logit | 0.858 [0.846, 0.869] | 0.56 | 0.862 [0.843, 0.882] | 0.35 |
| M4 + Debt and housing | XGBoost | 0.863 [0.851, 0.874] | 0.57 | 0.867 [0.848, 0.885] | 0.36 |

*Notes: Models trained on 2010–2019. Brackets are 95% confidence intervals from SCF replicate weights and implicates.*

### 4.2 Where the information is

Figure 2 traces performance as blocks are added, separately for all families and for families with debt. For all
families, the debt and housing block brings the largest gain (0.07–0.08 AUC), but part of that gain is mechanical:
debt variables reveal which families have payments that can be missed. Among families with debt, the picture changes.
Wealth and liquidity account for most of the improvement over demographics, and the full debt block adds only about
0.02 AUC (0.016 for any late payment, 95% CI 0.008–0.025).

![Figure 2](../results/figures/fig4_incremental_blocks_2025.png)

*Figure 2. Performance on the 2025 test as feature blocks are added. Solid lines: XGBoost; dashed: logistic
regression. Blue: all families; orange: families with debt.*

Feature-level importance confirms this pattern (Figure 3). Liquid assets are the most important single variable in
nearly every model; low liquidity is associated with sharply higher predicted delinquency. Debt-to-assets and the
unsecured share of debt follow, along with income that was unusually low relative to normal. Debt levels,
payment-to-income ratios and credit-card balances matter little once liquidity and debt structure are accounted for.

![Figure 3](../results/figures/fig5b_feature_importance_2025.png)

*Figure 3. Permutation importance in the 2025 test wave (XGBoost, full feature set): the fall in AUC when a variable
is randomly shuffled.*

### 4.3 Model complexity

Gradient boosting offers no meaningful advantage. With the full feature set, the difference between XGBoost and
logistic regression never exceeds 0.01 AUC and is not statistically significant in any sample or outcome. Once
heavy-tailed balance-sheet variables are transformed appropriately, the relationship between finances and delinquency
is captured by a linear index.

### 4.4 Rankings and levels over time

Models trained on 2010–2019 rank 2025 households at least as well as they ranked households in the validation years
(AUC 0.86 in 2025 versus 0.81–0.82 when predicting 2016 and 2019). They do not, however, capture the 2025 level of
delinquency (Figure 4). Every model under-predicts: the calibration intercept is between 0.36 and 0.71 on the log-odds
scale, and in the riskiest decile 63 percent of families were delinquent against 44 percent predicted. The shortfall is
largest for households under 55 and for the bottom three income quintiles. Because the models deliberately exclude
calendar time, they carry the historical relationship between balance sheets and delinquency into a period in which
the same balance sheet was associated with more distress.

![Figure 4](../results/figures/fig3_calibration_2025.png)

*Figure 4. Calibration in 2025. Left: observed versus predicted delinquency by decile of predicted risk. Right:
distribution of predicted risk for delinquent and non-delinquent families.*

---

## 5. Prediction Results

### 5.1 Two years ahead

Table 4 and Figure 5 report the panel results. Balance sheets observed in 2007 predict 2009 delinquency with an AUC of
0.76 for any late payment and 0.78 for serious delinquency, compared with 0.69–0.72 for 2007 demographics. Adding 2007
delinquency raises the AUC to 0.78 and 0.80. The same families' 2009 balance sheets classify 2009 delinquency with AUC
0.84 and 0.88.

**Table 4. Predicting 2009 delinquency (logistic regression, cross-validated AUC)**

| Information | All families: `LATE` | All families: `LATE60` | Families with debt in 2007: `LATE` | Not late in 2007: `LATE` |
|---|---|---|---|---|
| 2007 demographics | 0.691 | 0.716 | 0.664 | 0.698 |
| 2007 balance sheet | 0.756 | 0.780 | 0.712 | 0.758 |
| 2007 delinquency only | 0.626 | 0.618 | 0.620 | — |
| 2007 balance sheet + 2007 delinquency | 0.783 | 0.795 | 0.749 | — |
| 2009 balance sheet (same-year benchmark) | 0.839 | 0.875 | 0.808 | 0.842 |

*Notes: SCF 2007–2009 panel; family-grouped five-fold cross-validation with nested tuning. Standard errors are
0.007–0.02; confidence intervals in `results/panel/tables/cv_metrics_ci.csv`.*

![Figure 5](../results/panel/figures/figP1_prospective_auc.png)

*Figure 5. Cross-validated AUC for predicting 2009 delinquency from different information sets, with 95% confidence
intervals. The bottom row uses the same families' 2009 balance sheets.*

### 5.2 Persistence and onset

Delinquency is persistent: 45 percent of families that were late in 2007 were late again in 2009, compared with 13
percent of families that had been current. Persistence alone is a weak predictor, however (AUC 0.62–0.63), because
most delinquency is new. About 60 percent of the families late in 2009 had been current in 2007, and their onset is
predictable from 2007 balance sheets with an AUC of 0.76 (0.79 for serious delinquency). Among families with debt in
2007, debt detail again adds little beyond liquidity and wealth.

### 5.3 What predicts ahead

Permutation importance in the prospective models assigns most of the signal to the wealth-and-liquidity and
debt-and-housing blocks. Demographics matter more than in the cross-section, plausibly reflecting life-cycle
differences in exposure to the crisis, while the income block contributes nothing once wealth is known. Prior
delinquency is the most important single variable, followed in the logistic model by liquid assets, education and age.

---

## 6. Discussion

**Liquidity as the margin of distress.** Both designs point to the same mechanism. What separates borrowers who fall
behind from those who do not is less how much they owe than whether they have a liquid buffer relative to their
obligations. This is consistent with the view that delinquency arises when income or expense shocks meet thin
liquidity, rather than from high leverage alone.

**How much of the signal is forward-looking.** Table 5 compares the two designs. Measured at the same time as the
outcome, the balance sheet improves on demographics by 0.15–0.18 AUC; measured two years earlier, by 0.065. Roughly
half of the cross-sectional signal therefore anticipates future distress, and the rest reflects distress already
under way—depleted savings and accumulated balances that accompany missed payments.

**Table 5. Same-year versus two-years-ahead information**

| | Same-year information | Two years earlier |
|---|---|---|
| Gain of balance sheet over demographics (AUC) | 0.15–0.18 | 0.065 |
| Best AUC | 0.86 (2025 test); 0.84–0.88 (panel, 2009) | 0.78–0.80 (panel, 2007 with past delinquency) |
| Calibration | shifts across eras | stable within a period |

**Implications for measurement and early warning.** The cross-sectional models can impute recent payment problems in
surveys and administrative data that record finances but not delinquency, and they describe which households are
under strain at a point in time. For early warning, the panel results show that a single balance-sheet snapshot
identifies at-risk households years ahead, including those with no history of delinquency, but that predicted
probabilities must be recalibrated to current conditions: the ranking of households is stable, while the level of
delinquency moves with the macroeconomy.

**Scope.** The cross-sectional results describe recent rather than future delinquency, the panel covers a single
crisis episode, and the importance measures describe predictive content rather than causal effects.

---

## 7. Conclusion

Household balance sheets contain substantial information about debt-payment problems, and that information is
concentrated in liquidity. Simple, transparent models extract it as well as flexible ones; the resulting rankings
hold up across fifteen years of survey data and two years into the future, while the overall level of delinquency
shifts with economic conditions. A natural next step is to combine balance-sheet rankings with aggregate indicators,
so that early-warning models inherit the stability of household-level rankings and the timeliness of macroeconomic
data.

---

## Appendix A. Data Construction and Validation

**Fed program port.** A Python interpreter executes the Federal Reserve's summary-variable programs
(`bulletin.macro.txt` for the cross-sections, `fedstables.macro.txt` for the panel). For the cross-sections, all
350 variables match the Fed's published extracts exactly for every family-implicate in all five waves. For the panel,
173 of 188 variables (2007) and 181 of 188 (2009) match exactly; the differences come from one family whose stock
holdings were revised in a later data release and from one extract variable published in nominal rather than real
dollars.

**Labels.** `LATE = 1` if the household reports a late or missed loan payment (`X3004 = 5`); `LATE60 = 1` if any
payment was 60+ days late (`X3005 = 1`). Families without loan payments are coded 0.

**Weights and implicates.** `WGT = X42001/5` in the cross-sections; panel weight `P42001`. Each model is fit per
implicate; implicates of a family always share a fold.

## Appendix B. Estimation Details

| | Classification | Prediction |
|---|---|---|
| Tuning | forward-chaining folds (2010–13→2016, 2010–16→2019) | 5 outer × 3 inner family-grouped folds |
| Logit penalty `C` | {0.001, …, 100}; selected 0.1 for the full model | selected mostly 0.01–0.1 |
| XGBoost | depth {2, 3, 4, 6}, learning rate 0.05, early stopping; selected depth 3–4, 126–185 trees | depth {2, 3, 4}; selected mostly depth 2 |
| Selection criterion | validation log loss, averaged over folds and implicates | inner log loss, averaged over folds and implicates |

## Appendix C. Robustness

| Check | Result |
|---|---|
| Debt sample defined by payments due in the past year | AUC within 0.01 of the baseline debt sample |
| Removing buy-now-pay-later balances from 2025 debt totals | AUC changes by less than 0.004 |
| Variation across implicates | ±0.005–0.01 AUC |
| Subgroups by age and income quintile (2025) | AUC 0.80–0.95 in every group (Figure C1) |
| Tuning sensitivity | validation log loss varies by less than 0.002 across XGBoost depths |

![Figure C1](../results/figures/fig6_subgroups_2025.png)

*Figure C1. Discrimination and calibration by age and income quintile, 2025 test wave.*

## Appendix D. Replication

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e '.[ml,explain,dev]' pyarrow
.venv/bin/python -m scf_distress.pipeline --config config.yaml --build
.venv/bin/python -m scf_distress.pipeline --config config.yaml --run --explain --variance --figures
.venv/bin/python -m scf_distress.pipeline --config config.yaml --panel
.venv/bin/python -m pytest -q
```

The SCF microdata are public and are downloaded from the Federal Reserve Board; file sources and checksums are listed
in `data/PROVENANCE.json`. The full pipeline runs in under ten minutes on a laptop. Detailed results, including every
table behind this paper, are in `results/report.md` (cross-sections) and `results/panel/report.md` (panel).
