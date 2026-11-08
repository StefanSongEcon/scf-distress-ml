# Research and Implementation Proposal
## Machine Learning Classification of Household Financial Distress Using the Survey of Consumer Finances

**Project:** SCF Financial Distress Classification  
**Primary outcomes:** `LATE` and `LATE60`  
**Data:** Public-use Survey of Consumer Finances (SCF), Federal Reserve Board  
**Implementation:** Python; a fully reproducible, tested pipeline  
**Scope:** Retrospective classification and out-of-survey-wave generalization, **not** prospective default forecasting or causal identification.

---

## 1. Research objective

Use household-level financial, balance-sheet, debt, and demographic information in the SCF to classify households that experienced delinquency in the past twelve months. The empirical questions are:

1. How accurately can observable household characteristics classify **any late payment** (`LATE`)?
2. How accurately can they classify **serious delinquency of 60+ days** (`LATE60`)?
3. How much incremental predictive information is contained in income, wealth/liquidity, and debt burden beyond basic household demographics?
4. Which features and nonlinearities account for performance, and how stable is the model when applied to a **new SCF survey wave**?
5. How do the results differ between **all households** and **currently indebted households**?

The project should deliver transparent econometric benchmarks, machine-learning classifiers, interpretable feature analyses, and a fully reproducible empirical pipeline. The project does **not** attempt age-period-cohort decomposition; that is a separate, parked proposal.

## 2. Official data sources and variable verification

**Primary source:** Federal Reserve Board SCF public microdata. Use the **Summary Extract Public Data (CSV)** for each survey wave. The extract contains consistently named constructed variables and the dollar fields are harmonized to **2025 dollars** in the current official extract release.

- Current SCF data and CSV downloads: https://www.federalreserve.gov/econres/scfindex.htm
- Earlier waves: https://www.federalreserve.gov/econres/scf-previous-surveys.htm
- Official codebook: https://www.federalreserve.gov/apps/scfcb/
- Official summary-variable construction SAS macro: https://www.federalreserve.gov/econres/files/bulletin.macro.txt
- Fed FAQ regarding weights and multiple imputation: https://www.federalreserve.gov/econres/scf_faqs.htm

**Outcome verification:** The official macro defines `LATE = (X3004 == 5)` and `LATE60 = (X3005 == 1)`. Use the **provided `LATE` and `LATE60` columns** from the summary extract as the canonical labels. Do not recode skipped responses from raw questionnaire fields without examining official routing and coding. Cross-check each wave's label distributions against Fed publications where comparable.

**Date policy:** First implement **2010, 2013, 2016, 2019, 2022, 2025** for a reasonable combination of historical sample size and harmonized features. Do not assume that every proposed feature is consistently observed in every year. After successful baseline, implement an **optional 1989–2025 extension** using the intersecting feature set, subject to documented definition and questionnaire comparability. Prefer a smaller reliable set of years to an inadequately harmonized larger sample.

The download script should discover/download official files, cache originals without modifying them, verify ZIP integrity, record source URL, download date and checksum, and support an offline mode with locally provided ZIP/CSV files. Never silently replace unavailable data with simulated data.

## 3. Outcomes and sample definitions

For each family `i` in survey `t`, define:

- **Any delinquency:** `Y_late(i,t) = 1[LATE == 1]`.
- **Serious delinquency:** `Y_late60(i,t) = 1[LATE60 == 1]`.

These outcomes are **past-12-month delinquency indicators**, not lifetime default, charge-off, bankruptcy, or future repayment failure.

Two estimation populations:

- **Sample A (all families):** Retain eligible SCF families with valid survey weights and an interpretable outcome. This is the primary population-level descriptive classification.
- **Sample B (indebted families):** Restrict to families with `HDEBT == 1` (or an equivalent validated current-debt indicator). This is the critical conditional-risk exercise because in Sample A the absence of current borrowing may mechanically help classification. Note that current indebtedness is measured at interview and might not exactly represent debt exposure throughout the preceding twelve months; document this selection limitation.

For each wave and population, report: unique families, five-implicate record count, counts and survey-weighted proportions for `LATE`/`LATE60`, and the share with nonmissing/valid labels. Investigate all apparent logical violations of `LATE60 <= LATE`, unexpected missings, and survey routing before modeling. Do **not** automatically code missing labels as zero.

## 4. Predictor construction and leakage prevention

Create a **whitelist-only** set of predictors. Start with the categories below, validating availability and definitions by survey wave. Use continuous variables when available and meaningful, plus carefully constructed ratios and binary indicators. Do not pass unidentified SCF columns into the model.

| Block | Candidate SCF variables | Economic interpretation |
|---|---|---|
| Demographics | `AGE`, `EDCL` (prefer to raw `EDUC` for long panels), `MARRIED`, `KIDS`, `FAMSTRUCT` | Life cycle, schooling, household composition |
| Income | `INCOME`, `WAGEINC`, `NORMINC` | Earnings and income capacity |
| Wealth and liquidity | `ASSET`, `FIN`, `LIQ`, `NETWORTH`, `RETQLIQ` | Wealth buffers and liquid resources |
| Debt and credit exposure | `HDEBT`, `DEBT`, `CCBAL`, `MRTHEL`, `INSTALL`, `EDN_INST`, `VEH_INST` | Debt participation, levels, and composition |
| Payment burdens | `TPAY`, `PIRTOTAL`, `PIRCONS`, `PIRREV`, `DEBT2INC` | Debt-service and leverage pressure |
| Housing | `HOUSECL`, `HOUSES`, `HOMEEQ` | Tenure and housing wealth |
| Optional behavioral features | `SAVED`, `YESFINRISK`, `FINLIT` | Savings/risk preferences/knowledge; **use only in an explicitly labeled expanded model** after auditing year coverage and timing |

Suggested derived predictors, with explicit treatments for zero/negative denominators:

- `liquid_assets_to_income` and `liquid_assets_to_debt`;
- `unsecured_debt_share`, `mortgage_debt_share`, `education_debt_share`;
- `negative_net_worth`;
- `income_nonpositive`, `no_liquid_assets`, `no_current_debt`;
- optional `log1p` or `asinh` transforms of heavy-tailed dollar amounts, preserving signs for net worth and income.

Never replace negative income or negative net worth with zero. Pre-specify handling for divisions by zero, outliers, imputation, and missingness. Fit transformations and statistical imputers **on training data only**.

**Strictly banned as predictors in the main model:** `LATE`, `LATE60`, raw questionnaire source variables `X3004`, `X3005`, any directly duplicated or derived copies of the outcome, family IDs, implicate identifiers, survey weights, and columns that identify a held-out split. In addition, withhold other realized hardship / credit-event variables (e.g., `BNKRUPLAST5`, `FORECLLAST5`, `HPAYDAY`, `TURNDOWN`, `TURNFEAR`, `EMERGPSTP` and related emergency-payment responses) from the **core financial-fundamentals model**. They can be assessed later as a *separately labeled* expanded-information model with a full temporal/leakage caveat.

`YEAR` is allowed for descriptive tables, stratification, and validation logic but **not** as a main predictor for out-of-time forecasting. Document explicitly which covariates are contemporaneous or may reflect consequences of prior delinquency. Such models classify retrospective distress and must not be marketed as models that predict next year's default.

## 5. Missing data, survey weights, and five implicates

The SCF uses complex sampling and **five multiple-imputation implicates**, not five independent households. Treat a family-year as one conceptual household observation.

Required design:

1. Detect and document the correct family ID and implicate ID fields from the official files; do not assume column names without inspecting the extracts. Construct an unambiguous `survey_year + family_id` key and verify **exactly five implicates per retained family**, subject to documented exceptions.
2. Use the official SCF survey weight (`WGT` in summary extracts). The current official summary macro sets `WGT = X42001 / 5`; **do not divide `WGT` by five again** if it is already scaled. For single-implicate estimates, uniform rescaling of its weights does not change normalized metrics.
3. Recommended baseline: train and evaluate each model **separately on implicates 1–5**, using identical wave-based splits, then report the average and range across implicates. Where household-level predicted probabilities are pooled, align the same families carefully and distinguish averaged predictions from imputation-pooled uncertainty.
4. Never let different implicates of the same family fall into different train/test partitions. Never count five implicates as five independent survey households.
5. All headline event rates, classification metrics, and calibration summaries should be **survey-weighted**. Provide unweighted counterparts as descriptive diagnostics.
6. For formal confidence intervals, use the Fed's **replicate weights plus multiple-imputation variance-combination guidance** when practical, especially for population summaries. If model-metric uncertainty is estimated using approximate resampling instead, label it explicitly as approximate and do not call it design-correct SCF inference.

Use weights both in estimator fitting (when supported) and evaluation. To keep early/late waves from dominating purely because population weights differ in scale, normalize training weights transparently **within survey year** for the baseline, and run a sensitivity check using raw household population weighting. Clearly distinguish the target distributions of the two schemes.

## 6. Temporal validation design

The key evaluation asks whether a model trained on historical SCF surveys works on a newly released survey.

**Primary forward test:**

- **Training:** 2010, 2013, 2016, 2019.
- **Validation / hyperparameter choice:** 2022.
- **Final locked test:** 2025, used only once after model specifications are fixed.

**Additional robustness:**

- Forward-chaining backtests (e.g., train through 2013, validate 2016, test 2019; train through 2016, validate 2019, test 2022), using only waves available at each point.
- Modern-wave-only robustness (train 2016/2019, validate 2022, test 2025) to assess whether older data reduce transportability.
- Evaluate 2025 by major age/income/debt participation strata, with cell sample sizes and uncertainty caveats.

Keep **all** preprocessing, imputation, scaling, feature-selection, calibration fitting, and hyperparameter tuning inside the development sample. Never use 2025 data to decide features, winsorization cutoffs, tuning ranges, thresholds, or reweighting rules. Any probability calibration must be learned on historical held-out data, not on 2025.

## 7. Models and nested feature-block experiments

Run both outcomes on both sample definitions. For each outcome/population/implicate:

**Models:**

1. **Constant-probability baseline:** weighted mean positive rate from the training sample.
2. **Regularized logistic regression:** standardized or transformed continuous features, encoded categorical variables, Elastic Net or L2 with carefully tuned regularization. Ensure proper weighting.
3. **Histogram Gradient Boosting or XGBoost:** a non-linear tabular classifier with early stopping/tuned depth and shrinkage. Prefer XGBoost where it is available; provide a sklearn-only fallback.

Avoid elaborate model stacks or deep neural networks until simpler models outperform meaningful baselines out of time.

**Four nested feature blocks:**

- **M1 — Demographics:** age, education, marital status, children, household structure.
- **M2 — M1 + Income:** total income, wage income, normal income.
- **M3 — M2 + Wealth:** total assets, financial assets, liquid assets, net worth.
- **M4 — M3 + Debt and housing:** balances, composition, payment-to-income, leverage, housing equity/tenure.

Core scientific result: the **incremental out-of-time predictive value** of successive financial-information blocks relative to demographics alone. Run optional expanded models with savings/risk preferences as a *separate appendix-style* exercise only if those variables are available and comparable.

## 8. Evaluation metrics and interpretation

Calculate weighted metrics for the **2025 held-out wave** and, separately, for validation/backtest waves:

- ROC-AUC;
- Average precision (PR-AUC), compared against the **weighted prevalence** baseline;
- Brier score and, if robustly computable, log loss;
- Calibration plot (weighted bins; show counts/effective sample sizes) and intercept/slope diagnostics;
- Recall among the top 10% of predicted risk **by survey weight**, and event lift relative to random selection;
- Precision/recall at a threshold chosen exclusively on the validation sample (no arbitrary test-set optimization).

Report both discrimination and calibration; avoid overall accuracy as the headline metric, especially for the rarer `LATE60` outcome. Never describe this as estimating a causal effect of any feature.

## 9. Feature importance and substantive economic analysis

For the best-performing validated model:

- Standardized coefficients and uncertainty caveats for penalized logistic regression;
- **Held-out weighted permutation importance** where feasible;
- SHAP summary and dependence plots for the boosted model, if implemented and computable;
- One- or two-way response plots for `AGE`, `LIQ`, `DEBT2INC`, `PIRTOTAL`, `NETWORTH`, and credit card debt;
- Discrimination/calibration by age and income groups, and compare all-family versus borrower-only samples.

Do not interpret SHAP, partial dependence, or permutation importance as causal effects. Check correlation and redundancy: `DEBT`, `PIRTOTAL`, `DEBT2INC`, and asset aggregates can be strongly related. For nonlinear curves, make it clear whether plots are conditional associations or synthetic counterfactual changes to covariates.

## 10. Expected deliverables

Create this repository structure (adjust only if justified and documented):

```text
scf-distress-ml/
├── README.md
├── pyproject.toml
├── config.yaml
├── src/scf_distress/
│   ├── download.py          # Federal Reserve public data, caching, checksums
│   ├── schema.py            # Wave/variable inventory and harmonization checks
│   ├── prepare.py           # Family + implicate IDs, target validation, features
│   ├── weights.py           # Survey/imputation-aware weighted calculations
│   ├── splits.py            # Chronological splits and forward backtests
│   ├── models.py            # Baselines, logistic regression, boosting
│   ├── evaluate.py          # Weighted metrics and calibration
│   ├── explain.py           # Feature importance and SHAP (optional)
│   ├── figures.py           # Publication-quality plots
│   └── pipeline.py          # CLI orchestration
├── tests/
│   ├── test_targets.py
│   ├── test_imputation_ids.py
│   ├── test_weights.py
│   ├── test_temporal_leakage.py
│   └── test_metrics.py
├── data/raw/                # Official data; gitignored
├── data/processed/          # Clean files; gitignored
└── results/
    ├── tables/
    ├── figures/
    ├── predictions/
    ├── models/
    └── report.md
```

**Required tables:**

- `data_audit_by_year.csv`: survey year, source, unique families, implicates, missingness, variables and definition warnings.
- `outcome_prevalence.csv`: weighted delinquency rates by survey year and sample definition.
- `model_performance.csv`: each outcome × population × feature block × algorithm × test year, with mean and range across implicates.
- `feature_importance.csv`: named predictor, signed/logistic coefficient or weighted importance, by model and outcome.

**Required figures:**

1. Weighted `LATE`/`LATE60` prevalence by survey year.
2. ROC curves and precision-recall curves for 2025.
3. 2025 calibration plots and predicted-risk histograms, separately for `LATE` and `LATE60`.
4. Nested-block incremental predictive-value chart (especially PR-AUC and Brier improvement).
5. Permutation importance and optional SHAP summary.
6. Subgroup performance (at minimum age and income, clearly labeling unstable small cells).

**Required report (`results/report.md`):** data provenance, label definitions, sample construction, wave coverage, model configurations, sample sizes, weight/imputation procedures, descriptive results, locked 2025 test metrics, feature interpretations, robustness checks, failures and caveats. Include a section on **retrospective classification vs prospective prediction**.

## 11. Non-negotiable quality checks

- Confirm `LATE`, `LATE60` are genuine binary variables in each used year and inspect the `LATE60 <= LATE` relationship.
- Confirm the actual columns and coding of family ID and implicate ID, as well as `WGT` treatment, from the data and official documentation.
- No family/implicate overlap or fit/transform leakage between historical development and 2025 test.
- Do not include banned target-derived, distress-event, post-outcome, ID, or test-identifying variables in the baseline predictors.
- Compare weighted sample prevalence with credible official estimates; clearly distinguish differences due to borrower restrictions.
- If 2010–2025 schema is inconsistent, **document and drop/replace only problematic features**—not entire waves without explanation.
- Use a reproducible fixed random seed and record library versions.
- Unit tests for leakage, weight scaling, ID grouping, time splits, and weighted metrics must pass.
- Never manufacture sample counts, model results, variable coverage, or successful downloads.

## 13. Suggested scientific interpretation

The central comparison is not merely *Logit versus XGBoost*. It is:

> **How much additional information about recent household delinquency is contained in financial balance sheets and debt-service measures, beyond basic demographic and income characteristics?**

A second contribution is assessing whether patterns estimated in earlier SCF cross-sections transport to the newly available 2025 wave. This is a descriptive/predictive exercise; it does not identify what causes delinquency.

---

**Sources:** Federal Reserve Board SCF 2025 data page, official summary-variable SAS macro, SCF historical survey page, SCF FAQs and codebook, URLs above. All results must be generated rather than assumed.
