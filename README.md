# scf-distress-ml

Retrospective classification of household delinquency in the Survey of Consumer Finances:

- `LATE` — any loan payment late or missed in the past 12 months (`X3004 == 5`)
- `LATE60` — any payment 60+ days past due in the past 12 months (`X3005 == 1`)

This is **not** prospective default forecasting and **not** causal inference. See `docs/proposal.md`.

## Design (current)

| Role | Waves |
|---|---|
| Tuning (forward-chaining) | train 2010–2013 → validate 2016; train 2010–2016 → validate 2019 |
| Final training | 2010, 2013, 2016, 2019 |
| Locked out-of-sample test | 2025 (scored once) |
| Excluded | 2022 (all roles) |

Data source: **full public SCF datasets** (raw X-variables). Summary variables (`INCOME`, `NETWORTH`,
`PIRTOTAL`, ...) are rebuilt in Python from the Fed's `bulletin.macro.txt`, with dollar amounts converted to 2025
dollars using the macro's CPI-U-RS factors. Models are fit separately on each of the five implicates and
results are reported as mean and range across implicates.

## Data layout

Put each wave's full public dataset in its own folder (gitignored):

```text
data/raw/2010/   data/raw/2013/   data/raw/2016/   data/raw/2019/   data/raw/2025/
```

Replicate-weight files (`scf<year>rw1*`) go in the same folder as the main file for that wave.

## Commands

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e '.[ml,explain,dev]' pyarrow
.venv/bin/python -m scf_distress.pipeline --config config.yaml --build   # Fed variables from raw data
.venv/bin/python -m scf_distress.pipeline --config config.yaml --run     # audit, tuning, locked 2025 test
.venv/bin/python -m pytest -q
```

Outputs (`results/tables/`): `data_audit_by_year.csv`, `outcome_prevalence.csv`, `tuning_results.csv`,
`test_by_implicate.csv`, `model_performance.csv` (mean/min/max over implicates), `logit_coefficients.csv`;
predictions in `results/predictions/test_2025.parquet`; versions, hyperparameters and timings in
`results/run_metadata.json`.

## Building the Fed summary variables

`src/scf_distress/sasport.py` runs the Fed's own SAS macro (`docs/bulletin.macro.txt`) in Python: it resolves
the year branches for the public data and executes the DATA step vectorized over records. `prepare.build_wave`
wraps it. For 2010, 2013, 2016, 2019 and 2025, all 350 variables computed in the macro's main DATA step match the
Fed summary extract **exactly for every family-implicate** (`results/tables/macro_validation.csv`; checked in
`tests/test_sasport.py`). The 10 percentile-category variables computed after the main step (`NWCAT`, `INCCAT`, ...)
are not ported. Dollar values are in 2025 dollars (`cpibase=4789`).

Processed waves: `data/processed/bullit_<year>.parquet` (macro variables plus the raw X-variables they use).
File checksums and sources: `data/PROVENANCE.json`. The summary extracts in `data/external/` are used only for
this check, never as model inputs.

## Status

- [x] Config, temporal splits, weight handling, weighted metrics, unit tests
- [x] Data loading and ID/implicate validation
- [x] Port of Fed summary-variable macro, checked against the summary extracts
- [x] Feature construction (whitelist, blocks M1–M4), samples A/B/B2
- [x] Models, locked 2025 test, interpretation, figures, report (`results/report.md`)
- [x] Replicate-weight confidence intervals
- [x] 2007–2009 panel extension: prospective prediction (`--panel`; `results/panel/report.md`)
- [ ] Raw-weight and modern-waves-only robustness; historical recalibration; expanded behavioural model
