"""Classifiers: constant baseline, L2 logistic regression, gradient boosting (XGBoost).

All fitting uses survey weights. Standardization for the logit is fit on the
training rows only (weighted).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

LOGIT_GRID = [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
XGB_GRID = [{"max_depth": d} for d in (2, 3, 4, 6)]
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5.0,
                 reg_lambda=1.0, tree_method="hist", objective="binary:logistic", eval_metric="logloss")
XGB_MAX_TREES = 3000
XGB_EARLY_STOP = 100


@dataclass
class Fitted:
    kind: str
    obj: object
    params: dict
    extra: dict

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            return np.full(len(X), self.obj)
        if self.kind == "logit":
            mu, sd, clf = self.obj
            return clf.predict_proba((X - mu) / sd)[:, 1]
        if self.kind == "xgb":
            return self.obj.predict_proba(X)[:, 1]
        raise ValueError(self.kind)


def fit_constant(X, y, w, **_) -> Fitted:
    return Fitted("constant", float(np.average(y, weights=w)), {}, {})


def _weighted_scaler(X, w):
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w))
    sd[sd < 1e-12] = 1.0
    return mu, sd


def fit_logit(X, y, w, C: float = 1.0, seed: int = 0, **_) -> Fitted:
    mu, sd = _weighted_scaler(X, w)
    clf = LogisticRegression(C=C, max_iter=5000, tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit((X - mu) / sd, y, sample_weight=w)
    converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
    return Fitted("logit", (mu, sd, clf), {"C": C}, {"coef": clf.coef_[0] / sd, "coef_std": clf.coef_[0],
                                                     "intercept": float(clf.intercept_[0]),
                                                     "converged": converged})


def fit_xgb(X, y, w, max_depth: int = 3, n_estimators: int | None = None, X_val=None, y_val=None, w_val=None,
            seed: int = 0, n_jobs: int = 2, **_) -> Fitted:
    from xgboost import XGBClassifier

    if n_estimators is None:  # tuning: early stopping on the validation wave
        clf = XGBClassifier(n_estimators=XGB_MAX_TREES, max_depth=max_depth, early_stopping_rounds=XGB_EARLY_STOP,
                            random_state=seed, n_jobs=n_jobs, **XGB_FIXED)
        clf.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
        best = int(clf.best_iteration) + 1
    else:
        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=n_jobs,
                            **XGB_FIXED)
        clf.fit(X, y, sample_weight=w, verbose=False)
        best = n_estimators
    return Fitted("xgb", clf, {"max_depth": max_depth, "n_estimators": best}, {})


FITTERS = {"constant": fit_constant, "logit": fit_logit, "xgb": fit_xgb}
GRIDS = {"constant": [{}], "logit": LOGIT_GRID, "xgb": XGB_GRID}
