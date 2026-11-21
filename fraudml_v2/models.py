"""Saved, schema-aware models using one fitted preprocessing pipeline."""
from __future__ import annotations

import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import TrainQuantileClipper, build_preprocessor, canonicalize_codes


class FraudModel:
    def __init__(self, spec, plan, *, max_iter=None):
        self.spec = dict(spec)
        self.plan = plan
        self.max_iter = max_iter or plan["logistic_defaults"]["max_iter"]

    def fit(self, X, y):
        self.columns = list(X.columns)
        forbidden = {"TransactionID", "TransactionDT", "isFraud", "day_id"}
        if forbidden.intersection(self.columns):
            raise ValueError("Identifier, time or outcome entered model inputs")
        seed = self.plan["seeds"]["model"]
        if self.spec["family"] == "logistic":
            settings = {key: value for key, value in self.plan["logistic_defaults"].items() if key != "ranking_score"}
            settings["max_iter"] = self.max_iter
            estimator = LogisticRegression(C=self.spec["C"], random_state=seed, **settings)
        else:
            settings = {key: value for key, value in self.plan["lightgbm_defaults"].items() if key != "ranking_score"}
            estimator = LGBMClassifier(random_state=seed, verbosity=-1, **settings)
        clip = TrainQuantileClipper(enabled=self.spec.get("clipping", False))
        quantiles = self.plan["preprocessing"]["clipping"]["quantiles"]
        clip.low, clip.high = quantiles
        self.pipeline = Pipeline([
            ("clip", clip), ("preprocess", build_preprocessor(X)), ("model", estimator),
        ])
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.pipeline.fit(X, y)
        self.fit_seconds = perf_counter() - started
        self.warnings = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in caught]
        self.converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
        return self

    def prepare(self, X):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model fields: {sorted(missing)}")
        frame = X.loc[:, self.columns]
        frame = canonicalize_codes(frame, self.plan["preprocessing"]["canonical_integer_codes"])
        for name, _, columns in self.pipeline.named_steps["preprocess"].transformers_:
            if name == "num":
                for column in columns:
                    frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame

    def predict(self, X, *, prepared=False):
        frame = X.loc[:, self.columns] if prepared else self.prepare(X)
        clipped = self.pipeline.named_steps["clip"].transform(frame)
        transformed = self.pipeline.named_steps["preprocess"].transform(clipped)
        estimator = self.pipeline.named_steps["model"]
        # LightGBM generates internal feature names for this transformed matrix.
        # Input columns and fitted transform order have already been checked above.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)
            probability = np.asarray(estimator.predict_proba(transformed)[:, 1], dtype=float)
            if self.spec["family"] == "logistic":
                margin = np.asarray(estimator.decision_function(transformed), dtype=float)
            else:
                margin = np.asarray(estimator.predict(transformed, raw_score=True), dtype=float)
        if not np.isfinite(probability).all() or not np.isfinite(margin).all():
            raise ValueError("Model returned nonfinite predictions")
        score = margin if self.spec["family"] == "logistic" else probability
        return {"score": score, "probability": probability, "margin": margin}

    def diagnostics(self):
        estimator = self.pipeline.named_steps["model"]
        n_iter = getattr(estimator, "n_iter_", None)
        return {
            "spec": self.spec, "converged": self.converged,
            "max_iter": self.max_iter if self.spec["family"] == "logistic" else None,
            "iterations": np.asarray(n_iter).tolist() if n_iter is not None else None,
            "fit_seconds": self.fit_seconds, "warnings": self.warnings,
            "clip_bounds": self.pipeline.named_steps["clip"].bounds_,
            "input_fields": self.columns,
            "parameters": estimator.get_params(),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=1)

    @staticmethod
    def load(path):
        return joblib.load(path)
