"""Train-only encodings and strictly-past entity features.

The vectorized builder is used in experiments. StreamingHistory is an
independent stateful implementation used to check arbitrary chunk boundaries.
"""
from __future__ import annotations

import math
import numbers
import re
from collections import deque
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def integer_code(value):
    """Canonicalize a scalar integer-coded proxy independently of its batch."""
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean values are not integer entity codes")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if math.isnan(value):
            return np.nan
        if not math.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"Non-integer or nonfinite entity code: {value!r}")
        return str(int(value))
    if not isinstance(value, str):
        raise ValueError(f"Unsupported entity code type: {type(value).__name__}")
    try:
        code = Decimal(value.strip())
    except InvalidOperation as error:
        raise ValueError(f"Invalid integer entity code: {value!r}") from error
    if not code.is_finite() or code != code.to_integral_value():
        raise ValueError(f"Non-integer or nonfinite entity code: {value!r}")
    return str(int(code))


def canonicalize_codes(frame, columns=("card1", "addr1")):
    out = frame.copy(deep=False)
    for column in columns:
        if column not in out:
            raise KeyError(column)
        out[column] = out[column].map(integer_code).astype(object)
    return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.maps_ = {
            column: frame[column].astype("string").value_counts(dropna=True) / len(frame)
            for column in frame
        }
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        if list(frame.columns) != self.columns_:
            raise ValueError("Frequency encoder columns differ from fitted schema")
        return np.column_stack([
            frame[column].astype("string").map(self.maps_[column]).fillna(0).to_numpy(dtype=float)
            for column in self.columns_
        ])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class TrainQuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=False, low=0.001, high=0.999):
        self.enabled, self.low, self.high = enabled, low, high

    def fit(self, X, y=None):
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("Invalid clipping quantiles")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.bounds_ = {}
        if self.enabled:
            for column in X.select_dtypes(include="number"):
                if column in {"TransactionAmt", "dist1", "dist2"} or re.fullmatch(r"[CDV][0-9]+", column):
                    values = X[column].quantile([self.low, self.high]).to_numpy()
                    if np.isfinite(values).all():
                        self.bounds_[column] = tuple(float(value) for value in values)
        return self

    def transform(self, X):
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Clipper input schema changed")
        out = X.copy(deep=False)
        for column, (lower, upper) in self.bounds_.items():
            out[column] = out[column].clip(lower, upper)
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor(train):
    numeric = train.select_dtypes(include="number").columns.tolist()
    categories = [column for column in train if column not in numeric]
    low = [column for column in categories if train[column].nunique(dropna=False) <= 32]
    high = [column for column in categories if column not in low]
    transforms = []
    if numeric:
        transforms.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if low:
        transforms.append(("cat_low", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), low))
    if high:
        transforms.append(("cat_freq", FrequencyEncoder(), high))
    return ColumnTransformer(transforms, remainder="drop", sparse_threshold=1.0)


def history_columns(entity, recent=False):
    columns = [f"{entity}__past_count", f"{entity}__past_amount_mean"]
    if recent:
        columns += [f"{entity}__count_24h", f"{entity}__seconds_since_last"]
    return columns


def validate_order(frame):
    time = frame["TransactionDT"].to_numpy(dtype=float)
    if not np.isfinite(time).all() or np.any(np.diff(time) < 0):
        raise ValueError("History input must have finite, nondecreasing timestamps")


def history_variants(frame, entities=("card1", "addr1")):
    """Return H1/H2/H3 feature tables aligned to an ordered input frame.

    Grouping by (entity, timestamp) excludes the whole current timestamp.
    Searchsorted uses a left-inclusive rolling boundary and excludes the
    current bucket, preserving strictly-past semantics at tied timestamps.
    """
    validate_order(frame)
    h1, h2, h3 = {}, {}, {}
    for entity in entities:
        groups = frame.groupby([entity, "TransactionDT"], dropna=False, sort=False, observed=True)
        buckets = groups["TransactionAmt"].agg(["size", "sum", "count"])
        totals = buckets.groupby(level=0, dropna=False, sort=False, observed=True).cumsum()
        previous = totals.groupby(level=0, dropna=False, sort=False, observed=True).shift(fill_value=0)
        bucket_ids = groups.ngroup().to_numpy()
        counts = previous["size"].to_numpy(dtype=float)[bucket_ids]
        means = (previous["sum"] / previous["count"].replace(0, np.nan)).to_numpy()[bucket_ids]
        names = history_columns(entity)
        h1[names[0]], h1[names[1]] = counts, means
        missing = frame[entity].isna().to_numpy()
        h2[names[0]] = np.where(missing, np.nan, counts)
        h2[names[1]] = np.where(missing, np.nan, means)
        rolling = np.zeros(len(buckets), dtype=float)
        elapsed = np.full(len(buckets), np.nan)
        times = buckets.index.get_level_values(1).to_numpy(dtype=float)
        sizes = buckets["size"].to_numpy()
        for positions in buckets.groupby(level=0, dropna=False, sort=False, observed=True).indices.values():
            t = times[positions]
            left = np.searchsorted(t, t - 86400, side="left")
            prefix = np.concatenate(([0], np.cumsum(sizes[positions])))
            rolling[positions] = prefix[:-1] - prefix[left]
            elapsed[positions[1:]] = np.diff(t)
        h3[names[0]], h3[names[1]] = h2[names[0]], h2[names[1]]
        h3[f"{entity}__count_24h"] = np.where(missing, np.nan, rolling[bucket_ids])
        h3[f"{entity}__seconds_since_last"] = np.where(missing, np.nan, elapsed[bucket_ids])
    return {
        "LGB-H1": pd.DataFrame(h1, index=frame.index),
        "LGB-H2": pd.DataFrame(h2, index=frame.index),
        "LGB-H3": pd.DataFrame(h3, index=frame.index),
    }


class StreamingHistory:
    """Stateful causal reference that buffers an unfinished timestamp group.

    push(final=False) may emit fewer rows than received. Call push(final=True)
    once at stream end. This is a correctness reference, not a serving benchmark.
    """
    def __init__(self, entities=("card1", "addr1"), share_missing=False):
        self.entities = tuple(entities)
        self.share_missing = share_missing
        self.pending = []
        self.states = {entity: {} for entity in entities}
        self.last_received = None
        self.closed = False

    def push(self, frame=None, *, final=False):
        if self.closed:
            raise ValueError("History stream is already finalized")
        records = [] if frame is None else frame.to_dict("records")
        for row in records:
            t = row["TransactionDT"]
            if not math.isfinite(t) or (self.last_received is not None and t < self.last_received):
                raise ValueError("Out-of-order or invalid stream timestamp")
            self.last_received = t
        self.pending.extend(records)
        if not self.pending:
            self.closed = final
            return pd.DataFrame()
        cutoff = len(self.pending)
        if not final:
            last_t = self.pending[-1]["TransactionDT"]
            while cutoff and self.pending[cutoff - 1]["TransactionDT"] == last_t:
                cutoff -= 1
        ready, self.pending = self.pending[:cutoff], self.pending[cutoff:]
        output = []
        start = 0
        while start < len(ready):
            end = start + 1
            t = ready[start]["TransactionDT"]
            while end < len(ready) and ready[end]["TransactionDT"] == t:
                end += 1
            updates = {}
            for row in ready[start:end]:
                result = {column: row[column] for column in ("TransactionID", "TransactionDT") if column in row}
                for entity in self.entities:
                    missing = pd.isna(row[entity])
                    names = history_columns(entity, recent=True)
                    if missing and not self.share_missing:
                        result.update(dict.fromkeys(names, np.nan))
                        continue
                    key = None if missing else row[entity]
                    state = self.states[entity].setdefault(key, {
                        "count": 0, "sum": 0.0, "known": 0,
                        "last": None, "queue": deque(), "recent_count": 0,
                    })
                    while state["queue"] and state["queue"][0][0] < t - 86400:
                        _, old_size = state["queue"].popleft()
                        state["recent_count"] -= old_size
                    result.update({
                        names[0]: state["count"],
                        names[1]: state["sum"] / state["known"] if state["known"] else np.nan,
                        names[2]: state["recent_count"],
                        names[3]: t - state["last"] if state["last"] is not None else np.nan,
                    })
                    update = updates.setdefault((entity, key), [])
                    update.append(row["TransactionAmt"])
                output.append(result)
            for (entity, key), amounts in updates.items():
                state = self.states[entity][key]
                known = [amount for amount in amounts if pd.notna(amount)]
                state["count"] += len(amounts)
                state["sum"] += sum(known)
                state["known"] += len(known)
                state["last"] = t
                state["queue"].append((t, len(amounts)))
                state["recent_count"] += len(amounts)
            start = end
        self.closed = final
        return pd.DataFrame(output)
