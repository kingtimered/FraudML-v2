"""Deterministic ranking and matched-capacity daily and period evaluations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def rank_order(scores, times, ids):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite")
    return np.lexsort((np.asarray(ids), np.asarray(times), -scores))


def metrics_at_k(y, scores, times, ids, k, *, ap=True):
    y = np.asarray(y, dtype=int)
    if not np.isin(y, [0, 1]).all() or len(y) != len(scores):
        raise ValueError("Expected aligned binary labels and scores")
    if not 0 <= k <= len(y):
        raise ValueError("Capacity outside sample size")
    order = rank_order(scores, times, ids)
    hits = int(y[order[:k]].sum())
    fraud = int(y.sum())
    return {
        "rows": len(y), "fraud": fraud, "reviews": int(k), "hits": hits,
        "precision": hits / k if k else None,
        "recall": hits / fraud if fraud else None,
        "average_precision": float(average_precision_score(y, scores)) if ap and fraud else None,
    }


def scenario_names(plan):
    return [f"rate_{plan['budget']['daily_rate']:g}"] + [f"fixed_{budget}" for budget in plan["budget"]["daily_fixed"]]


def daily_capacity(count, scenario):
    if scenario.startswith("rate_"):
        return min(count, math.ceil(count * float(scenario[5:])))
    return min(count, int(scenario[6:]))


def evaluate_period(frame, scores, plan, *, keep_queues=True):
    y = frame["isFraud"].to_numpy(dtype=int)
    times = frame["TransactionDT"].to_numpy()
    ids = frame["TransactionID"].to_numpy()
    days = times // plan["data"]["day_seconds"]
    global_order = rank_order(scores, times, ids)
    ap = float(average_precision_score(y, scores)) if y.sum() else None
    results = {"period_rates": {}, "legacy_budgets": {}, "daily": {}}
    for rate in plan["budget"]["period_rates"]:
        row = metrics_at_k(y, scores, times, ids, min(len(y), math.ceil(rate * len(y))), ap=False)
        row["average_precision"] = ap
        row["random_expected_hits"] = row["reviews"] * row["fraud"] / row["rows"]
        results["period_rates"][f"{rate:g}"] = row
    for k in [100, 500]:
        if k <= len(y):
            row = metrics_at_k(y, scores, times, ids, k, ap=False)
            row["average_precision"] = ap
            results["legacy_budgets"][str(k)] = row
    daily_rows, queue_rows, weekly_rows = [], [], []
    first_day = int(days.min())
    for scenario in scenario_names(plan):
        selected = []
        day_records = []
        for day in np.unique(days):
            positions = np.flatnonzero(days == day)
            k = daily_capacity(len(positions), scenario)
            order = rank_order(np.asarray(scores)[positions], times[positions], ids[positions])
            chosen = positions[order[:k]]
            selected.extend(chosen.tolist())
            positives = int(y[positions].sum())
            hits = int(y[chosen].sum())
            record = {
                "scenario": scenario, "day_id": int(day),
                "week_index": int((day - first_day) // 7),
                "rows": len(positions), "fraud": positives, "reviews": k,
                "hits": hits, "precision": hits / k if k else None,
                "recall": hits / positives if positives else None,
                "random_expected_hits": k * positives / len(positions),
            }
            day_records.append(record)
            if keep_queues:
                for rank, index in enumerate(chosen, 1):
                    queue_rows.append({
                        "scenario": scenario, "day_id": int(day), "review_rank": rank,
                        "TransactionID": int(ids[index]), "TransactionDT": int(times[index]),
                        "score": float(scores[index]),
                    })
        daily_rows.extend(day_records)
        total_k = len(selected)
        hit_count = int(y[selected].sum())
        matched_hits = int(y[global_order[:total_k]].sum())
        results["daily"][scenario] = {
            "rows": len(y), "fraud": int(y.sum()), "reviews": total_k,
            "hits": hit_count, "precision": hit_count / total_k if total_k else None,
            "recall": hit_count / int(y.sum()) if y.sum() else None,
            "random_expected_hits": sum(row["random_expected_hits"] for row in day_records),
            "matched_global": {
                "reviews": total_k, "hits": matched_hits,
                "precision": matched_hits / total_k if total_k else None,
                "recall": matched_hits / int(y.sum()) if y.sum() else None,
            },
        }
        daily_table = pd.DataFrame(day_records)
        for week, group in daily_table.groupby("week_index", sort=True):
            reviews, fraud, hits = int(group.reviews.sum()), int(group.fraud.sum()), int(group.hits.sum())
            weekly_rows.append({
                "scenario": scenario, "week_index": int(week),
                "days": len(group), "partial_week": len(group) < 7,
                "rows": int(group.rows.sum()), "fraud": fraud,
                "reviews": reviews, "hits": hits,
                "precision": hits / reviews if reviews else None,
                "recall": hits / fraud if fraud else None,
            })
    return results, pd.DataFrame(daily_rows), pd.DataFrame(weekly_rows), pd.DataFrame(queue_rows)


class WeightedRanking:
    """Exact duplicated-day metrics without materializing bootstrap row copies."""
    def __init__(self, frame, scores):
        scores = np.asarray(scores, dtype=float)
        self.order = rank_order(scores, frame.TransactionDT, frame.TransactionID)
        self.y = frame.isFraud.to_numpy(dtype=int)[self.order]
        days = frame.TransactionDT.to_numpy() // 86400
        self.days, day_codes = np.unique(days, return_inverse=True)
        self.day_codes = day_codes[self.order]
        sorted_scores = scores[self.order]
        self.tie_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])

    def compute(self, day_weights, rate, *, capacity=None):
        weights = np.asarray(day_weights, dtype=int)[self.day_codes]
        counts = np.cumsum(weights)
        tp = np.cumsum(weights * self.y)
        total = int(counts[-1])
        positives = int(tp[-1])
        k = min(total, math.ceil(rate * total)) if capacity is None else int(capacity)
        if not 0 <= k <= total:
            raise ValueError("Invalid weighted capacity")
        hits = 0
        if k:
            end = int(np.searchsorted(counts, k, side="left"))
            before_count = int(counts[end - 1]) if end else 0
            before_hits = int(tp[end - 1]) if end else 0
            hits = before_hits + (k - before_count) * int(self.y[end])
        if positives:
            cumulative_tp, cumulative_count = tp[self.tie_ends], counts[self.tie_ends]
            increments = np.diff(np.r_[0, cumulative_tp])
            precision = np.divide(cumulative_tp, cumulative_count, out=np.zeros(len(cumulative_tp), dtype=float), where=cumulative_count > 0)
            ap = float(np.sum(increments * precision) / positives)
        else:
            ap = np.nan
        return np.asarray([hits, hits / k if k else np.nan, hits / positives if positives else np.nan, ap])
