from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudml_v2.evaluation import WeightedRanking, evaluate_period, metrics_at_k
from fraudml_v2.features import (
    StreamingHistory, TrainQuantileClipper, canonicalize_codes,
    history_columns, history_variants, integer_code,
)
from fraudml_v2.models import FraudModel


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return json.loads((ROOT / "configs/study_plan.json").read_text())


def sample_frame(n=100, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "TransactionID": np.arange(10000, 10000 + n),
        "TransactionDT": np.sort(rng.integers(0, 4 * 86400, n)),
        "TransactionAmt": rng.uniform(10, 500, n),
        "card1": rng.choice([1, 2, 3, np.nan], n),
        "addr1": rng.choice([10, 20, np.nan], n),
        "C1": rng.normal(size=n), "ProductCD": rng.choice(["C", "W"], n),
        "isFraud": rng.integers(0, 2, n),
    })
    return canonicalize_codes(frame)


def naive_history(frame, shared_missing=False, recent=True):
    result = {}
    for entity in ["card1", "addr1"]:
        rows = []
        for _, row in frame.iterrows():
            if pd.isna(row[entity]) and not shared_missing:
                rows.append([np.nan] * (4 if recent else 2))
                continue
            mask = frame[entity].isna() if pd.isna(row[entity]) else frame[entity].eq(row[entity])
            past = frame.loc[mask & frame.TransactionDT.lt(row.TransactionDT)]
            values = [len(past), past.TransactionAmt.mean()]
            if recent:
                values += [int(past.TransactionDT.ge(row.TransactionDT - 86400).sum()),
                           row.TransactionDT - past.TransactionDT.max() if len(past) else np.nan]
            rows.append(values)
        for index, name in enumerate(history_columns(entity, recent)):
            result[name] = [row[index] for row in rows]
    return pd.DataFrame(result, index=frame.index)


def test_code_normalization_is_scalar_and_strict():
    assert [integer_code(value) for value in [123, 123.0, "123", "123.0", " 00123 "]] == ["123"] * 5
    for value in [None, np.nan, pd.NA]:
        assert pd.isna(integer_code(value))
    for value in [True, False, 1.5, "1.5", float("inf"), "nan", "", "abc", []]:
        with pytest.raises(ValueError):
            integer_code(value)
    alone = canonicalize_codes(pd.DataFrame({"card1": [123.0], "addr1": [10]}))
    mixed = canonicalize_codes(pd.DataFrame({"card1": [123.0, "456"], "addr1": [10, "20.0"]}))
    assert alone.card1.iloc[0] == mixed.card1.iloc[0]


def test_histories_equal_naive_with_ties_and_window_boundary():
    frame = canonicalize_codes(pd.DataFrame({
        "TransactionID": range(9),
        "TransactionDT": [0, 0, 1, 1, 86400, 86400, 86401, 90000, 180000],
        "TransactionAmt": [100, np.nan, 300, 400, 500, 600, np.nan, 800, 900],
        "card1": [1, 1, 1, np.nan, 1, 2, 1, np.nan, 1],
        "addr1": [10, 10, 20, np.nan, 10, 10, 10, np.nan, 10],
    }))
    variants = history_variants(frame)
    pd.testing.assert_frame_equal(variants["LGB-H1"], naive_history(frame, shared_missing=True, recent=False), check_dtype=False)
    pd.testing.assert_frame_equal(variants["LGB-H2"], naive_history(frame, recent=False), check_dtype=False)
    pd.testing.assert_frame_equal(variants["LGB-H3"], naive_history(frame), check_dtype=False)
    # At t=86400 the two records from t=0 remain inside [t-86400,t).
    assert variants["LGB-H3"].loc[4, "card1__count_24h"] == 3
    assert variants["LGB-H3"].loc[5, "card1__past_count"] == 0


def test_randomized_histories_and_no_future_dependence():
    frame = sample_frame(180)
    # Create timestamp ties including distinct entity values.
    frame["TransactionDT"] = frame.TransactionDT // 5000 * 5000
    variants = history_variants(frame)
    pd.testing.assert_frame_equal(variants["LGB-H3"], naive_history(frame), check_dtype=False)
    cutoff = frame.TransactionDT.iloc[90]
    prefix = frame.loc[frame.TransactionDT.lt(cutoff)]
    short = history_variants(prefix)["LGB-H3"]
    pd.testing.assert_frame_equal(short, variants["LGB-H3"].loc[prefix.index])
    altered = frame.copy()
    altered["isFraud"] = 1 - altered.isFraud
    pd.testing.assert_frame_equal(history_variants(altered)["LGB-H3"], variants["LGB-H3"])


def test_streaming_chunk_invariance_across_ties():
    frame = sample_frame(90)
    frame["TransactionDT"] = frame.TransactionDT // 20000 * 20000
    expected = history_variants(frame)["LGB-H3"]
    for chunk_size in [1, 7, 19, 90]:
        state = StreamingHistory()
        outputs = []
        for start in range(0, len(frame), chunk_size):
            outputs.append(state.push(frame.iloc[start:start + chunk_size]))
        outputs.append(state.push(final=True))
        result = pd.concat(outputs, ignore_index=True)
        assert result.TransactionID.tolist() == frame.TransactionID.tolist()
        pd.testing.assert_frame_equal(result[expected.columns], expected, check_dtype=False)
    with pytest.raises(ValueError, match="finalized"):
        state.push(frame.iloc[:1])


def test_clipping_uses_train_only_and_excludes_codes():
    train = pd.DataFrame({"TransactionAmt": [0., 1., 2., 3.], "V1": [1., 1., 1., 1.], "card2": [1., 2., 3., 4.]})
    clip = TrainQuantileClipper(True, 0.25, 0.75).fit(train)
    future = train.iloc[[0]].copy()
    future["TransactionAmt"] = 9999
    future["card2"] = 9999
    got = clip.transform(future)
    assert got.TransactionAmt.iloc[0] == 2.25
    assert got.card2.iloc[0] == 9999
    assert clip.bounds_["TransactionAmt"] == (0.75, 2.25)


def test_daily_capacities_match_global_and_do_not_use_labels(plan):
    frame = sample_frame(70)
    scores = np.linspace(0, 1, len(frame))
    metrics, daily, weekly, queue = evaluate_period(frame, scores, plan)
    changed = frame.copy()
    changed["isFraud"] = 1 - changed.isFraud
    _, _, _, alternate = evaluate_period(changed, scores, plan)
    pd.testing.assert_frame_equal(queue, alternate)
    for scenario, group in daily.groupby("scenario"):
        summary = metrics["daily"][scenario]
        assert summary["reviews"] == int(group.reviews.sum()) == summary["matched_global"]["reviews"]
        assert summary["hits"] == int(group.hits.sum())
        assert len(queue.loc[queue.scenario.eq(scenario)]) == summary["reviews"]
        assert int(weekly.loc[weekly.scenario.eq(scenario), "reviews"].sum()) == summary["reviews"]
    # Deterministic ties prioritize lower time, then lower ID.
    assert metrics_at_k([0, 1, 1], [1., 1., 1.], [2, 1, 1], [1, 3, 2], 1)["hits"] == 1


def test_weighted_bootstrap_exactly_matches_explicit_duplicate_days():
    frame = sample_frame(80)
    scores = np.round(np.linspace(-2, 3, len(frame)), 0)  # Many probability-like ties.
    ranking = WeightedRanking(frame, scores)
    weights = np.asarray([2, 0, 1, 3])
    got = ranking.compute(weights, 0.17)
    positions = np.concatenate([
        np.tile(np.flatnonzero((frame.TransactionDT.to_numpy() // 86400) == day), int(weight))
        for day, weight in zip(ranking.days, weights)
    ])
    duplicated = frame.iloc[positions]
    s = scores[positions]
    k = int(np.ceil(0.17 * len(duplicated)))
    expected = metrics_at_k(duplicated.isFraud, s, duplicated.TransactionDT, duplicated.TransactionID, k)
    np.testing.assert_allclose(got, [expected["hits"], expected["precision"], expected["recall"], expected["average_precision"]])
    close = ranking.compute(np.ones(4, dtype=int), .005)
    assert np.isclose(close[3], average_precision_score(frame.isFraud, scores))


def test_bootstrap_capacity_reuse_matches_independent_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from analyze_results import bootstrap_metrics
    frame = sample_frame(80)
    scores = np.round(np.linspace(-2, 3, len(frame)), 0)
    ranking = WeightedRanking(frame, scores)
    weights = np.asarray([[2, 0, 1, 3], [0, 4, 0, 0], [1, 1, 1, 1]])
    capacities = {"fixed": np.asarray([16, 16, 16]), "zero": np.asarray([0, 0, 0])}
    primary, matched = bootstrap_metrics(ranking, weights, .17, capacities)
    for index, weight in enumerate(weights):
        np.testing.assert_allclose(primary[index], ranking.compute(weight, .17), equal_nan=True)
        for scenario, values in capacities.items():
            expected = ranking.compute(weight, .17, capacity=int(values[index]))
            np.testing.assert_allclose(matched[scenario][index, :3], expected[:3], equal_nan=True)


@pytest.mark.parametrize("model_id", ["LR2", "LGB-H0"])
def test_saved_model_prediction_is_batch_independent(plan, tmp_path, model_id):
    frame = sample_frame(300)
    X = frame.drop(columns=["TransactionID", "TransactionDT", "isFraud"])
    spec = next(row for row in plan["models"] if row["id"] == model_id)
    model = FraudModel(spec, plan).fit(X, frame.isFraud)
    artifact = tmp_path / "model.joblib"
    model.save(artifact)
    loaded = FraudModel.load(artifact)
    rows = X.iloc[:3].copy()
    rows["card1"] = [123.0, "2.0", None]
    rows["addr1"] = ["10.0", 20, np.nan]
    batch = loaded.predict(rows)["score"]
    individual = np.asarray([loaded.predict(rows.iloc[[i]])["score"][0] for i in range(3)])
    np.testing.assert_allclose(batch, individual, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch, model.predict(rows)["score"], rtol=1e-12, atol=1e-12)
