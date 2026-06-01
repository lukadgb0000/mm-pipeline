import math

import pytest

from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.scoring.model_registry import DEFAULT_MODEL_NAME, get_model_spec, list_models


def _pd():
    return pytest.importorskip("pandas")


def _np():
    return pytest.importorskip("numpy")


def _sample_table():
    pd = _pd()
    np = _np()
    rows = []
    datasets = ("d1", "d2", "d3")
    for ds_idx, dataset_id in enumerate(datasets):
        for pair_idx in range(4):
            pair_id = f"{dataset_id}:p{pair_idx}"
            jitter = 0.01 * (ds_idx + pair_idx)
            for is_correct, offset in ((True, 0.0), (False, 1.0)):
                row = {
                    "dataset_id": dataset_id,
                    "pair_id": pair_id,
                    "is_correct": is_correct,
                    "n_candidates_pair": 2,
                    "sample_rank": 1 if is_correct else 2,
                    "n_links": 1,
                    "n_exits": 0,
                    "n_divides": 0 if is_correct else 1,
                }
                for feat_idx, feature in enumerate(FEATURE_COLUMNS):
                    if feature.startswith("div_") and is_correct and pair_idx % 2 == 0:
                        row[feature] = np.nan
                    elif is_correct:
                        row[feature] = 1.0 + jitter + feat_idx * 0.005
                    else:
                        row[feature] = 3.0 + offset + jitter + feat_idx * 0.02
                row["max_shrink_pct"] = 2.0 + jitter if is_correct else 40.0 + jitter
                row["total_area_ratio_exit_adjusted"] = 1.0 + jitter if is_correct else 1.7 + jitter
                row["link_iou_shifted_median"] = 0.9 - jitter if is_correct else 0.2 + jitter
                rows.append(row)
    return pd.DataFrame(rows)


def test_registry_lists_phase8_models_without_importing_optional_estimators():
    names = set(list_models())

    assert DEFAULT_MODEL_NAME == "logreg_l2_balanced"
    assert {
        "logreg_l2_balanced",
        "logreg_l1_balanced",
        "random_forest_balanced",
        "hist_gbm",
        "linear_svm_balanced",
        "rbf_svm_balanced",
        "naive_bayes_parametric",
        "naive_bayes_kde",
    } <= names
    assert get_model_spec().name == "logreg_l2_balanced"
    with pytest.raises(KeyError, match="Available models"):
        get_model_spec("missing_model")


def test_logreg_scoring_contract_and_pair_probabilities():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _sample_table()

    scorer = fit_scorer(df, model_name="logreg_l2_balanced")
    scored = score_candidates(df, scorer)

    for col in (
        "raw_score",
        "raw_score_kind",
        "candidate_correctness_probability",
        "pair_probability",
        "score_rank",
        "y_score",
        "pair_prob",
        "pair_score_rank",
        "score_model",
        "score_feature_subset",
        "score_is_calibrated",
    ):
        assert col in scored.columns

    assert set(scored["raw_score_kind"]) == {"logit"}
    assert set(scored["score_model"]) == {"logreg_l2_balanced"}
    assert scored["candidate_correctness_probability"].between(0.0, 1.0).all()
    for _, group in scored.groupby("pair_id"):
        assert group["pair_probability"].sum() == pytest.approx(1.0)
        best = group.sort_values("raw_score", ascending=False).iloc[0]
        assert best["score_rank"] == pytest.approx(1.0)
    assert np.allclose(scored["pair_probability"], scored["pair_prob"])


def test_linear_svm_without_calibration_has_raw_score_but_no_probability():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _sample_table()

    scorer = fit_scorer(df, model_name="linear_svm_balanced")
    scored = score_candidates(df, scorer)

    assert set(scored["raw_score_kind"]) == {"decision"}
    assert np.isfinite(scored["raw_score"].to_numpy(dtype=float)).all()
    assert scored["candidate_correctness_probability"].isna().all()
    assert np.allclose(scored["y_score"], scored["raw_score"])


def test_lodo_scoring_and_missing_scorer_failure():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from mm_pipeline.scoring import fit_lodo_scorers, score_with_lodo_scorers

    df = _sample_table()

    scorers = fit_lodo_scorers(df, model_name="logreg_l2_balanced")
    assert set(scorers) == {"d1", "d2", "d3"}
    scored = score_with_lodo_scorers(df, scorers)
    assert len(scored) == len(df)

    scorers.pop("d3")
    with pytest.raises(KeyError, match="No fitted scorer"):
        score_with_lodo_scorers(df, scorers)


def test_scorer_save_load_roundtrip(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, load_scorer, save_scorer, score_candidates

    df = _sample_table()

    scorer = fit_scorer(df, model_name="logreg_l2_balanced")
    before = score_candidates(df, scorer)
    path = save_scorer(scorer, tmp_path / "scorer.joblib")
    loaded = load_scorer(path)
    after = score_candidates(df, loaded)

    assert loaded.model_name == scorer.model_name
    assert loaded.raw_score_kind == scorer.raw_score_kind
    assert np.allclose(before["raw_score"], after["raw_score"])
    assert np.allclose(before["candidate_correctness_probability"], after["candidate_correctness_probability"])


def test_naive_bayes_modes_accept_missing_division_features_and_expose_contributions():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _sample_table()

    for model_name in ("naive_bayes_parametric", "naive_bayes_kde"):
        scorer = fit_scorer(df, model_name=model_name)
        scored = score_candidates(df, scorer)
        contribs = scorer.feature_contributions(df)

        assert set(scored["raw_score_kind"]) == {"llr"}
        assert np.isfinite(scored["raw_score"].to_numpy(dtype=float)).all()
        assert scored["candidate_correctness_probability"].between(0.0, 1.0).all()
        assert scored.groupby("pair_id")["pair_probability"].sum().apply(lambda v: math.isclose(v, 1.0)).all()
        assert list(contribs.columns) == list(FEATURE_COLUMNS)


def _overlapping_sample_table():
    """Build a fixture where correct/incorrect distributions overlap enough that
    the Naive Bayes log-likelihood ratio stays in the linear sigmoid regime."""

    pd = _pd()
    np = _np()
    rng = np.random.default_rng(seed=7)
    rows = []
    for ds_idx, dataset_id in enumerate(("d1", "d2")):
        for pair_idx in range(8):
            pair_id = f"{dataset_id}:p{pair_idx}"
            for is_correct in (True, False):
                row = {
                    "dataset_id": dataset_id,
                    "pair_id": pair_id,
                    "is_correct": is_correct,
                    "n_candidates_pair": 2,
                    "sample_rank": 1 if is_correct else 2,
                    "n_links": 1,
                    "n_exits": 0,
                    "n_divides": 0,
                }
                # Heavily overlapping unimodal features around 1.0; only a tiny
                # mean shift separates classes, keeping LLRs near zero.
                base = 1.0 + (0.05 if is_correct else 0.15) + 0.02 * ds_idx
                for feature in FEATURE_COLUMNS:
                    row[feature] = float(base + rng.normal(0.0, 0.3))
                rows.append(row)
    return pd.DataFrame(rows)


def test_naive_bayes_prior_changes_probability_not_ranking():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _overlapping_sample_table()

    low_prior, high_prior = 0.1, 0.8
    scorer_low = fit_scorer(df, model_name="naive_bayes_parametric", nb_prior=low_prior)
    scorer_high = fit_scorer(df, model_name="naive_bayes_parametric", nb_prior=high_prior)
    low = score_candidates(df, scorer_low)
    high = score_candidates(df, scorer_high)

    assert np.allclose(low["raw_score"], high["raw_score"])
    assert np.allclose(low["score_rank"], high["score_rank"])
    assert np.allclose(low["pair_probability"], high["pair_probability"])

    # The posterior is sigmoid(raw + logit(prior)); in log-odds space the gap
    # between the two priors equals logit(high) - logit(low) regardless of
    # raw_score magnitude.
    p_low = low["candidate_correctness_probability"].to_numpy(dtype=float)
    p_high = high["candidate_correctness_probability"].to_numpy(dtype=float)
    assert ((p_low > 1e-6) & (p_low < 1.0 - 1e-6)).any()
    logit_low = np.log(p_low / (1.0 - p_low))
    logit_high = np.log(p_high / (1.0 - p_high))
    expected_delta = math.log(high_prior / (1.0 - high_prior)) - math.log(low_prior / (1.0 - low_prior))
    assert np.allclose(logit_high - logit_low, expected_delta, atol=1e-6)


def test_naive_bayes_uniform_within_pair_prior_uses_group_size():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _overlapping_sample_table()

    # Add a third candidate to one pair so group sizes differ across pairs;
    # uniform_within_pair must use 1/group_size, not a single constant.
    extra = df.iloc[[0]].copy()
    extra["is_correct"] = False
    df_mixed = _pd().concat([df, extra], ignore_index=True)

    scorer = fit_scorer(df_mixed, model_name="naive_bayes_parametric", nb_prior="uniform_within_pair")
    scored = score_candidates(df_mixed, scorer)

    raw = scored["raw_score"].to_numpy(dtype=float)
    posterior = scored["candidate_correctness_probability"].to_numpy(dtype=float)
    group_sizes = df_mixed.groupby("pair_id", sort=False)["pair_id"].transform("size").to_numpy(dtype=float)
    prior_per_row = 1.0 / group_sizes
    expected = 1.0 / (1.0 + np.exp(-(raw + np.log(prior_per_row / (1.0 - prior_per_row)))))
    assert np.allclose(posterior, expected, atol=1e-9)
    # Pairs with three candidates and pairs with two must have used different
    # priors (1/3 vs 1/2), which is the whole point of this mode.
    assert len(np.unique(prior_per_row)) >= 2


def test_calibration_happy_path_produces_probability_estimator():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    np = _np()
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _overlapping_sample_table()

    scorer = fit_scorer(
        df,
        model_name="logreg_l2_balanced",
        calibrate=True,
        calibration_method="sigmoid",
        calibration_cv=3,
    )
    scored = score_candidates(df, scorer)

    assert scorer.is_calibrated is True
    assert scorer.probability_estimator is not None
    assert scorer.calibration_method == "sigmoid"
    assert scorer.calibration_cv == 3
    assert set(scored["score_is_calibrated"]) == {True}
    probs = scored["candidate_correctness_probability"].to_numpy(dtype=float)
    assert np.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_validation_errors_are_clear():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from mm_pipeline.scoring import fit_scorer, score_candidates

    df = _sample_table()

    with pytest.raises(KeyError, match="Missing feature columns"):
        fit_scorer(df.drop(columns=[FEATURE_COLUMNS[0]]))
    with pytest.raises(KeyError, match="Missing target column"):
        fit_scorer(df.drop(columns=["is_correct"]))
    with pytest.raises(ValueError, match="both correct and incorrect"):
        fit_scorer(df.assign(is_correct=True))

    scorer = fit_scorer(df)
    with pytest.raises(ValueError, match="pair_temperature"):
        score_candidates(df, scorer, pair_temperature=0.0)

    small = df.groupby("is_correct", group_keys=False).head(2)
    with pytest.raises(ValueError, match="Calibration requires"):
        fit_scorer(small, calibrate=True, calibration_cv=3)
