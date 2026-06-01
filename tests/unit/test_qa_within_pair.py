"""Tests for the within-pair scorers"""

from __future__ import annotations

import pytest

from mm_pipeline.qa.within_pair import ClassifierMax, DPCostMin, Ensemble, build_scorer


def _pd():
    return pytest.importorskip("pandas")


def _fixture():
    pd = _pd()
    # Three candidates: lowest DP cost is row 0; highest raw_score is row 2.
    return pd.DataFrame(
        {
            "dp_cost": [0.1, 0.5, 0.8],
            "raw_score": [0.2, 0.3, 0.9],
        },
        index=[10, 20, 30],
    )


def test_dp_cost_min_picks_lowest_cost():
    df = _fixture()
    pick = DPCostMin().pick(df)
    assert pick.chosen_idx == 10
    assert pick.chosen_score == pytest.approx(0.1)


def test_classifier_max_picks_highest_score():
    df = _fixture()
    pick = ClassifierMax().pick(df)
    assert pick.chosen_idx == 30
    assert pick.chosen_score == pytest.approx(0.9)


def test_dp_cost_min_requires_dp_cost():
    pd = _pd()
    df = pd.DataFrame({"raw_score": [0.1, 0.9]})
    with pytest.raises(KeyError, match="dp_cost"):
        DPCostMin().pick(df)


def test_classifier_max_requires_raw_score():
    pd = _pd()
    df = pd.DataFrame({"dp_cost": [0.1, 0.9]})
    with pytest.raises(KeyError, match="raw_score"):
        ClassifierMax().pick(df)


def test_ensemble_rank_alpha_one_matches_dp():
    df = _fixture()
    pick = Ensemble(alpha=1.0, mode="rank").pick(df)
    assert pick.chosen_idx == DPCostMin().pick(df).chosen_idx


def test_ensemble_rank_alpha_zero_matches_classifier():
    df = _fixture()
    pick = Ensemble(alpha=0.0, mode="rank").pick(df)
    assert pick.chosen_idx == ClassifierMax().pick(df).chosen_idx


def test_ensemble_zscore_alpha_endpoints_match():
    df = _fixture()
    assert Ensemble(alpha=1.0, mode="zscore").pick(df).chosen_idx == DPCostMin().pick(df).chosen_idx
    assert Ensemble(alpha=0.0, mode="zscore").pick(df).chosen_idx == ClassifierMax().pick(df).chosen_idx


def test_ensemble_intermediate_alpha_resolves_consistently():
    pd = _pd()
    # DP top-1 = row 10 (cost 0.1), classifier top-1 = row 30 (score 0.9). Row
    # 20 is mid-ranked on both. With α=0.5 the combined rank sums favour the
    # endpoints; choose the DP-favoured tie break.
    df = pd.DataFrame(
        {
            "dp_cost": [0.1, 0.5, 0.8],
            "raw_score": [0.2, 0.3, 0.9],
        },
        index=[10, 20, 30],
    )
    pick = Ensemble(alpha=0.5, mode="rank").pick(df)
    # With per-input ranks (10:1,20:2,30:3) and (10:3,20:2,30:1), combined ranks
    # are (10:2.0, 20:2.0, 30:2.0). Tie broken by DP rank → 10.
    assert pick.chosen_idx == 10


def test_scorers_preserve_non_integer_index_labels():
    pd = _pd()
    df = pd.DataFrame(
        {
            "dp_cost": [0.5, 0.1],
            "raw_score": [0.9, 0.2],
        },
        index=["candidate-a", "candidate-b"],
    )
    assert DPCostMin().pick(df).chosen_idx == "candidate-b"
    assert ClassifierMax().pick(df).chosen_idx == "candidate-a"
    assert Ensemble(alpha=1.0).pick(df).chosen_idx == "candidate-b"
    assert Ensemble(alpha=0.0, mode="zscore").pick(df).chosen_idx == "candidate-a"


def test_build_scorer_resolves_names():
    assert isinstance(build_scorer("dp_cost_min"), DPCostMin)
    assert isinstance(build_scorer("classifier"), ClassifierMax)
    scorer = build_scorer("ensemble", ensemble_alpha=0.3, ensemble_mode="rank")
    assert isinstance(scorer, Ensemble)
    assert scorer.alpha == pytest.approx(0.3)
    assert scorer.mode == "rank"


def test_build_scorer_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown within_pair_scorer"):
        build_scorer("not_a_scorer")
