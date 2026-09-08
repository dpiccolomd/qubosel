import numpy as np
import pytest

from qubosel.datasets import CLINICAL_COHORT_NAMES, make_clinical_cohort
from qubosel.simulation import recovery_curve, recovery_table, standard_objectives


def test_make_clinical_cohort():
    X, y, info = make_clinical_cohort(n_samples=100, random_state=0)
    assert list(X.columns) == CLINICAL_COHORT_NAMES and X.shape == (100, 14)
    assert set(np.unique(y)) == {0, 1} and info["truth"] == [1, 5, 6]
    assert np.array_equal(X["CountTotal"], X["CountA"] + X["CountB"])
    X2, y2, _ = make_clinical_cohort(n_samples=100, random_state=0)
    assert X.equals(X2) and np.array_equal(y, y2)
    with pytest.raises(ValueError):
        make_clinical_cohort(truth=("Nope",))


def test_standard_objectives_keys():
    objs = standard_objectives(3)
    assert {
        "mucke_hist10",
        "mucke_mm_quantile4",
        "mucke_permutation",
        "mucke_expected",
        "rank",
        "anova_topk",
        "mucke_ksg",
    } == set(objs)
    assert "mucke_ksg" not in standard_objectives(3, include_ksg=False)


@pytest.mark.slow
def test_recovery_curve_small():
    objs = {
        k: v
        for k, v in standard_objectives(3, include_ksg=False).items()
        if k in ("mucke_hist10", "rank", "anova_topk")
    }
    df = recovery_curve(
        objs,
        n_list=(31, 300),
        n_replicates=6,
        random_state=0,
        n_jobs=1,
        scenarios={"c": ("Age", "CountTotal", "Ordinal")},
    )
    assert set(df.columns) == {"scenario", "n", "method", "exact_recovery", "jaccard", "se"}
    assert df.shape[0] == 6
    table = recovery_table(df, "c")
    assert table.shape == (2, 3) and ((table >= 0) & (table <= 1)).all().all()
    # the rank objective must recover the truth more often at n=300 than the biased histogram MI
    assert table.loc[300, "rank"] > table.loc[300, "mucke_hist10"]
