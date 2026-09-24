"""
Tests for the revised "common representation" rule
(:mod:`functional_supervised_classification.representation`) and its wiring into
the classification pipeline.

Run directly (no pytest needed)::

    python tests/test_representation.py

or with pytest if available::

    pytest tests/test_representation.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from functional_supervised_classification.coeffient_compute import (  # noqa: E402
    coeff_matrix,
    dyadic_level,
)
from functional_supervised_classification.representation import (  # noqa: E402
    discriminative_usage,
    energy,
    feature_ranking,
    normalize_rows,
    selection_stability,
    significance_mask,
    usage_frequency,
    usage_frequency_by_class,
)


# ── significance_mask ────────────────────────────────────────────────────────

def test_top_m_flags_exactly_m_per_row():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(30, 12))
    for m in (1, 3, 7, 12):
        mask = significance_mask(C, rule="top_m", param=m)
        assert mask.shape == C.shape and mask.dtype == bool
        assert np.all(mask.sum(axis=1) == m)
        # the flagged entries are the m largest magnitudes of the row
        for i in range(C.shape[0]):
            thr = np.sort(np.abs(C[i]))[-m]
            assert np.all(np.abs(C[i])[mask[i]] >= thr - 1e-12)


def test_top_m_clamps_and_validates():
    C = np.random.default_rng(1).normal(size=(5, 4))
    assert np.all(significance_mask(C, "top_m", 99).sum(axis=1) == 4)   # clamp to p
    for bad in (0, -2):
        try:
            significance_mask(C, "top_m", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"top_m={bad} should raise")


def test_quantile_flags_expected_fraction():
    rng = np.random.default_rng(2)
    C = rng.normal(size=(200, 50))
    for q in (0.80, 0.90, 0.95):
        frac = significance_mask(C, rule="quantile", param=q).mean()
        assert abs(frac - (1.0 - q)) < 0.02
    for bad in (0.0, 1.0, 1.5, -0.1):
        try:
            significance_mask(C, "quantile", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"quantile={bad} should raise")


def test_universal_catches_spikes_not_noise():
    rng = np.random.default_rng(3)
    C = rng.normal(scale=1.0, size=(40, 8))
    C[0:5, 2] = 20.0                       # unambiguous spikes
    mask = significance_mask(C, rule="universal", param=1.0)
    assert mask.shape == C.shape and mask.dtype == bool
    assert np.all(mask[0:5, 2])            # every spike flagged
    assert mask.mean() < 0.25             # the bulk (noise) is not flagged


def test_unknown_rule_raises():
    C = np.zeros((3, 3))
    try:
        significance_mask(C, rule="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown rule should raise")


# ── usage frequency & energy ────────────────────────────────────────────────

def test_usage_frequency_matches_mask_mean():
    rng = np.random.default_rng(4)
    C = rng.normal(size=(25, 9))
    mask = significance_mask(C, "top_m", 3)
    phi = usage_frequency(mask)
    assert phi.shape == (9,)
    assert np.all((phi >= 0) & (phi <= 1))
    assert np.allclose(phi, mask.mean(axis=0))
    assert abs(phi.sum() - 3.0) < 1e-9    # 3 flags per row, averaged over rows


def test_energy_matches_sum_of_squares():
    rng = np.random.default_rng(5)
    C = rng.normal(size=(15, 7))
    assert np.allclose(energy(C), np.sum(C ** 2, axis=0))


# ── feature_ranking : the core claim of the report ─────────────────────────

def _shared_vs_outlier_matrix():
    """Feature 2: moderate, present in *every* signal. Feature 4: one huge outlier."""
    rng = np.random.default_rng(6)
    C = rng.normal(scale=0.01, size=(40, 6))
    C[:, 2] = rng.normal(scale=0.3, size=40)
    C[5, 4] = 50.0
    return C


def test_energy_rule_is_fooled_by_the_outlier():
    C = _shared_vs_outlier_matrix()
    rank = feature_ranking(C, selection_rule="energy")
    assert sorted(rank.tolist()) == list(range(6))          # a permutation
    assert np.array_equal(rank, np.argsort(-energy(C), kind="stable"))
    assert rank[0] == 4                                     # single spike wins on energy


def test_usage_rule_prefers_the_shared_feature():
    C = _shared_vs_outlier_matrix()
    rank = feature_ranking(C, selection_rule="usage", significance_rule="top_m",
                           significance_param=2)
    assert rank[0] == 2                                     # the genuinely common feature
    assert rank.tolist().index(4) >= 4                      # the outlier is demoted


def test_discriminant_rule_finds_the_class_specific_feature():
    rng = np.random.default_rng(7)
    C = rng.normal(scale=0.01, size=(40, 6))
    y = np.array([0] * 20 + [1] * 20)
    C[20:, 3] = rng.normal(scale=0.5, size=20)              # feature 3 used by class 1 only

    classes, freqs = usage_frequency_by_class(significance_mask(C, "top_m", 2), y)
    assert classes.tolist() == [0, 1]
    assert freqs.shape == (2, 6)

    score = discriminative_usage(significance_mask(C, "top_m", 2), y)
    assert np.argmax(score) == 3
    assert feature_ranking(C, y=y, selection_rule="usage_discriminant",
                           significance_rule="top_m", significance_param=2)[0] == 3


def test_discriminant_rule_requires_labels():
    try:
        feature_ranking(np.zeros((4, 4)), selection_rule="usage_discriminant")
    except ValueError:
        pass
    else:
        raise AssertionError("usage_discriminant without y should raise")


def test_feature_ranking_rejects_unknown_rule():
    try:
        feature_ranking(np.zeros((4, 4)), selection_rule="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown selection_rule should raise")


# ── selection_stability ────────────────────────────────────────────────────

def test_selection_stability_bounds():
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1, 0])
    assert selection_stability([a, a, a], d=3) == 1.0            # identical top-3
    assert selection_stability([a, b], d=3) == 0.0               # {0,1,2} vs {5,4,3}
    assert np.isnan(selection_stability([a], d=3))               # need >= 2
    mid = selection_stability([a, b], d=4)                       # {0,1,2,3} vs {5,4,3,2}
    assert abs(mid - (2 / 6)) < 1e-9


# ── the ranking composes correctly inside a cross-validation loop ──────────

def test_ranking_inside_cv_recovers_a_separable_signal():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(8)
    n, p = 80, 10
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    C = rng.normal(size=(n, p))
    C[y == 1, 0] += 3.0                                         # class gap in feature 0

    accs = []
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(C, y):
        order = feature_ranking(C[tr], y[tr], selection_rule="usage_discriminant",
                                significance_rule="top_m", significance_param=3)
        assert order[0] == 0                                    # picks the informative feature
        clf = KNeighborsClassifier().fit(C[tr][:, order][:, :2], y[tr])
        accs.append(accuracy_score(y[va], clf.predict(C[va][:, order][:, :2])))
    assert np.mean(accs) > 0.85


# ── end-to-end pipeline smoke test (subprocess, tiny config) ───────────────

def _run_pipeline(rule: str) -> subprocess.CompletedProcess:
    cfg = (
        'dataset: "ecg200"\n'
        'input_domain: "raw"\n'
        f'selection_rule: "{rule}"\n'
        "significance:\n"
        '  rule: "top_m"\n'
        "  param: null\n"
        "cv_folds: 2\n"
        "d_max: 6\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(cfg)
        cfg_path = fh.name
    try:
        env = {**os.environ, "FSC_CONFIG": cfg_path, "MPLBACKEND": "Agg"}
        return subprocess.run(
            [sys.executable, "-m", "functional_supervised_classification.model"],
            cwd=str(_ROOT), env=env, capture_output=True, text=True, timeout=300,
        )
    finally:
        os.unlink(cfg_path)


def test_pipeline_runs_with_usage_rule():
    proc = _run_pipeline("usage")
    if proc.returncode != 0 and (
        "ConnectionError" in proc.stderr or "Failed to download" in proc.stderr
    ):
        print("  SKIP test_pipeline_runs_with_usage_rule (dataset download unavailable)")
        return
    assert proc.returncode == 0, f"pipeline failed:\nSTDOUT{proc.stdout}\nSTDERR{proc.stderr}"
    assert "acc=" in proc.stdout and "jaccard=" in proc.stdout
    assert "representation rule: usage" in proc.stdout


def test_pipeline_still_runs_with_energy_rule():
    proc = _run_pipeline("energy")
    if proc.returncode != 0 and (
        "ConnectionError" in proc.stderr or "Failed to download" in proc.stderr
    ):
        print("  SKIP test_pipeline_still_runs_with_energy_rule (dataset download unavailable)")
        return
    assert proc.returncode == 0, f"pipeline failed:\nSTDOUT{proc.stdout}\nSTDERR{proc.stderr}"
    assert "acc=" in proc.stdout


# ── orthogonal DWT, normalisation and the global-threshold rule ─────────────

def test_periodized_dwt_is_orthogonal():
    rng = np.random.default_rng(10)
    for T, L in ((96, 5), (256, 8), (1024, 10)):
        assert dyadic_level(T) == L
        S = rng.normal(size=(4, T))
        for wavelet in ("db1", "db4", "db10"):
            C = coeff_matrix(S, wavelet=wavelet)
            assert C.shape == (4, T)                            # T coefficients, no padding
            assert np.allclose(np.sum(C ** 2, axis=1), np.sum(S ** 2, axis=1))  # Parseval


def test_normalize_rows_gives_unit_energy():
    C = np.random.default_rng(11).normal(size=(6, 9))
    C[2] = 0.0
    N = normalize_rows(C)
    norms = np.linalg.norm(N, axis=1)
    assert np.allclose(np.delete(norms, 2), 1.0)
    assert np.all(N[2] == 0.0)                                  # null row left as is


def test_global_threshold_matches_definition():
    rng = np.random.default_rng(12)
    C = normalize_rows(rng.normal(size=(20, 16)))
    for c in (0.5, 1.0, 2.0):
        assert np.array_equal(significance_mask(C, "global", c), np.abs(C) > c / 4.0)
    for bad in (0.0, -1.0):
        try:
            significance_mask(C, "global", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"global={bad} should raise")


def test_global_usage_is_invariant_to_curve_scaling():
    rng = np.random.default_rng(13)
    C = rng.normal(size=(30, 16)) * rng.uniform(0.5, 1.5, size=16)
    scaled = C * rng.uniform(0.01, 100.0, size=(30, 1))         # one gain per curve
    r1 = feature_ranking(C, selection_rule="usage", significance_rule="global", significance_param=1.0)
    r2 = feature_ranking(scaled, selection_rule="usage", significance_rule="global", significance_param=1.0)
    assert np.array_equal(r1, r2)


def test_global_usage_prefers_the_shared_feature():
    C = _shared_vs_outlier_matrix()
    rank = feature_ranking(C, selection_rule="usage", significance_rule="global",
                           significance_param=1.0)
    assert rank[0] == 2
    assert rank.tolist().index(4) >= 2                      # the lone spike is not on top


def test_changing_one_curve_moves_usage_by_at_most_one_over_n():
    rng = np.random.default_rng(14)
    C = rng.normal(size=(50, 32))
    C2 = C.copy()
    C2[7] = rng.normal(scale=1e6, size=32)                   # arbitrary replacement
    phi1 = usage_frequency(significance_mask(normalize_rows(C), "global", 1.0))
    phi2 = usage_frequency(significance_mask(normalize_rows(C2), "global", 1.0))
    assert np.max(np.abs(phi1 - phi2)) <= 1.0 / 50 + 1e-12
    # the energy of the same features can be moved arbitrarily far
    assert np.max(np.abs(energy(C) - energy(C2))) > 1e6


# ── standalone runner ──────────────────────────────────────────────────────

def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}\n      {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
