"""
Turn the benchmark results (``results/*.csv``) into the tables and figures of
the thesis (``memoire/tables`` and ``memoire/figures``).

Usage::

    python -m functional_supervised_classification.report
"""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from functional_supervised_classification.coeffient_compute import coeff_matrix, to_periodogram
from functional_supervised_classification.representation import (
    energy,
    normalize_rows,
    significance_mask,
    usage_frequency,
)
from functional_supervised_classification.simulation import contaminate, simulate_curves

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "memoire" / "tables"
FIG = ROOT / "memoire" / "figures"

WAVELETS = ["db1", "db2", "db4", "db6", "db8", "db10"]
FAMILIES = ["W-NN", "W-QDA", "W-CART"]
RULE_FR = {"energy": "Énergie", "usage": "Usage"}
COLORS = {"energy": "#C0602B", "usage": "#1F4E8C"}

# Tables 1 and 3 of Berlinet, Biau & Rouvière (estimated error rates)
PAPER = {
    "phoneme": {"W-NN": [0.111, 0.110, 0.112, 0.114, 0.113, 0.112],
                "W-QDA": [0.097, 0.102, 0.108, 0.108, 0.113, 0.115],
                "W-CART": [0.112, 0.130, 0.162, 0.159, 0.163, 0.185]},
    "simulation": {"W-NN": [0.146, 0.143, 0.146, 0.156, 0.155, 0.159],
                   "W-QDA": [0.078, 0.082, 0.085, 0.082, 0.085, 0.084],
                   "W-CART": [0.185, 0.174, 0.170, 0.177, 0.179, 0.161]},
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _fr(x: float, nd: int = 3) -> str:
    """French decimal comma for the LaTeX tables."""
    return f"{x:.{nd}f}".replace(".", "{,}")


def _top_set(s: str, k: int) -> set:
    return set(s.split()[:k])


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _mean_pairwise_jaccard(sets: list) -> float:
    return float(np.mean([_jaccard(a, b) for a, b in combinations(sets, 2)]))


def _paired(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Energy and usage test errors side by side for the same partitions."""
    wide = df.pivot_table(index=keys + ["rep"], columns="rule", values="test_err").reset_index()
    return wide.dropna(subset=["energy", "usage"])


def _pvalue(e: np.ndarray, u: np.ndarray) -> float:
    if np.allclose(e, u):
        return 1.0
    return float(wilcoxon(e, u, zero_method="zsplit").pvalue)


def _pstr(p: float) -> str:
    return r"$<10^{-3}$" if p < 1e-3 else _fr(p, 3)


# ── Tables 1 & 3 of the paper: six bases, three families ────────────────────

def table_bases(name: str) -> None:
    df = pd.read_csv(RES / f"{name}.csv")
    df = df[df.n_outliers == 0]
    R = df.rep.nunique()
    lines = [r"\begin{tabular}{@{}ll" + "c" * len(WAVELETS) + r"@{}}", r"\toprule",
             r"Famille & Règle & " + " & ".join(f"\\texttt{{{w}}}" for w in WAVELETS) + r" \\",
             r"\midrule"]
    for fam in FAMILIES:
        sub = df[df.family == fam]
        rows = [("Article", PAPER[name][fam], None)]
        for rule in ("energy", "usage"):
            g = sub[sub.rule == rule].groupby("wavelet").test_err
            rows.append((RULE_FR[rule], [g.mean()[w] for w in WAVELETS],
                         [g.std()[w] / np.sqrt(R) for w in WAVELETS]))
        best = [min(rows[1][1][i], rows[2][1][i]) for i in range(len(WAVELETS))]
        for k, (lab, vals, ses) in enumerate(rows):
            cells = []
            for i, v in enumerate(vals):
                s = _fr(v)
                if ses is not None and np.isclose(v, best[i]):
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            head = r"\multirow{3}{*}{" + fam + "}" if k == 0 else ""
            lines.append(f"{head} & {lab} & " + " & ".join(cells) + r" \\")
        if fam != FAMILIES[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{name}_bases.tex").write_text("\n".join(lines) + "\n")


def table_summary(name: str) -> dict:
    """Energy vs usage, db4, per family: error, se, paired test, selected d, stability."""
    df = pd.read_csv(RES / f"{name}.csv")
    df = df[(df.n_outliers == 0) & (df.wavelet == "db4")]
    out = {}
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Famille & Règle & Erreur test & $\hat d$ médian & Stabilité top-10 & Écart & $p$-valeur \\",
             r"\midrule"]
    for fam in FAMILIES + ["ALL"]:
        sub = df[df.family == fam]
        paired = _paired(sub, [])
        diff = paired.usage - paired.energy
        p = _pvalue(paired.energy.to_numpy(), paired.usage.to_numpy())
        for k, rule in enumerate(("energy", "usage")):
            r = sub[sub.rule == rule]
            R = len(r)
            stab = _mean_pairwise_jaccard([_top_set(s, 10) for s in r.top_c1])
            err = f"{_fr(r.test_err.mean())} ({_fr(r.test_err.std() / np.sqrt(R))})"
            fam_lab = "Toutes" if fam == "ALL" else fam
            head = r"\multirow{2}{*}{" + fam_lab + "}" if k == 0 else ""
            tail = (r"\multirow{2}{*}{$" + ("+" if diff.mean() >= 0 else "") + _fr(diff.mean()) + "$} & "
                    + r"\multirow{2}{*}{" + _pstr(p) + "}") if k == 0 else " & "
            lines.append(f"{head} & {RULE_FR[rule]} & {err} & {int(r.d.median())} & {_fr(stab, 2)} & {tail}" + r" \\")
            out[(fam, rule)] = dict(err=r.test_err.mean(), se=r.test_err.std() / np.sqrt(R),
                                    d=r.d.median(), stab=stab)
        out[(fam, "p")] = p
        out[(fam, "diff")] = diff.mean()
        if fam != "ALL":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{name}_resume.tex").write_text("\n".join(lines) + "\n")
    return out


def table_ecg() -> dict:
    df = pd.read_csv(RES / "ecg200.csv")
    out = {}
    lines = [r"\begin{tabular}{@{}lllccc@{}}", r"\toprule",
             r"Domaine & Famille & Règle & Erreur test & $\hat d$ médian & $p$-valeur \\", r"\midrule"]
    for dom, lab in (("ecg200-raw", "Temporel"), ("ecg200-spectrum", "Périodogramme")):
        for fam in FAMILIES + ["ALL"]:
            sub = df[(df.setting == dom) & (df.family == fam)]
            paired = _paired(sub, [])
            p = _pvalue(paired.energy.to_numpy(), paired.usage.to_numpy())
            for k, rule in enumerate(("energy", "usage")):
                r = sub[sub.rule == rule]
                R = len(r)
                dom_lab = lab if (fam == FAMILIES[0] and k == 0) else ""
                fam_lab = ("Toutes" if fam == "ALL" else fam) if k == 0 else ""
                pcell = _pstr(p) if k == 0 else ""
                lines.append(f"{dom_lab} & {fam_lab} & {RULE_FR[rule]} & "
                             f"{_fr(r.test_err.mean())} ({_fr(r.test_err.std() / np.sqrt(R))}) & "
                             f"{int(r.d.median())} & {pcell}" + r" \\")
                out[(dom, fam, rule)] = dict(err=r.test_err.mean(), d=r.d.median())
            out[(dom, fam, "p")] = p
        if dom == "ecg200-raw":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "ecg200.tex").write_text("\n".join(lines) + "\n")
    return out


def contamination() -> dict:
    df = pd.read_csv(RES / "contamination.csv")
    df = df[df.family == "ALL"]
    clean = df[df.n_outliers == 0].set_index(["rule", "rep"]).top_c1
    df = df.assign(jac=[_jaccard(_top_set(t, 10), _top_set(clean[(r, rep)], 10))
                        for t, r, rep in zip(df.top_c1, df.rule, df.rep)])
    g = df.groupby(["rule", "n_outliers"]).agg(err=("test_err", "mean"), err_sd=("test_err", "std"),
                                               d=("d", "median"), jac=("jac", "mean"),
                                               R=("rep", "nunique")).reset_index()
    ks = sorted(df.n_outliers.unique())
    lines = [r"\begin{tabular}{@{}l" + "c" * len(ks) + r"@{}}", r"\toprule",
             r"Courbes contaminées (sur $n=50$) & " + " & ".join(str(k) for k in ks) + r" \\", r"\midrule"]
    for metric, lab, nd in (("jac", "Jaccard top-10 avec l'échantillon propre", 2),
                            ("err", "Erreur test", 3), ("d", r"$\hat d$ médian", 0)):
        for rule in ("energy", "usage"):
            vals = [g[(g.rule == rule) & (g.n_outliers == k)][metric].iloc[0] for k in ks]
            cells = [str(int(v)) if nd == 0 else _fr(v, nd) for v in vals]
            lines.append(f"{lab} -- {RULE_FR[rule].lower()} & " + " & ".join(cells) + r" \\")
        if metric != "d":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "contamination.tex").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for rule in ("energy", "usage"):
        s = g[g.rule == rule]
        axes[0].plot(s.n_outliers, s.jac, "o-", color=COLORS[rule], label=RULE_FR[rule])
        axes[1].errorbar(s.n_outliers, s.err, yerr=s.err_sd / np.sqrt(s.R), fmt="o-",
                         color=COLORS[rule], capsize=2, label=RULE_FR[rule])
    axes[0].set(xlabel="courbes contaminées", ylabel="Jaccard top-10", ylim=(0, 1.05), xticks=ks)
    axes[1].set(xlabel="courbes contaminées", ylabel="erreur test", xticks=ks)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "contamination.pdf")
    plt.close(fig)
    return {(r, k): g[(g.rule == r) & (g.n_outliers == k)].iloc[0].to_dict()
            for r in ("energy", "usage") for k in ks}


def sample_size() -> dict:
    df = pd.read_csv(RES / "sample_size.csv")
    df = df[df.family == "ALL"]
    g = df.groupby(["rule", "n"]).agg(err=("test_err", "mean"), sd=("test_err", "std"),
                                      d=("d", "median"), R=("rep", "nunique")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for rule in ("energy", "usage"):
        s = g[g.rule == rule]
        axes[0].errorbar(s.n, s.err, yerr=s.sd / np.sqrt(s.R), fmt="o-", color=COLORS[rule],
                         capsize=2, label=RULE_FR[rule])
        axes[1].plot(s.n, s.d, "o-", color=COLORS[rule], label=RULE_FR[rule])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sorted(g.n.unique()))
        ax.set_xticklabels([str(v) for v in sorted(g.n.unique())])
        ax.set_xlabel("$n = m$")
    axes[0].set_ylabel("erreur test")
    axes[1].set_ylabel(r"$\hat d$ médian")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "taille_echantillon.pdf")
    plt.close(fig)
    return {(r, n): g[(g.rule == r) & (g.n == n)].iloc[0].to_dict()
            for r in ("energy", "usage") for n in sorted(g.n.unique())}


def dimension_boxplots() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, name, title in ((axes[0], "phoneme", "phoneme"), (axes[1], "simulation", "simulation")):
        df = pd.read_csv(RES / f"{name}.csv")
        df = df[(df.wavelet == "db4") & (df.n_outliers == 0)]
        data, labels, cols = [], [], []
        for fam in FAMILIES:
            for rule in ("energy", "usage"):
                data.append(df[(df.family == fam) & (df.rule == rule)].d.to_numpy())
                labels.append(f"{fam}\n{RULE_FR[rule][:3]}.")
                cols.append(COLORS[rule])
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black"),
                        flierprops=dict(markersize=2))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(r"$\hat d$")
    fig.tight_layout()
    fig.savefig(FIG / "dimensions.pdf")
    plt.close(fig)


# ── illustrative figures (no benchmark needed) ───────────────────────────────

def figure_data() -> None:
    from functional_supervised_classification.benchmark import _load_ecg200_all, _load_phoneme_all
    Xp, yp = _load_phoneme_all()
    Xe, ye = _load_ecg200_all()
    rng = np.random.default_rng(3)
    Xs, ys = simulate_curves(6, rng)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
    ax = axes[0, 0]
    for k, cl in enumerate(np.unique(yp)):
        ax.plot(np.linspace(0, 1, 256), Xp[np.where(yp == cl)[0][0]], lw=0.7, label=cl)
    ax.set_title("phoneme : log-périodogrammes")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax = axes[0, 1]
    for cl, c in zip(np.unique(ye), ("#C0602B", "#1F4E8C")):
        for i in np.where(ye == cl)[0][:4]:
            ax.plot(Xe[i], color=c, lw=0.6, alpha=0.8)
    ax.set_title("ECG200 : signaux (classes $-1$ / $1$)")
    ax = axes[0, 2]
    P = to_periodogram(Xe)
    for cl, c in zip(np.unique(ye), ("#C0602B", "#1F4E8C")):
        for i in np.where(ye == cl)[0][:4]:
            ax.plot(np.arange(1, 49) / 96, P[i], color=c, lw=0.6, alpha=0.8)
    ax.set_title("ECG200 : log-périodogrammes")
    ax.set_xlabel("fréquence (cycles / pas)")
    t = np.arange(1024) / 1024
    for j in range(3):
        ax = axes[1, j]
        ax.plot(t, Xs[j], lw=0.4, color="#1F4E8C" if ys[j] else "#C0602B")
        ax.set_title(f"simulation, $Y={ys[j]}$")
    fig.tight_layout()
    fig.savefig(FIG / "donnees.pdf")
    plt.close(fig)


def figure_rankings() -> None:
    """Usage frequency vs energy on phoneme, and their reaction to one spike."""
    from functional_supervised_classification.benchmark import _load_phoneme_all
    Xp, _ = _load_phoneme_all()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xp), 250, replace=False)
    C = coeff_matrix(Xp[idx], wavelet="db4")
    Xc, _ = contaminate(Xp[idx], 1, rng)
    Cc = coeff_matrix(Xc, wavelet="db4")
    p = C.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for M, style, lab in ((C, "-", "propre"), (Cc, "--", "1 courbe contaminée")):
        E = energy(M)
        E = np.sort(E / E.sum())[::-1]
        U = np.sort(usage_frequency(significance_mask(normalize_rows(M), "global", 1.0)))[::-1]
        axes[0].plot(np.arange(1, 41), E[:40], style, color=COLORS["energy"], label=lab)
        axes[1].plot(np.arange(1, 41), U[:40], style, color=COLORS["usage"], label=lab)
    axes[0].set(title="énergie (part du total), triée", xlabel="rang", yscale="log")
    axes[1].set(title=r"fréquence d'utilisation $\hat\varphi_j(1)$, triée", xlabel="rang", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "classements.pdf")
    plt.close(fig)

    top = lambda M, rule: set((np.argsort(-energy(M), kind="stable") if rule == "energy" else
                               np.argsort(-usage_frequency(significance_mask(normalize_rows(M), "global", 1.0)),
                                          kind="stable"))[:10])
    j_e = _jaccard(top(C, "energy"), top(Cc, "energy"))
    j_u = _jaccard(top(C, "usage"), top(Cc, "usage"))
    (TAB / "classements_valeurs.tex").write_text(
        f"\\newcommand{{\\JacEnergieUne}}{{{_fr(j_e, 2)}}}\n\\newcommand{{\\JacUsageUne}}{{{_fr(j_u, 2)}}}\n"
        f"\\newcommand{{\\NbCoeffsPhoneme}}{{{p}}}\n")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    figure_data()
    figure_rankings()
    vals = {}
    for name in ("phoneme", "simulation"):
        if (RES / f"{name}.csv").exists():
            table_bases(name)
            vals[name] = table_summary(name)
    if (RES / "ecg200.csv").exists():
        vals["ecg"] = table_ecg()
    if (RES / "contamination.csv").exists():
        vals["contamination"] = contamination()
    if (RES / "sample_size.csv").exists():
        vals["sample_size"] = sample_size()
    if (RES / "phoneme.csv").exists() and (RES / "simulation.csv").exists():
        dimension_boxplots()
    for k, v in vals.items():
        print(f"== {k}")
        for kk, vv in v.items():
            print("  ", kk, vv)


if __name__ == "__main__":
    main()
