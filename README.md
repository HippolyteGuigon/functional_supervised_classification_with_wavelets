# Functional Supervised Classification with Wavelets

> Experimental implementation and theoretical study of the methodology introduced in **"Functional Supervised Classification with Wavelets"** by Alain Berlinet, Gérard Biau and Laurent Rouvière.

---

## Overview

This repository is part of my **Master's thesis in Theoretical Mathematics** at **Aix-Marseille University**, conducted under the supervision of **Prof. Jean-Marc Freyermuth**.

The objective of this project is **not simply to reproduce the original paper**, but rather to provide a rigorous experimental framework for studying, implementing, extending and evaluating the proposed methodology for **functional data classification using wavelets**.

The repository combines theoretical developments from Functional Data Analysis (FDA), Statistical Learning Theory and Wavelet Analysis with practical implementations allowing reproducible numerical experiments.

---

# Paper Summary

The paper addresses the problem of **supervised classification when observations are functions instead of finite-dimensional vectors**.

More formally, the learning sample consists of independent observations

`(Xᵢ, Yᵢ),  i = 1, ..., n`

where

- `Xᵢ ∈ 𝓗` belongs to an infinite-dimensional Hilbert space (typically `L²([0,1])`);
- `Yᵢ ∈ {0,1}` denotes the class label.

Unlike classical machine learning where data naturally lie in `ℝᵈ`, functional observations are entire curves, signals or time series.

This infinite-dimensional setting immediately raises one of the main challenges of Functional Data Analysis:

> **How can we perform classification while avoiding the curse of dimensionality?**

The proposed solution consists of reducing the dimensionality **without losing the discriminative information contained in the original signals.** :contentReference[oaicite:1]{index=1}

---

# Mathematical Foundations

## Functional Representation

Each observation is assumed to belong to

\[
L^2([0,1]),
\]

and admits the wavelet expansion

\[
X(t)
=
\sum_{j=0}^{\infty}
\sum_{k=0}^{2^j-1}
\zeta_{j,k}\psi_{j,k}(t)
+
\eta\phi(t),
\]

where

- \(\phi\) is the scaling function,
- \(\psi_{j,k}\) are wavelet basis functions,
- \(\zeta_{j,k}\) are the associated wavelet coefficients.

This representation provides a localized description of the signal in both **time** and **frequency**, which is one of the major advantages of wavelets over Fourier bases. :contentReference[oaicite:2]{index=2}

---

## Dimension Reduction

Instead of working in the infinite-dimensional Hilbert space, the signal is approximated by truncating the expansion at a fixed resolution level \(J\):

\[
X(t)
\approx
\sum_{j=1}^{2^J}
X_j\psi_j(t).
\]

The resulting representation is finite-dimensional but still contains many coefficients.

A key contribution of the paper consists in **ranking the wavelet coefficients according to their empirical energy over the training sample**

\[
\sum_{i=1}^{n} X_{ij}^2,
\]

and selecting only the most informative ones.

This adaptive ordering produces a data-driven reduction of the feature space while preserving the coefficients carrying the highest amount of information. :contentReference[oaicite:3]{index=3}

---

## Automatic Model Selection

For every possible retained dimension

\[
d=1,\ldots,2^J,
\]

a classifier is trained on the first \(d\) reordered coefficients.

The optimal pair

\[
(\hat d,\hat g)
\]

is selected by minimizing the empirical classification error over an independent validation set

\[
(\hat d,\hat g)
=
\operatorname*{arg\,min}
\frac1m
\sum_{i=n+1}^{n+m}
\mathbf1_{\{
g(X_i)\neq Y_i
\}}.
\]

This automatic selection simultaneously determines

- the optimal wavelet dimension,
- the optimal classifier.

The paper illustrates this framework using several classifiers including

- k-Nearest Neighbours
- Quadratic Discriminant Analysis (QDA)
- Classification and Regression Trees (CART). :contentReference[oaicite:4]{index=4}

---

## Statistical Guarantee

One of the strongest theoretical results established in the paper is a consistency guarantee.

The selected classifier satisfies an upper bound of the form

\[
\mathbb E[L(\hat g)]-L^*
\le
\text{Approximation Error}
+
\text{Estimation Error}
+
\text{Complexity Term},
\]

where

- \(L^*\) denotes the Bayes risk,
- the approximation error comes from truncating the wavelet expansion,
- the estimation error depends on the chosen classifier,
- the complexity term is controlled through Vapnik–Chervonenkis theory and shatter coefficients.

Under suitable assumptions, the classifier is proved to be **consistent**, meaning that its expected risk converges to the Bayes optimal risk. :contentReference[oaicite:5]{index=5}

---

# Repository Objectives

This repository aims to provide a modern experimental implementation of the methodology introduced in the paper.

More specifically, it is intended to

- implement the complete wavelet classification pipeline from scratch;
- reproduce the numerical experiments reported in the paper;
- compare different wavelet families (Daubechies, Haar, Symlets, Coiflets, ...);
- evaluate multiple classifiers on the wavelet representations;
- investigate alternative coefficient ranking strategies;
- study the impact of thresholding and dimension selection;
- compare the original approach with more recent Functional Data Analysis techniques;
- explore possible extensions using contemporary machine learning algorithms.

Rather than being a reproduction only, this repository serves as an **experimental research platform** for studying wavelet-based functional classification.

---

# Research Context

This work is conducted as part of a **Master's Thesis in Theoretical Mathematics** at

**Aix-Marseille University**

under the supervision of

**Prof. Jean-Marc Freyermuth**.

The purpose of this repository is to bridge the gap between the theoretical results established in the original article and their practical implementation, while exploring possible methodological improvements and new research directions.

---

# Repository Status

This repository is currently under active development.

Planned milestones include

- Wavelet decomposition module
- Functional dataset generation
- Automatic coefficient selection
- Classifier benchmark
- Experimental reproduction of the paper
- Statistical evaluation
- Visualization utilities
- Extended experiments beyond the original publication

---

# Reference

Berlinet, A., Biau, G., & Rouvière, L.

**Functional Supervised Classification with Wavelets**

*Annales de l'ISUP*.
