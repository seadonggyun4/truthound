# Statistical Methods

Truthound implements well-established statistical methods for drift detection, anomaly detection, and distribution analysis. This document provides detailed explanations of each method.

---

## Table of Contents

1. [Outlier Detection (IQR Method)](#1-outlier-detection-iqr-method)
2. [Kolmogorov-Smirnov Test](#2-kolmogorov-smirnov-test)
3. [Population Stability Index (PSI)](#3-population-stability-index-psi)
4. [Chi-Square Test](#4-chi-square-test)
5. [Jensen-Shannon Divergence](#5-jensen-shannon-divergence)
6. [Kullback-Leibler Divergence](#6-kullback-leibler-divergence)
7. [Mahalanobis Distance](#7-mahalanobis-distance)
8. [Isolation Forest](#8-isolation-forest)
9. [Wasserstein Distance](#9-wasserstein-distance)
10. [Local Outlier Factor (LOF)](#10-local-outlier-factor-lof)

---

## 1. Outlier Detection (IQR Method)

The Interquartile Range (IQR) method identifies statistical outliers based on the spread of the middle 50% of the data.

### Formula

```
IQR = Q3 - Q1
Lower Bound = Q1 - k × IQR
Upper Bound = Q3 + k × IQR
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| k | 1.5 | Standard outliers |
| k | 3.0 | Extreme outliers |

### Interpretation

- Values below the lower bound or above the upper bound are considered outliers
- The method is distribution-free and robust to extreme values
- Works best for symmetric or approximately symmetric distributions

### Implementation in Truthound

```python
from truthound.validators.anomaly import IQRAnomalyValidator

validator = IQRAnomalyValidator(
    column="value",
    k=1.5  # 1.5 for standard, 3.0 for extreme
)
```

---

## 2. Kolmogorov-Smirnov Test

Measures the maximum difference between two empirical cumulative distribution functions (ECDFs).

### Formula

```
D = max|F₁(x) - F₂(x)|
```

Where:
- F₁(x) is the ECDF of the baseline distribution
- F₂(x) is the ECDF of the current distribution

### Interpretation

| p-value | Interpretation |
|---------|----------------|
| p < 0.05 | Significant difference (reject null hypothesis) |
| p ≥ 0.05 | No significant difference |

### Advantages

- Non-parametric (no distribution assumptions)
- Sensitive to differences in location, scale, and shape
- P-value approximation uses the asymptotic Kolmogorov distribution

### Implementation in Truthound

```python
from truthound.validators.drift import KSTestValidator

validator = KSTestValidator(
    column="feature",
    baseline_data=baseline_df,
    alpha=0.05
)
```

---

## 3. Population Stability Index (PSI)

Quantifies distribution shift between baseline and current populations. Widely used in credit scoring and model monitoring.

### Formula

```
PSI = Σ (Pᵢ - Qᵢ) × ln(Pᵢ / Qᵢ)
```

Where:
- Pᵢ is the proportion in bin i for the baseline distribution
- Qᵢ is the proportion in bin i for the current distribution

### Industry Standard Interpretation

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| PSI < 0.1 | No significant change | None required |
| 0.1 ≤ PSI < 0.25 | Moderate change | Monitor closely |
| PSI ≥ 0.25 | Significant change | Investigation required |

### Implementation in Truthound

```python
from truthound.validators.drift import PSIValidator

validator = PSIValidator(
    column="score",
    baseline_data=baseline_df,
    n_bins=10,
    threshold=0.25
)
```

---

## 4. Chi-Square Test

Tests independence between observed and expected categorical frequencies.

### Formula

```
χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ
```

Where:
- Oᵢ is the observed frequency in category i
- Eᵢ is the expected frequency in category i

### Degrees of Freedom

```
df = (number of categories) - 1
```

### Interpretation

| p-value | Interpretation |
|---------|----------------|
| p < 0.05 | Categories have different distributions |
| p ≥ 0.05 | No significant difference |

### Implementation in Truthound

```python
from truthound.validators.drift import ChiSquareDriftValidator

validator = ChiSquareDriftValidator(
    column="category",
    baseline_data=baseline_df,
    alpha=0.05
)
```

---

## 5. Jensen-Shannon Divergence

Symmetric measure of distribution similarity, bounded between 0 and 1.

### Formula

```
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
```

Where:
- M = 0.5 × (P + Q)
- KL is the Kullback-Leibler divergence

### Properties

- **Symmetric**: JS(P||Q) = JS(Q||P)
- **Bounded**: 0 ≤ JS ≤ 1 (when using log base 2)
- **Metric**: Square root of JS is a valid distance metric

### Interpretation

| JS Value | Interpretation |
|----------|----------------|
| JS ≈ 0 | Distributions are identical |
| JS < 0.1 | Very similar distributions |
| 0.1 ≤ JS < 0.3 | Moderate difference |
| JS ≥ 0.3 | Significant difference |

### Implementation in Truthound

```python
from truthound.validators.drift import JensenShannonValidator

validator = JensenShannonValidator(
    column="feature",
    baseline_data=baseline_df,
    threshold=0.1
)
```

---

## 6. Kullback-Leibler Divergence

Measures information loss when approximating one distribution with another.

### Formula

```
KL(P||Q) = Σ P(x) × log(P(x) / Q(x))
```

### Properties

- **Asymmetric**: KL(P||Q) ≠ KL(Q||P)
- **Non-negative**: KL ≥ 0, with KL = 0 iff P = Q
- **Unbounded**: Can be infinite if Q(x) = 0 where P(x) > 0

### Use Cases

- Information theory applications
- Measuring model prediction quality
- Feature importance analysis

### Implementation in Truthound

```python
from truthound.validators.distribution import KLDivergenceValidator

validator = KLDivergenceValidator(
    column="probability",
    baseline_data=baseline_df,
    threshold=0.5
)
```

---

## 7. Mahalanobis Distance

Multivariate distance that accounts for correlations between variables.

### Formula

```
D = √((x - μ)ᵀ × Σ⁻¹ × (x - μ))
```

Where:
- x is the observation vector
- μ is the mean vector
- Σ is the covariance matrix

### Properties

- Scale-invariant
- Accounts for correlations between variables
- Reduces to Euclidean distance when variables are uncorrelated and have unit variance

### Interpretation (for multivariate normal data)

D² follows a chi-square distribution with p degrees of freedom (where p is the number of variables).

| Threshold | Interpretation |
|-----------|----------------|
| D > 3 | Potential outlier |
| D > χ²(p, 0.99) | Outlier at 99% confidence |

### Implementation in Truthound

```python
from truthound.validators.anomaly import MahalanobisValidator

validator = MahalanobisValidator(
    columns=["x", "y", "z"],
    threshold=3.0
)
```

---

## 8. Isolation Forest

Anomaly detection algorithm based on the principle that anomalies are easier to isolate.

### Algorithm

1. Randomly select a feature
2. Randomly select a split value between min and max
3. Recursively partition until isolation
4. Anomalies require fewer splits (shorter path length)

### Anomaly Score

```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where:
- h(x) is the path length for observation x
- c(n) is the average path length for n samples
- E(h(x)) is the expected path length

### Interpretation

| Score | Interpretation |
|-------|----------------|
| s ≈ 1 | Anomaly |
| s ≈ 0.5 | Normal |
| s < 0.5 | Definitely normal |

### Implementation in Truthound

```python
from truthound.validators.anomaly import IsolationForestValidator

validator = IsolationForestValidator(
    columns=["feature1", "feature2", "feature3"],
    contamination=0.05,
    max_anomaly_ratio=0.1
)
```

---

## 9. Wasserstein Distance

Also known as Earth Mover's Distance (EMD). Measures the minimum cost to transform one distribution into another.

### Formula (1D case)

```
W(P, Q) = ∫|F_P(x) - F_Q(x)| dx
```

Where F_P and F_Q are the cumulative distribution functions.

### Properties

- Metric (satisfies triangle inequality)
- Meaningful even when distributions have non-overlapping support
- Interpretable as "work" needed to move probability mass

### Implementation in Truthound

```python
from truthound.validators.drift import WassersteinValidator

validator = WassersteinValidator(
    column="value",
    baseline_data=baseline_df,
    threshold=0.5
)
```

---

## 10. Local Outlier Factor (LOF)

Density-based anomaly detection that compares local density of a point to its neighbors.

### Algorithm

1. Compute k-nearest neighbors for each point
2. Calculate local reachability density (LRD)
3. Compare LRD of a point to LRDs of its neighbors

### Formula

```
LOF(x) = (Σ LRD(neighbor) / LRD(x)) / k
```

### Interpretation

| LOF Value | Interpretation |
|-----------|----------------|
| LOF ≈ 1 | Normal (similar density to neighbors) |
| LOF > 1 | Lower density than neighbors (potential outlier) |
| LOF >> 1 | Significant outlier |

### Implementation in Truthound

```python
from truthound.validators.anomaly import LOFValidator

validator = LOFValidator(
    columns=["x", "y"],
    n_neighbors=20,
    threshold=1.5
)
```

---

## References

1. Tukey, J. W. (1977). "Exploratory Data Analysis"
2. Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
3. Pearson, K. (1900). "On the criterion that a given system of deviations..."
4. Lin, J. (1991). "Divergence measures based on the Shannon entropy"
5. Kullback, S., Leibler, R. A. (1951). "On Information and Sufficiency"
6. Mahalanobis, P. C. (1936). "On the generalized distance in statistics"
7. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
8. Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers"
9. Vaserstein, L. N. (1969). "Markov processes over denumerable products of spaces"

---

[← Back to README](../README.md)
