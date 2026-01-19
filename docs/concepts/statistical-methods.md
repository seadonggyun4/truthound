# Statistical Methods

This document provides a comprehensive reference for the statistical methods employed in Truthound for drift detection, anomaly detection, and distributional analysis.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Drift Detection Methods](#2-drift-detection-methods)
3. [Anomaly Detection Methods](#3-anomaly-detection-methods)
4. [Distribution Analysis](#4-distribution-analysis)
5. [Statistical Thresholds](#5-statistical-thresholds)
6. [Method Selection Guide](#6-method-selection-guide)
7. [Mathematical Foundations](#7-mathematical-foundations)
8. [References](#8-references)

---

## 1. Overview

Truthound implements a comprehensive suite of statistical methods for data quality validation:

| Category | Methods | Primary Use Case |
|----------|---------|------------------|
| Drift Detection | 13 methods | Distribution comparison between datasets |
| Anomaly Detection | 15 methods | Outlier and anomaly identification |
| Distribution Analysis | 8 methods | Statistical characterization |

All methods are optimized for Polars LazyFrame execution, enabling efficient processing of large-scale datasets.

---

## 2. Drift Detection Methods

### 2.1 Kolmogorov-Smirnov Test (KS)

The Kolmogorov-Smirnov test measures the maximum distance between two empirical cumulative distribution functions.

**Mathematical Definition**:

```
D = max|F₁(x) - F₂(x)|
```

Where F₁ and F₂ are the empirical CDFs of the two samples.

**Usage**:

```python
drift = th.compare(baseline, current, method="ks")
```

**Characteristics**:

| Aspect | Description |
|--------|-------------|
| Best For | Continuous numeric distributions |
| Sensitivity | Shape and location differences |
| Output | D-statistic (0-1), p-value |
| Threshold | p-value < 0.05 indicates drift |

**Interpretation**:

| p-value | Interpretation |
|---------|----------------|
| p < 0.05 | Significant difference (reject null hypothesis) |
| p >= 0.05 | No significant difference |

### 2.2 Population Stability Index (PSI)

PSI quantifies how much a variable's distribution has shifted between two samples. Widely used in credit scoring and model monitoring.

**Mathematical Definition**:

```
PSI = Σ (Pᵢ - Qᵢ) × ln(Pᵢ / Qᵢ)
```

Where Pᵢ and Qᵢ are the proportions in bin i for baseline and current distributions.

**Usage**:

```python
drift = th.compare(baseline, current, method="psi")
```

**Interpretation**:

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No significant shift | None required |
| 0.1 - 0.25 | Moderate shift | Monitor closely |
| > 0.25 | Significant shift | Investigation required |

**Characteristics**:
- Automatic decile binning
- Smoothing applied to prevent division by zero
- Industry standard for model monitoring

### 2.3 Chi-Square Test

The chi-square test assesses independence between observed and expected categorical frequencies.

**Mathematical Definition**:

```
χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ
```

Where Oᵢ is the observed frequency and Eᵢ is the expected frequency.

**Usage**:

```python
drift = th.compare(baseline, current, method="chi2")
```

**Characteristics**:

| Aspect | Description |
|--------|-------------|
| Best For | Categorical variables |
| Output | χ²-statistic, p-value |
| Degrees of Freedom | k - 1 (where k is number of categories) |

### 2.4 Jensen-Shannon Divergence (JS)

Jensen-Shannon divergence is a symmetrized and smoothed version of KL divergence.

**Mathematical Definition**:

```
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
```

Where M = 0.5 × (P + Q).

**Usage**:

```python
drift = th.compare(baseline, current, method="js")
```

**Interpretation**:

| JS Value | Interpretation |
|----------|----------------|
| JS ≈ 0 | Distributions are identical |
| JS < 0.1 | Very similar distributions |
| 0.1 <= JS < 0.3 | Moderate difference |
| JS >= 0.3 | Significant difference |

**Properties**:
- **Symmetric**: JS(P||Q) = JS(Q||P)
- **Bounded**: 0 <= JS <= 1 (when using log base 2)
- **Metric**: Square root of JS is a valid distance metric

### 2.5 Kullback-Leibler Divergence (KL)

KL divergence measures information loss when approximating one distribution with another.

**Mathematical Definition**:

```
KL(P||Q) = Σ P(x) × log(P(x) / Q(x))
```

**Usage**:

```python
drift = th.compare(baseline, current, method="kl")
```

**Properties**:
- **Asymmetric**: KL(P||Q) ≠ KL(Q||P)
- **Non-negative**: KL >= 0, with KL = 0 iff P = Q
- **Unbounded**: Can be infinite if Q(x) = 0 where P(x) > 0

### 2.6 Wasserstein Distance (Earth Mover's Distance)

Wasserstein distance measures the minimum "work" required to transform one distribution into another.

**Mathematical Definition (1D case)**:

```
W(P, Q) = ∫|F_P(x) - F_Q(x)| dx
```

Where F_P and F_Q are the cumulative distribution functions.

**Usage**:

```python
drift = th.compare(baseline, current, method="wasserstein")
```

**Characteristics**:
- Metric (satisfies triangle inequality)
- Meaningful even when distributions have non-overlapping support
- Interpretable as "work" needed to move probability mass
- Scale-dependent: sensitive to the scale of the variable

### 2.7 Cramér-von Mises Test

Cramér-von Mises is an alternative to KS that integrates squared differences between CDFs.

**Mathematical Definition**:

```
ω² = ∫[Fₙ(x) - F(x)]² dF(x)
```

**Usage**:

```python
drift = th.compare(baseline, current, method="cvm")
```

**Characteristics**:
- More sensitive to differences in tails than KS
- Output: ω² statistic, p-value

### 2.8 Anderson-Darling Test

Anderson-Darling test gives more weight to the tails of the distribution.

**Usage**:

```python
drift = th.compare(baseline, current, method="anderson")
```

**Characteristics**:
- More sensitive to tail deviations than KS or CvM
- Useful for detecting differences in distribution extremes

### 2.9 Additional Drift Methods

| Method | Function | Best For |
|--------|----------|----------|
| `hellinger` | Hellinger distance | Probability distributions |
| `bhattacharyya` | Bhattacharyya distance | Probability overlap |
| `total_variation` | Total variation distance | Binary classification |
| `energy` | Energy distance | Multivariate distributions |
| `mmd` | Maximum Mean Discrepancy | High-dimensional data |

---

## 3. Anomaly Detection Methods

### 3.1 Z-Score Method

Z-score identifies outliers based on standard deviations from the mean.

**Mathematical Definition**:

```
z = (x - μ) / σ
```

**Usage**:

```python
from truthound.ml import ZScoreAnomalyDetector

detector = ZScoreAnomalyDetector(threshold=3.0)
detector.fit(df)
result = detector.detect(df)
```

**Characteristics**:

| Aspect | Description |
|--------|-------------|
| Best For | Normally distributed data |
| Threshold | Typically |z| > 3 (0.3% of normal data) |
| Assumption | Data is approximately Gaussian |

### 3.2 Interquartile Range (IQR)

IQR method uses quartiles to define outlier boundaries.

**Mathematical Definition**:

```
IQR = Q3 - Q1
Lower Bound = Q1 - k × IQR
Upper Bound = Q3 + k × IQR
```

**Usage**:

```python
from truthound.ml import IQRAnomalyDetector

detector = IQRAnomalyDetector(multiplier=1.5)
```

**Parameters**:

| k Value | Description |
|---------|-------------|
| 1.5 | Standard outliers |
| 3.0 | Extreme outliers |

**Characteristics**:
- Distribution-free and robust to extreme values
- Works best for symmetric or approximately symmetric distributions
- Resistant to extreme outliers

### 3.3 Modified Z-Score (MAD)

Modified Z-score uses Median Absolute Deviation for robustness.

**Mathematical Definition**:

```
MAD = median(|xᵢ - median(x)|)
M = 0.6745 × (xᵢ - median(x)) / MAD
```

**Usage**:

```python
from truthound.ml import MADAnomalyDetector

detector = MADAnomalyDetector(threshold=3.5)
```

**Characteristics**:
- Highly resistant to outliers
- Threshold: |M| > 3.5 typically

### 3.4 Isolation Forest

Isolation Forest isolates anomalies by random recursive partitioning.

**Principle**: Anomalies are easier to isolate and require fewer splits.

**Anomaly Score**:

```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where:
- h(x) is the path length for observation x
- c(n) is the average path length for n samples
- E(h(x)) is the expected path length

**Usage**:

```python
from truthound.ml import IsolationForestDetector

detector = IsolationForestDetector(
    contamination=0.1,
    n_estimators=100
)
detector.fit(df)
result = detector.detect(df)
```

**Interpretation**:

| Score | Interpretation |
|-------|----------------|
| s ≈ 1 | Anomaly |
| s ≈ 0.5 | Normal |
| s < 0.5 | Definitely normal |

**Characteristics**:
- Linear time complexity O(n)
- Effective for high-dimensional data

### 3.5 Local Outlier Factor (LOF)

LOF identifies anomalies based on local density deviation.

**Algorithm**:
1. Compute k-nearest neighbors for each point
2. Calculate local reachability density (LRD)
3. Compare LRD of a point to LRDs of its neighbors

**Formula**:

```
LOF(x) = (Σ LRD(neighbor) / LRD(x)) / k
```

**Usage**:

```python
from truthound.ml import LOFDetector

detector = LOFDetector(n_neighbors=20)
```

**Interpretation**:

| LOF Value | Interpretation |
|-----------|----------------|
| LOF ≈ 1 | Normal (similar density to neighbors) |
| LOF > 1 | Lower density than neighbors (potential outlier) |
| LOF >> 1 | Significant outlier |

### 3.6 Mahalanobis Distance

Mahalanobis distance accounts for correlations between variables.

**Mathematical Definition**:

```
D = √((x - μ)ᵀ × Σ⁻¹ × (x - μ))
```

Where:
- x is the observation vector
- μ is the mean vector
- Σ is the covariance matrix

**Usage**:

```python
from truthound.ml import MahalanobisDetector

detector = MahalanobisDetector(threshold=3.0)
```

**Characteristics**:
- Scale-invariant
- Accounts for correlations between variables
- D² follows a chi-square distribution with p degrees of freedom

### 3.7 Additional Anomaly Methods

| Method | Approach | Best For |
|--------|----------|----------|
| `grubbs` | Grubbs' test | Single outliers in univariate data |
| `esd` | Generalized ESD | Multiple outliers in univariate data |
| `dbscan` | Density-based clustering | Arbitrary-shaped clusters |
| `svm` | One-Class SVM | Non-linear boundaries |
| `autoencoder` | Reconstruction error | High-dimensional, complex patterns |
| `percentile` | Percentile bounds | Simple threshold-based detection |
| `tukey` | Tukey fences | Robust statistical bounds |

### 3.8 Ensemble Methods

```python
from truthound.ml import EnsembleAnomalyDetector

ensemble = EnsembleAnomalyDetector(
    detectors=[zscore_detector, iqr_detector, iso_detector],
    voting_strategy="majority"  # or "unanimous", "any"
)
```

---

## 4. Distribution Analysis

### 4.1 Normality Tests

| Test | Method | Best For |
|------|--------|----------|
| Shapiro-Wilk | W statistic | Small samples (n < 5000) |
| D'Agostino-Pearson | Skewness + kurtosis | Medium samples |
| Kolmogorov-Smirnov | CDF comparison | Large samples |
| Anderson-Darling | Weighted CDF | General use |

### 4.2 Descriptive Statistics

```python
profile = th.profile(df)

# Available statistics per column:
# - count, null_count, null_ratio
# - mean, std, variance
# - min, max, range
# - q25, median (q50), q75
# - skewness, kurtosis
# - unique_count, unique_ratio
```

### 4.3 Entropy and Information

**Shannon Entropy**:

```
H(X) = -Σ p(xᵢ) × log₂(p(xᵢ))
```

**Usage**:

```python
profile = th.profile(df)
# Entropy available in profile.columns[col].entropy
```

---

## 5. Statistical Thresholds

### 5.1 Default Thresholds

| Method | Threshold | Interpretation |
|--------|-----------|----------------|
| Z-Score | 3.0 | > 3 standard deviations |
| IQR | 1.5 | Outside 1.5 × IQR |
| MAD | 3.5 | > 3.5 modified z-scores |
| LOF | 1.5 | LOF score > 1.5 |
| Isolation Forest | 0.1 | Top 10% anomaly scores |
| KS Test | 0.05 | p-value threshold |
| PSI | 0.25 | Significant drift threshold |
| Chi-Square | 0.05 | p-value threshold |

### 5.2 Threshold Configuration

```python
# Drift detection with custom thresholds
drift = th.compare(
    baseline, current,
    method="psi",
    threshold=0.1  # More sensitive threshold
)

# Anomaly detection with custom thresholds
from truthound.ml import ZScoreAnomalyDetector

detector = ZScoreAnomalyDetector(threshold=2.5)  # More sensitive
```

---

## 6. Method Selection Guide

### 6.1 By Data Type

| Data Type | Drift Method | Anomaly Method |
|-----------|--------------|----------------|
| Continuous | KS, Wasserstein | Z-Score, IQR, Isolation Forest |
| Categorical | Chi-Square, JS | Mode deviation, category frequency |
| Ordinal | KS, Wasserstein | IQR, percentile |
| High-dimensional | MMD, Energy | Isolation Forest, Autoencoder |
| Time Series | KS with windows | LOF, ARIMA residuals |

### 6.2 By Sample Size

| Sample Size | Recommended Methods |
|-------------|---------------------|
| n < 100 | Exact tests, bootstrap |
| 100 < n < 10,000 | KS, Chi-Square, Z-Score |
| n > 10,000 | PSI, JS, Isolation Forest |
| n > 1,000,000 | Sampled methods, streaming |

### 6.3 By Sensitivity Requirements

| Requirement | Methods |
|-------------|---------|
| High sensitivity | Anderson-Darling, MAD (low threshold) |
| Balanced | KS, PSI, IQR |
| Low false positives | Mahalanobis, Ensemble voting |

### 6.4 Decision Tree

```
Is data continuous?
├─ Yes
│   ├─ Normally distributed? → Z-Score for anomaly, KS for drift
│   └─ Skewed/Unknown? → IQR for anomaly, PSI for drift
└─ No (Categorical)
    ├─ Few categories (< 20)? → Chi-Square, mode deviation
    └─ Many categories? → Frequency analysis, JS divergence

Is data high-dimensional?
├─ Yes (> 10 features) → Isolation Forest, MMD
└─ No → Standard univariate methods

Are there existing outliers?
├─ Yes → MAD, IQR (robust methods)
└─ No → Z-Score, Mahalanobis
```

---

## 7. Mathematical Foundations

### 7.1 Empirical Distribution Functions

The empirical CDF is defined as:

```
F̂ₙ(x) = (1/n) × Σ 1_{Xᵢ ≤ x}
```

This forms the basis for KS, CvM, and Anderson-Darling tests.

### 7.2 Information Theory Basics

**Entropy** measures uncertainty:

```
H(X) = -Σ P(x) × log P(x)
```

**Relative Entropy (KL Divergence)** measures distribution difference:

```
D_KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

### 7.3 Hypothesis Testing Framework

Statistical tests follow the framework:

1. **Null Hypothesis (H₀)**: No difference between distributions
2. **Alternative Hypothesis (H₁)**: Distributions differ
3. **Test Statistic**: Computed from data
4. **p-value**: Probability of observing test statistic under H₀
5. **Decision**: Reject H₀ if p-value < α (typically 0.05)

### 7.4 Multiple Testing Correction

When testing multiple columns:

| Method | Formula | Use Case |
|--------|---------|----------|
| Bonferroni | α' = α / n | Conservative, independent tests |
| Holm | Sequential adjustment | Less conservative |
| Benjamini-Hochberg | FDR control | Many tests, some false positives acceptable |

```python
# Truthound applies Benjamini-Hochberg by default for multiple columns
drift = th.compare(baseline, current, correction="bh")
```

---

## 8. References

1. Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
2. Smirnov, N. (1948). "Table for Estimating the Goodness of Fit of Empirical Distributions"
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
4. Breunig, M. M., et al. (2000). "LOF: Identifying Density-Based Local Outliers"
5. Iglewicz, B., & Hoaglin, D. C. (1993). "How to Detect and Handle Outliers"
6. Kullback, S., & Leibler, R. A. (1951). "On Information and Sufficiency"
7. Lin, J. (1991). "Divergence measures based on the Shannon entropy"
8. Mahalanobis, P. C. (1936). "On the generalized distance in statistics"
9. Vaserstein, L. N. (1969). "Markov processes over denumerable products of spaces"
10. Tukey, J. W. (1977). "Exploratory Data Analysis"
11. Pearson, K. (1900). "On the criterion that a given system of deviations..."

---

## See Also

- [Validators Reference](VALIDATORS.md) — All validator implementations
- [Advanced Features](ADVANCED.md) — ML module documentation
- [API Reference](API_REFERENCE.md) — Complete API documentation
