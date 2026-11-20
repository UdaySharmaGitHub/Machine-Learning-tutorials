# Feature Scaling and Preprocessing for KNN

## Table of Contents

- [Overview](#overview)
- [Why Feature Scaling Is Critical for KNN](#why-feature-scaling-is-critical-for-knn)
- [Why Scaling Changes Neighbours](#why-scaling-changes-neighbours)
- [Scaling Methods](#scaling-methods)
  - [1. Standardization (Z-Score Normalization)](#1-standardization-z-score-normalization)
  - [2. Min-Max Scaling (Normalization)](#2-min-max-scaling-normalization)
  - [3. Robust Scaling](#3-robust-scaling)
  - [4. Log Scaling](#4-log-scaling)
  - [5. L2 / Unit Normalization](#5-l2--unit-normalization)
- [Scaling Methods Comparison](#scaling-methods-comparison)
- [Distance Metric and Scaling Choice](#distance-metric-and-scaling-choice)
- [Outlier Detection and Treatment](#outlier-detection-and-treatment)
- [sklearn Implementation](#sklearn-implementation)
- [Special Cases](#special-cases)
  - [Constant Features](#constant-features)
  - [Missing Values](#missing-values)
  - [Categorical Variables](#categorical-variables)
- [Correct Preprocessing Pipeline](#correct-preprocessing-pipeline)
- [Common Pitfalls](#common-pitfalls)
- [Scaling vs Feature Relevance](#scaling-vs-feature-relevance)

---

## Overview

Because ***KNN ranks distances***, a ***feature's numerical unit*** acts like an ***implicit weight***. Sources consistently emphasize ***scaling as a critical practical step*** for KNN success. Without proper scaling, features with ***larger numerical ranges*** will ***dominate the distance calculation***, even if they are ***less predictive***.

> [!IMPORTANT]
> ***Feature scaling is not optional in KNN*** — it can be the difference between a model that works well and one that performs poorly due to one or two features overwhelming the distance metric.

---

## Why Feature Scaling Is Critical for KNN

### The Distance Weighting Problem

Because KNN uses ***distance-based decisions***, features with ***larger numerical scales*** implicitly receive ***higher weight*** in the distance calculation. This can lead to:

- ***Bias toward high-magnitude features*** regardless of their predictive importance
- ***Ignoring low-magnitude features*** that might be highly predictive
- ***Inconsistent model behavior*** across different feature scales

> [!WARNING]
> ***Unscaled features can completely change the rank ordering of neighbors***, leading to poor predictions. A single large-scale feature can dominate the entire decision boundary.

---

## Why Scaling Changes Neighbours

Suppose two features are ***age (in years)*** and ***annual income (in rupees)***.

**For two people:**
- Age difference: $\Delta \text{age} = 10$ years
- Income difference: $\Delta \text{income} = 100,000$ rupees

**Contribution to squared Euclidean distance:**
- Age: $(\Delta \text{age})^2 = 10^2 = 100$
- Income: $(\Delta \text{income})^2 = (100,000)^2 = 10^{10}$

The ***income difference dominates*** by a factor of $100,000,000$! Even if ***age is more predictive***, the model will ignore it because income nearly determines the neighbor ranking.

| Feature | Scale | Distance Contribution | Influence |
|---------|-------|----------------------|-----------|
| ***Age (years)*** | 10–100 | $10^2 = 100$ | ~0% |
| ***Income (rupees)*** | 10K–10M | $(100K)^2 = 10^{10}$ | ~99.9% |

**After scaling both to [0, 1]:**
| Feature | Scale | Distance Contribution | Influence |
|---------|-------|----------------------|-----------|
| ***Age (scaled)*** | 0–1 | ~0.01 | ~50% |
| ***Income (scaled)*** | 0–1 | ~0.01 | ~50% |

Both features now contribute ***equally*** to the distance calculation.

---

## Scaling Methods

### 1. Standardization (Z-Score Normalization)

***Standardization*** transforms each feature to have ***zero mean*** and ***unit standard deviation***.

**Formula:**

For feature $j$, estimate training ***mean*** $\mu_j$ and ***standard deviation*** $s_j$:

$$
\boxed{z_{ij} = \frac{x_{ij} - \mu_j}{s_j}}
$$

Where:
- ***$x_{ij}$*** = original value of feature $j$ for observation $i$
- ***$\mu_j$*** = mean of feature $j$ from training data
- ***$s_j$*** = standard deviation of feature $j$ from training data
- ***$z_{ij}$*** = standardized value

**Example:**

Feature: Height (in cm) with $\mu = 170$, $s = 10$
- Original: $x = 185$ cm
- Standardized: $z = \frac{185 - 170}{10} = 1.5$

**Properties:**

- ***Centered near zero*** with ***unit variance***
- ***Symmetric distribution*** (if original was symmetric)
- ***Sensitive to extreme outliers*** because both mean and standard deviation are affected
- Produces values typically in the range ***[-3, +3]***

**Use When:**

- Features are ***roughly normally distributed***
- ***Outliers are rare*** or not a concern
- You need ***interpretable standardized values***

> [!NOTE]
> Standardization ***does not*** make a distribution Gaussian. It only centers and scales; if the original is skewed, the standardized version will remain skewed.

---

### 2. Min-Max Scaling (Normalization)

***Min-Max scaling*** rescales each feature to lie in a fixed range, typically ***[0, 1]***.

**Formula:**

$$
\boxed{x'_{ij} = \frac{x_{ij} - m_j}{M_j - m_j}}
$$

Where:
- ***$m_j$*** = training ***minimum*** of feature $j$
- ***$M_j$*** = training ***maximum*** of feature $j$
- ***$x'_{ij}$*** = scaled value in $[0, 1]$

**Example:**

Feature: Income (in rupees) with $m = 50,000$, $M = 500,000$
- Original: $x = 250,000$ rupees
- Scaled: $x' = \frac{250,000 - 50,000}{500,000 - 50,000} = \frac{200,000}{450,000} \approx 0.444$

**Properties:**

- Guaranteed to map training data to ***[0, 1]***
- ***Simple and interpretable***
- ***Future values may fall outside [0, 1]*** if they exceed training min/max
- ***One extreme value can compress all others*** — a single outlier in training data makes min or max extreme

**Use When:**

- You need values in a ***fixed, interpretable range***
- Features have ***natural bounds*** (e.g., percentages 0–100)
- ***No extreme outliers*** in training data

> [!WARNING]
> A single outlier in training data can ***compress*** almost all other observations into a narrow range. If the min is 1 and max is 1000 due to an outlier, all normal values between 1–100 get mapped to [0, 0.1], losing resolution.

---

### 3. Robust Scaling

***Robust scaling*** uses the ***median*** and ***interquartile range (IQR)*** instead of mean and standard deviation, making it ***resistant to outliers***.

**Formula:**

Using ***median*** $q_{0.5,j}$ and ***IQR*** $IQR_j = q_{0.75,j} - q_{0.25,j}$:

$$
\boxed{x'_{ij} = \frac{x_{ij} - q_{0.5,j}}{IQR_j}}
$$

Where:
- ***$q_{0.5,j}$*** = ***50th percentile*** (median) of feature $j$
- ***$q_{0.75,j}$*** = ***75th percentile*** of feature $j$
- ***$q_{0.25,j}$*** = ***25th percentile*** of feature $j$
- ***$IQR_j$*** = ***interquartile range*** (difference between 75th and 25th percentiles)

**Example:**

Feature: Salary (in thousands) with $q_{0.25} = 30$, $q_{0.5} = 50$, $q_{0.75} = 70$, so $IQR = 40$
- Original: $x = 200$ (extreme outlier)
- Robust scaled: $x' = \frac{200 - 50}{40} = \frac{150}{40} = 3.75$

**Properties:**

- ***Resistant to outliers*** — uses median and IQR which are robust statistics
- Values outside $[0, 1]$ are expected (they represent extremes)
- ***Zero IQR*** (constant feature or all values the same) requires special handling
- Outliers still affect the calculation but much less severely

**Use When:**

- Data contains ***significant outliers***
- Features have ***heavy-tailed distributions***
- You want ***robust scaling*** without data loss

> [!TIP]
> ***Robust scaling is the best choice for real-world data*** because datasets often contain outliers or skewed distributions. The median and IQR are much less affected by extreme values than mean and standard deviation.

---

### 4. Log Scaling

***Log scaling*** applies a logarithmic transformation, useful for ***skewed features*** with a wide range.

**Formula:**

$$
\boxed{x'_{ij} = \log(x_{ij} + \epsilon)}
$$

Where $\epsilon$ is a small constant (e.g., 1) to handle zero or negative values.

**Example:**

Feature: Website traffic (highly skewed, 1–10,000,000 visits)
- Original: $x = 1,000,000$ visits
- Log-scaled: $x' = \log(1,000,000 + 1) \approx 6.0$

**Properties:**

- ***Compresses large values*** and ***spreads small values***
- Transforms ***exponential/power-law distributions*** to more linear relationships
- Requires ***positive values*** (or adding a constant offset)
- Often combined with standardization afterward

**Use When:**

- Features follow a ***power-law or exponential distribution***
- There's ***extreme range variation*** (e.g., 1 to 1 million)
- Domain knowledge suggests a ***multiplicative*** relationship

---

### 5. L2 / Unit Normalization

***L2 normalization*** (also called ***unit normalization*** or ***vector normalization***) scales each ***sample*** (row) so that its Euclidean length equals 1. This is ***per-sample***, unlike the other methods which are ***per-feature*** (column-wise).

**Formula:**

$$
\boxed{x'_i = \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2} = \frac{\mathbf{x}_i}{\sqrt{\sum_j x_{ij}^2}}}
$$

**Example:**

Sample: $\mathbf{x} = (3, 4, 0)$
- $\|\mathbf{x}\|_2 = \sqrt{9 + 16 + 0} = 5$
- Normalized: $\mathbf{x}' = (0.6,\ 0.8,\ 0.0)$

**Properties:**

- ***Each row*** becomes a unit vector (length = 1)
- ***Magnitude is discarded*** — only direction matters
- ***Not fitted*** on training data — no statistics to store or leak
- Independent of other samples — can be applied to any row independently

**Use When:**

- Using ***cosine distance*** — two unit vectors have cosine similarity equal to their dot product
- ***Text / NLP*** data (TF-IDF vectors, word embeddings)
- ***High-dimensional sparse data*** where magnitude is uninformative
- Features already represent ***proportions or ratios***

> [!NOTE]
> ***L2 normalization is the only scaling method that operates row-wise*** (per sample). All others — Standardization, Min-Max, Robust, Log — operate column-wise (per feature). These serve entirely different purposes and can be combined.

> [!TIP]
> When using ***cosine distance in KNN***, apply L2 normalization to the entire dataset. After normalization, cosine similarity between two samples equals their dot product, making the KNN query computationally cheap.

---

## Scaling Methods Comparison

| Method | Formula | Output Range | Sensitive to Outliers | Best For | Requires Fit? |
|--------|---------|-------------|----------------------|----------|---------------|
| ***Standardization*** | $(x - \mu) / s$ | $(-\infty, +\infty)$, ~[-3, 3] | ***Yes*** | Normal distributions, no outliers | ✅ Yes |
| ***Min-Max*** | $(x - \min) / (\max - \min)$ | $[0, 1]$ | ***Very much*** | Fixed range needed, no outliers | ✅ Yes |
| ***Robust*** | $(x - q_{0.5}) / IQR$ | $(-\infty, +\infty)$ | ***No*** | Real-world data with outliers | ✅ Yes |
| ***Log*** | $\log(x + \epsilon)$ | $(-\infty, +\infty)$ | Partially | Right-skewed, exponential distributions | ❌ No |
| ***L2 Unit*** | $x / \|x\|_2$ | $[-1, 1]$ | ***No*** | Cosine distance, text/NLP, sparse data | ❌ No |

> [!IMPORTANT]
> ***Choose based on your data distribution and distance metric.*** No single scaler is best in all situations. When in doubt between Standardization and Robust, prefer ***Robust*** for real-world datasets.

---

## Distance Metric and Scaling Choice

The ***distance metric*** you choose for KNN directly influences which scaling method is most appropriate.

| Distance Metric | Recommended Scaling | Reason |
|----------------|--------------------|----|
| ***Euclidean (L₂)*** | Standardization or Robust | All features must be on the same scale; mean-centered helps when features have different distributions |
| ***Manhattan (L₁)*** | Min-Max or Standardization | Works with either; min-max gives bounded contributions per feature |
| ***Minkowski (Lₚ)*** | Standardization | Same reasoning as Euclidean — features must be comparable |
| ***Cosine*** | L2 Unit Normalization | Cosine distance ignores magnitude; normalizing rows makes dot-product = cosine similarity |
| ***Hamming*** | None needed | Operates on binary/categorical data; scaling does not apply |
| ***Mahalanobis*** | None (built-in) | Uses inverse covariance matrix — inherently accounts for scale and correlations |
| ***Chebyshev (L∞)*** | Standardization or Min-Max | Maximum difference determines distance — features must be comparable |
| ***Jaccard*** | None needed | Operates on sets; magnitude is irrelevant |

> [!NOTE]
> ***Mahalanobis distance is scale-invariant by construction*** because it uses the inverse covariance matrix, which captures both variance and correlations. Standard feature scaling is redundant when using Mahalanobis.

---

## Outlier Detection and Treatment

Outliers affect scaling — especially Standardization and Min-Max. Before scaling, decide how to handle them.

### Step 1: Detect Outliers

**Method 1 — Z-Score:**

A point is an outlier if its standardized value exceeds a threshold (commonly $|z| > 3$):

$$
z = \frac{x - \mu}{\sigma}, \quad \text{outlier if } |z| > 3
$$

**Method 2 — IQR Rule (more robust):**

$$
\text{Outlier if } x < Q_1 - 1.5 \times IQR \quad \text{or} \quad x > Q_3 + 1.5 \times IQR
$$

**Method 3 — Isolation Forest / DBSCAN** — algorithmic detection for multivariate outliers.

### Step 2: Choose a Treatment

| Treatment | Description | When to Use |
|-----------|-------------|-------------|
| ***Remove*** | Delete the outlier row | Confirmed data entry error; very few outliers |
| ***Winsorize / Clip*** | Cap at a percentile (e.g., 1st–99th) | Outliers are real but extreme; want to keep samples |
| ***Log transform*** | Compress high values; reduces outlier impact | Right-skewed features |
| ***Robust scaling*** | Scale using median and IQR | Outliers exist but shouldn't be removed |
| ***Keep as-is*** | No treatment | Outliers are meaningful signals (fraud, anomalies) |

```python
import numpy as np

# Winsorization: clip feature to [1st, 99th] percentile
lower = np.percentile(X_train[:, j], 1)
upper = np.percentile(X_train[:, j], 99)
X_train[:, j] = np.clip(X_train[:, j], lower, upper)
X_test[:, j] = np.clip(X_test[:, j], lower, upper)  # use training percentiles!
```

> [!WARNING]
> ***Compute outlier thresholds (percentiles, IQR bounds) on training data only*** and apply the same thresholds to test data. Never detect outliers using test-set statistics.

---

## sklearn Implementation

Practical code for each scaling method using `scikit-learn`:

```python
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    FunctionTransformer
)
import numpy as np

X_train = ...  # your training data (numpy array or DataFrame)
X_test  = ...  # your test data

# 1. Standardization (Z-Score)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # fit + transform training
X_test_std  = scaler.transform(X_test)         # transform test with training stats

# 2. Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_mm = scaler.fit_transform(X_train)
X_test_mm  = scaler.transform(X_test)

# 3. Robust Scaling
scaler = RobustScaler()
X_train_rb = scaler.fit_transform(X_train)
X_test_rb  = scaler.transform(X_test)

# 4. Log Scaling
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
X_train_log = log_transformer.fit_transform(X_train)  # log1p = log(x + 1), safe for zeros
X_test_log  = log_transformer.transform(X_test)

# 5. L2 Unit Normalization (per-sample, row-wise)
normalizer = Normalizer(norm='l2')
X_train_l2 = normalizer.fit_transform(X_train)
X_test_l2  = normalizer.transform(X_test)
```

> [!TIP]
> ***Use `fit_transform(X_train)` and `transform(X_test)` — never `fit_transform(X_test)`***. The `fit_transform` step on training data learns the statistics; `transform` on test data applies those same learned statistics without re-fitting.

---


### Constant Features

A ***constant feature*** (all values identical) provides ***no neighbor discrimination***.

**Problems:**

- ***Standardization:*** Division by $s = 0$ is undefined
- ***Min-Max scaling:*** $M_j = m_j$, so denominator is zero
- ***Robust scaling:*** $IQR = 0$, denominator is zero

**Solutions:**

1. ***Remove the constant feature*** — it provides no information
2. ***Use a robust transformer*** that handles this edge case
3. ***Feature engineering:*** Create a new feature combining it with others

> [!IMPORTANT]
> ***Constant features must be removed before scaling***. They add no predictive power and will cause scaling to fail or produce NaN values.

---

### Missing Values

***KNN cannot compute distance with missing values***. Before scaling, impute missing data.

**Imputation Methods:**

- ***Mean imputation:*** Replace with column mean (simple but may reduce variance)
- ***Median imputation:*** Replace with median (robust to outliers)
- ***K-NN imputation:*** Use K neighbors to predict missing values (iterative)
- ***Forward/backward fill:*** For time series data
- ***Remove observations:*** If few values are missing

> [!WARNING]
> ***Never scale before imputing***. Fit imputation statistics on training data only, then apply to validation/test data.

---

### Categorical Variables

***KNN works on numerical distances***, so categorical variables must be encoded.

**Encoding Methods:**

- ***Label encoding:*** Map categories to integers (1, 2, 3, ...) — use only for ordinal categories
- ***One-hot encoding:*** Binary vector for each category — use for nominal categories, creates many features
- ***Target encoding:*** Replace with target mean — be careful of data leakage

**Example (One-hot encoding):**

| Original | Encoded |
|----------|---------|
| Red | [1, 0, 0] |
| Green | [0, 1, 0] |
| Blue | [0, 0, 1] |

> [!NOTE]
> After encoding, ***scale the resulting numerical features***. One-hot encoded features are typically binary [0, 1] and don't need scaling, but ordinal-encoded features do.

---

## Correct Preprocessing Pipeline

The order of operations is ***critical*** to avoid data leakage.

### Step-by-Step Procedure

1. ***Split raw data*** into training, validation, and test sets
2. ***Fit*** imputation, encoding, scaling, feature selection, or PCA ***on the training fold only***
3. ***Transform*** the training fold with learned statistics
4. ***Transform*** validation and test folds with the ***same learned statistics***
5. ***Repeat inside cross-validation:*** For each fold, refit transformers on that fold's training portion
6. ***After model selection:*** Refit the entire pipeline on all training data (training + validation combined)
7. ***Transform and evaluate*** the untouched test set ***once***

### Pseudo-code

```python
# CORRECT: Fit on training, transform validation/test with training statistics
scaler = StandardScaler()
scaler.fit(X_train)          # Learn mean, std from training
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# WRONG: Scaling entire dataset before splitting
scaler.fit(X_all)            # Leaks information from val/test
X_all_scaled = scaler.transform(X_all)
X_train, X_val, X_test = split(X_all_scaled)  # Data leakage!
```

### Why This Matters

***Scaling the entire dataset before splitting*** leaks ***distribution information*** from validation and test observations. This inflates performance estimates and leads to poor generalization.

> [!CAUTION]
> ***Data leakage through preprocessing is a critical error***. Validation and test statistics (min, max, mean, std) must be unknown when fitting. Always fit transformers on training data only.

### Using a Pipeline Library

Modern ML libraries like `scikit-learn` provide ***Pipeline*** objects that automate this order:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Inside cross-validation, fit on fold training, transform all
pipeline.fit(X_train_fold, y_train_fold)
y_val_pred = pipeline.predict(X_val_fold)
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| ***No scaling at all*** | Large-scale features dominate; poor predictions | ***Always scale KNN features*** |
| ***Scaling before splitting*** | Data leakage; inflated performance estimates | ***Fit scaler on training data only*** |
| ***Forgetting to transform test data*** | Test set scaled differently; inconsistent predictions | ***Use same fitted scaler for test*** |
| ***Scaling categorical after one-hot*** | Binary encoded features don't need scaling | ***Check if already normalized*** |
| ***Using min-max with outliers*** | Single outlier compresses all values | ***Use robust scaling instead*** |
| ***Imputing after scaling*** | Missing value imputation doesn't use original scale | ***Impute before scaling*** |

---

## Scaling vs Feature Relevance

***Unit variance makes numerical contributions comparable***, but an ***irrelevant standardized feature*** still adds ***noise*** and can reduce model performance.

### What Scaling Does and Doesn't Do

**Scaling DOES:**
- Make features comparable in magnitude
- Prevent high-scale features from dominating
- Improve distance-based algorithm performance
- Stabilize optimization in some algorithms

**Scaling DOES NOT:**
- Remove irrelevant features
- Improve feature correlation with target
- Handle feature redundancy
- Solve fundamental feature engineering problems

### Solutions for Irrelevant Features

If a standardized feature still adds noise:

1. ***Feature selection:*** Remove features with low correlation to target
2. ***Domain-aware weighting:*** Assign lower weights to less predictive features
3. ***PCA:*** Reduce dimensionality while retaining variance
4. ***Metric learning:*** Learn feature weights from data
5. ***Different model:*** Tree-based models are invariant to scaling
6. ***Feature engineering:*** Create new, more predictive features

> [!TIP]
> ***After scaling and selecting K, use cross-validation to evaluate model performance***. If performance is still poor, the issue may be feature relevance, not scaling.

---

## Best Practices Summary

✅ **DO:**
- ***Standardize or robustly scale*** all numerical features
- ***Fit transformers on training data only***
- ***Apply transformations consistently*** across train/val/test
- ***Use a pipeline*** to automate correct ordering
- ***Handle categorical variables*** before scaling
- ***Impute missing values*** before scaling
- ***Remove constant features*** before scaling

❌ **DON'T:**
- Scale the entire dataset before splitting (data leakage)
- Forget to transform test data with training statistics
- Mix scaled and unscaled features in KNN
- Use min-max scaling when outliers are present
- Assume scaling solves all performance issues
- Scale the target variable (unless specified)

---

<p align="center">
  ⬅️ <strong>Previous:</strong> <a href="./04.4%20KNN%20Regressor%20Using%20Sickit-learn.ipynb">04.4 KNN Regressor Using Scikit-learn</a> &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Next:</strong> <a href="./06%20Numerical%20Example%20on%20KNN.md">06 Numerical Example on KNN</a> ➡️
</p>
