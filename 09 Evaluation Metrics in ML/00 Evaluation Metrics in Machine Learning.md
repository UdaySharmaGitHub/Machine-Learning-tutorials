# Evaluation Metrics in Machine Learning

Evaluation metrics are used to measure how well a machine learning model performs. They help assess whether the model is making accurate predictions and meeting the desired goals. This is important because:

- **Model Performance** — Measures how well the model works
- **Different Tasks** — Used for classification, regression, and clustering
- **Right Metric Choice** — Helps select the best way to evaluate a model
- **Better Decisions** — Ensures the model meets its objectives

---

## Table of Contents

- [Overview: Types of Evaluation Metrics](#overview-types-of-evaluation-metrics)
- [Regression Metrics](#regression-metrics)
  - [What Are Regression Metrics?](#what-are-regression-metrics)
  - [Where Are Regression Metrics Used?](#where-are-regression-metrics-used)
  - [1. Mean Absolute Error (MAE)](#1-mean-absolute-error-mae)
  - [2. Mean Squared Error (MSE)](#2-mean-squared-error-mse)
  - [3. Root Mean Squared Error (RMSE)](#3-root-mean-squared-error-rmse)
  - [4. R-squared (R²)](#4-r-squared-r)
  - [5. Adjusted R-squared (Adjusted R²)](#5-adjusted-r-squared-adjusted-r)
- [Classification Metrics](#classification-metrics)
  - [1. Accuracy](#1-accuracy)
  - [2. Precision](#2-precision)
  - [3. Recall (Sensitivity)](#3-recall-sensitivity)
  - [4. F1 Score](#4-f1-score)
  - [5. Logarithmic Loss (Log Loss)](#5-logarithmic-loss-log-loss)
  - [6. Area Under Curve (AUC) and ROC Curve](#6-area-under-curve-auc-and-roc-curve)
  - [7. Confusion Matrix](#7-confusion-matrix)
    - [Multiclass Confusion Matrix](#multiclass-confusion-matrix)
- [Clustering Metrics](#clustering-metrics)
  - [1. Silhouette Score](#1-silhouette-score)
  - [2. Davies-Bouldin Index](#2-davies-bouldin-index)

---

## Overview: Types of Evaluation Metrics

```mermaid
graph TD
    EM["Evaluation Metrics\nin Machine Learning"]

    EM --> RM["Regression Metrics\nPredicts continuous values"]
    EM --> CM["Classification Metrics\nPredicts discrete categories"]
    EM --> CLM["Clustering Metrics\nGroups similar data points"]

    RM --> MAE["MAE\nMean Absolute Error"]
    RM --> MSE["MSE\nMean Squared Error"]
    RM --> RMSE["RMSE\nRoot Mean Squared Error"]
    RM --> R2["R²\nR-squared"]
    RM --> AR2["Adjusted R²"]

    CM --> ACC["Accuracy"]
    CM --> PREC["Precision"]
    CM --> REC["Recall"]
    CM --> F1["F1 Score"]
    CM --> LL["Log Loss"]
    CM --> ROC["AUC-ROC Curve"]
    CM --> CONF["Confusion Matrix"]

    CLM --> SS["Silhouette Score"]
    CLM --> DBI["Davies-Bouldin Index"]

    style EM fill:#4A90D9,color:#fff,stroke:#2C5F8A
    style RM fill:#27AE60,color:#fff,stroke:#1A7A42
    style CM fill:#E67E22,color:#fff,stroke:#A85A0F
    style CLM fill:#8E44AD,color:#fff,stroke:#5D2D73
```

---

## Regression Metrics

### What Are Regression Metrics?

Regression metrics are numerical measures used to evaluate how well a regression model predicts **continuous target values**. Unlike classification tasks (which predict discrete labels like "spam / not spam"), regression tasks predict quantities — such as house prices, student salaries, temperatures, or delivery times — and regression metrics tell us **how far off the predictions are from the true values**.

The **residual (error)** for a single prediction is:

$$e_i = y_i - \hat{y}_i$$

- Positive residual → model **under-predicted**
- Negative residual → model **over-predicted**

Regression metrics aggregate these residuals in different ways. Some measure the **size of errors** (MAE, MSE, RMSE), others measure **how much variation the model explains** (R², Adjusted R²). Using multiple metrics together gives the most complete picture.

---

### Regression Metrics Map

```mermaid
graph TD
    RM["Regression Metrics"]

    RM --> EB["Error-Based Metrics\nMeasure size of prediction errors"]
    RM --> GF["Goodness-of-Fit Metrics\nMeasure explained variance"]

    EB --> MAE["MAE\nMean Absolute Error\nSame unit as target\nLower is better"]
    EB --> MSE["MSE\nMean Squared Error\nSquared unit\nLower is better"]
    EB --> RMSE["RMSE\nRoot Mean Squared Error\nSame unit as target\nLower is better"]

    GF --> R2["R²  R-squared\nProportion of variance explained\n0 to 1, Higher is better"]
    GF --> AR2["Adjusted R²\nR² penalized for\nirrelevant features\nHigher is better"]

    style RM fill:#27AE60,color:#fff,stroke:#1A7A42
    style EB fill:#2ECC71,color:#fff,stroke:#1A7A42
    style GF fill:#1ABC9C,color:#fff,stroke:#0E8A70
```

---

### Where Are Regression Metrics Used?

Regression metrics are applied in any domain where a model predicts a **numerical output**:

| Domain | Example Target Variable | Common Metrics |
|:---|:---|:---|
| **Recruitment / HR** | Salary / placement package prediction | MAE, Adjusted R² |
| **Real Estate** | House price prediction | MAE, RMSE |
| **Finance** | Stock price, loan amount forecasting | RMSE, R² |
| **Healthcare** | Patient recovery time, drug dosage | MAE, RMSE |
| **E-Commerce** | Demand forecasting, delivery time | MAE, RMSLE |
| **Energy** | Power consumption forecasting | RMSE, MAPE |
| **Climate Science** | Temperature, rainfall forecasting | RMSE, R² |

---

### Metrics Overview

| # | Metric | What It Measures | Better When |
|:---:|:---|:---|:---:|
| 1 | **MAE** | Average absolute error (same unit as target) | Lower ↓ |
| 2 | **MSE** | Average squared error (penalizes large errors heavily) | Lower ↓ |
| 3 | **RMSE** | Square root of MSE (back to original units) | Lower ↓ |
| 4 | **R²** | Proportion of target variance explained by model | Higher ↑ (max = 1) |
| 5 | **Adjusted R²** | R² penalized for adding irrelevant features | Higher ↑ (max = 1) |

---

### 1. Mean Absolute Error (MAE)

**Formula:**

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**What it measures:**
MAE is the average size of the errors, ignoring direction. Every error — large or small — is treated **equally**.

**How to interpret:**
An MAE of 0.50 LPA means: *"On average, the model's package prediction is off by ₹0.50 LPA."* This is directly human-readable because it is in the same unit as the target.

**When to use MAE:**
- When all errors are equally important (no preference for penalizing outliers)
- When you need a metric that non-technical stakeholders can understand
- When the data has outliers and you don't want them to dominate the score

| Pros | Cons |
|:---|:---|
| Easy to interpret — same unit as target | Treats small and large errors equally |
| Robust to outliers | Not differentiable at zero (optimization issue) |
| Directly meaningful in business context | Does not tell you if errors are systematic |

---

### 2. Mean Squared Error (MSE)

**Formula:**

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**What it measures:**
The average of the **squared** errors. Squaring ensures large errors contribute far more than small ones.

**How to interpret:**
An MSE of 0.12 LPA² means the average squared error is 0.12. The unit is squared and less intuitive — MSE is most useful for comparing models or as a training loss, not for direct communication.

**When to use MSE:**
- As a **loss function during model training** (differentiable and smooth)
- When large prediction errors are especially unacceptable (safety-critical systems)
- When comparing models mathematically

| Pros | Cons |
|:---|:---|
| Penalizes large errors heavily | Units are squared — not directly interpretable |
| Differentiable everywhere — ideal for gradient-based optimization | Sensitive to outliers |
| Widely used as a training loss | Direction of errors is lost |

---

### 3. Root Mean Squared Error (RMSE)

**Formula:**

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**What it measures:**
The square root of MSE — brings the metric back to the **same unit as the target** while still penalizing large errors more than MAE does.

**How to interpret:**
An RMSE of 0.35 LPA means: *"Typical predictions deviate from the true value by about ₹0.35 LPA, with large errors weighted more."*

> **Tip:** If `RMSE >> MAE`, the model is making a few **very large errors** (outliers dominating). If `RMSE ≈ MAE`, errors are consistent in size.

**When to use RMSE:**
- When large errors are more costly than small ones
- As a general-purpose metric when you want interpretability and outlier sensitivity
- When comparing models on the same dataset

| Pros | Cons |
|:---|:---|
| Same unit as target — interpretable | More sensitive to outliers than MAE |
| Penalizes large errors more heavily | Can be misleading if outliers are not genuine |
| Standard metric for regression benchmarks | Direction of bias is lost |

---

### 4. R-squared (R²)

**Formula:**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ = Sum of Squared Residuals (unexplained variance)
- $SS_{tot}$ = Total Sum of Squares (total variance in the target)
- $\bar{y}$ = Mean of actual values

**What it measures:**
The proportion of variance in the target variable **explained by the model**. R² answers: *"How much better is our model compared to just predicting the mean every time?"*

**How to interpret:**

| R² Value | Interpretation |
|:---:|:---|
| 1.0 | Perfect predictions — model explains all variance |
| 0.78 | Model explains 78% of the variance in the target |
| 0.0 | Model is no better than predicting the mean |
| < 0.0 | Model performs **worse** than predicting the mean |

**When to use R²:**
- When you want to explain model quality to stakeholders
- When comparing models trained on the **same target and dataset**
- When you need a scale-free, normalized metric

| Pros | Cons |
|:---|:---|
| Normalized (0 to 1) — easy to compare | Always increases when more features are added, even useless ones |
| Intuitive: "model explains X% of variance" | Does not indicate whether absolute errors are large or small |
| Scale-independent | Can be misleading with too many features |

---

### 5. Adjusted R-squared (Adjusted R²)

**Formula:**

$$Adjusted\ R^2 = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$$

Where:
- $n$ = number of samples
- $k$ = number of features (predictors)

**What it measures:**
A version of R² that **penalizes adding irrelevant features**. Ordinary R² never decreases when features are added — even random noise slightly inflates it. Adjusted R² corrects for this by accounting for the number of predictors.

**How to interpret:**

| Scenario | R² | Adjusted R² |
|:---|:---:|:---:|
| Add a **relevant** feature (e.g., IQ score) | Increases ↑ | Also increases ↑ |
| Add an **irrelevant** feature (e.g., random noise) | Slightly increases ↑ | Decreases ↓ or stays same |

**When to use Adjusted R²:**
- When comparing models with **different numbers of features**
- During **feature selection** — to detect if a new feature actually helps
- Whenever you're building a multi-feature regression model

| Pros | Cons |
|:---|:---|
| Penalizes unnecessary features | Harder to interpret than R² for beginners |
| Better for multi-feature model comparison | Can become negative for very poor models |
| Prevents overfitting via feature complexity | Still does not tell you about absolute error size |

---

## Classification Metrics

Classification problems aim to predict **discrete categories** (e.g., spam / not spam, disease / healthy). To evaluate classification model performance, we use the following metrics:

A classification metric compares the actual class labels ($y$) with the labels predicted by a model ($\hat{y}$). These metrics help us understand not only how many predictions are correct, but also the types of errors the model makes.

The positive class is represented by `1` and the negative class by `0`. We use the following quantities throughout this section:

- **True Positive (TP):** The actual class is positive and the model predicts positive.
- **True Negative (TN):** The actual class is negative and the model predicts negative.
- **False Positive (FP):** The actual class is negative but the model predicts positive. This is a Type I error.
- **False Negative (FN):** The actual class is positive but the model predicts negative. This is a Type II error.

### Classification Metrics Map

```mermaid
graph TD
    CM["Classification Metrics"]

    CM --> TB["Threshold-Based Metrics\nDepend on a fixed 0.5 cutoff"]
    CM --> PB["Probability-Based Metrics\nUse raw predicted probabilities"]
    CM --> VZ["Visual / Matrix Tools"]

    TB --> ACC["Accuracy\nOverall correct predictions"]
    TB --> PREC["Precision\nOf predicted positives,\nhow many are correct?"]
    TB --> REC["Recall\nOf actual positives,\nhow many were found?"]
    TB --> F1["F1 Score\nHarmonic mean of\nPrecision and Recall"]

    PB --> LL["Log Loss\nPenalizes confident\nwrong predictions"]
    PB --> AUC["AUC-ROC\nTPR vs FPR at\nall thresholds"]

    VZ --> CONF["Confusion Matrix\nTP / TN / FP / FN\ncounts table"]

    style CM fill:#E67E22,color:#fff,stroke:#A85A0F
    style TB fill:#F39C12,color:#fff,stroke:#B07D0E
    style PB fill:#E74C3C,color:#fff,stroke:#A53226
    style VZ fill:#C0392B,color:#fff,stroke:#8B2020
```

---

### 1. Accuracy

Accuracy is the proportion of correct predictions out of all predictions made.

$$\boxed{Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}}$$

While accuracy provides a quick snapshot, it can be **misleading for imbalanced datasets**. For example, in a dataset with 90% class A and 10% class B, a model that always predicts class A achieves 90% accuracy but fails to identify any class B instances.

#### When Is Accuracy Misleading?

Accuracy can be misleading when the classes are **imbalanced**, meaning one class has many more observations than the other. In that situation, a model can achieve a high accuracy score by mostly predicting the majority class while failing to identify the minority class.

**Example:** Suppose a dataset contains 100 patients:

- 90 patients do not have the disease.
- 10 patients have the disease.
- The model predicts **"no disease" for every patient**.

The model makes 90 correct predictions, so:

$$
	ext{Accuracy} = \frac{90}{100} = 90\%
$$

However, the model detects none of the 10 patients who have the disease:

- $TP = 0$
- $FN = 10$
- $TN = 90$
- $FP = 0$
- Recall $= \frac{TP}{TP + FN} = \frac{0}{0 + 10} = 0$

Therefore, the model has **90% accuracy but 0% recall** for the positive class. Accuracy alone gives the impression of good performance, while recall reveals that the model is failing at the most important task.

> **Note:** When classes are imbalanced, evaluate the model using the confusion matrix, precision, recall, F1-score, balanced accuracy, or class-specific metrics in addition to accuracy.
> **Use with caution** when class distributions are unequal — combine with Precision, Recall, or F1 for a complete picture.

---

### 2. Precision

Precision measures how many of the **positive predictions** made by the model are actually correct. It is useful when the **cost of false positives is high** (e.g., medical diagnosis, fraud detection).

$$\boxed{Precision = \frac{TP}{TP + FP}}$$

Where:
- $TP$ = True Positives
- $FP$ = False Positives

> High Precision → when the model says "positive", it is usually right.

---

### 3. Recall (Sensitivity)

Recall measures how many of the **actual positive cases** were correctly identified. It is important when **missing a positive case is costly** (e.g., cancer screening, safety systems).

$$\boxed{Recall = \frac{TP}{TP + FN}}$$

Where:
- $TP$ = True Positives
- $FN$ = False Negatives

> High Recall → the model catches most of the actual positive cases.

> **Note — Precision–Recall Trade-off**
>
> Increasing precision often decreases recall, and increasing recall often decreases precision. The best balance depends on which error is more costly:
>
> - Prefer **higher precision** when false positives are more costly.
> - Prefer **higher recall** when false negatives are more costly.

---

### 4. F1 Score

The F1 Score is the **harmonic mean of Precision and Recall**. It gives a ***single number that balances both*** metrics and is especially ***useful when class distributions are uneven.***

$$\boxed{F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}}$$

- Range: $[0, 1]$ — higher is better
- F1 = 1 means perfect Precision and Recall
- F1 penalizes extreme imbalances between Precision and Recall

> Use F1 when you need a **balance between Precision and Recall** and can't sacrifice either.

---

### 5. Logarithmic Loss (Log Loss)

Log Loss measures the **uncertainty of the model's predictions** by penalizing confident wrong predictions heavily. It is used for probabilistic classifiers.

$$\boxed{Log\ Loss = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \cdot \log(p_{ij})}$$

Where:
- $N$ = number of samples
- $M$ = number of classes
- $y_{ij}$ = 1 if sample $i$ belongs to class $j$, else 0
- $p_{ij}$ = predicted probability for sample $i$ and class $j$

> Lower Log Loss is better. A model that is confidently wrong is penalized much more than one that is uncertain.

---

### 6. Area Under Curve (AUC) and ROC Curve

AUC-ROC is used for **binary classification** tasks. The ROC curve plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at different classification thresholds.

#### Key Rates

| Rate | Formula | Meaning |
|:---|:---|:---|
| **True Positive Rate (TPR / Recall)** | $\dfrac{TP}{TP + FN}$ | Of all actual positives, how many were correctly identified? |
| **True Negative Rate (TNR / Specificity)** | $\dfrac{TN}{TN + FP}$ | Of all actual negatives, how many were correctly identified? |
| **False Positive Rate (FPR)** | $\dfrac{FP}{FP + TN}$ | Of all actual negatives, how many were wrongly flagged as positive? |
| **False Negative Rate (FNR)** | $\dfrac{FN}{FN + TP}$ | Of all actual positives, how many were missed? |

#### AUC Interpretation

| AUC Value | Interpretation |
|:---:|:---|
| 1.0 | Perfect model — always ranks positives above negatives |
| 0.5 | No better than random guessing |
| < 0.5 | Worse than random (model predictions are inverted) |

> The ROC curve shows the **trade-off between sensitivity and specificity** across all thresholds. AUC summarizes this as a single number.

---

### 7. Confusion Matrix

A confusion matrix is an $N \times N$ table showing the counts of actual vs predicted classes. For binary classification ($N = 2$), we use the explicit class order `[1, 0]`, with the positive class shown first on both axes:

| | **Predicted: 1** | **Predicted: 0** |
|:---|:---:|:---:|
| **Actual: 1** | TP = 100 | FN = 5 |
| **Actual: 0** | FP = 10 | TN = 50 |

*Example: n = 165 total samples*

The four cells:
- **True Positives (TP)** — Predicted Yes, Actual Yes ✓
- **True Negatives (TN)** — Predicted No, Actual No ✓
- **False Positives (FP)** — Predicted Yes, Actual No ✗ *(Type I error)*
- **False Negatives (FN)** — Predicted No, Actual Yes ✗ *(Type II error)*


#### Type I and Type II Errors

Classification errors occur when the model's prediction does not match the actual class.

- **Type I Error (False Positive):** The actual class is negative, but the model predicts positive. In the confusion matrix, this is $FP$. For example, a healthy person is incorrectly classified as having heart disease. The model raises a false alarm.
- **Type II Error (False Negative):** The actual class is positive, but the model predicts negative. In the confusion matrix, this is $FN$. For example, a person with heart disease is incorrectly classified as healthy. The model fails to detect a positive case.

The error rates can be written as:

$$
\boxed{\text{Type I Error Rate} = \frac{FP}{FP + TN}}
$$

$$
\boxed{\text{Type II Error Rate} = \frac{FN}{FN + TP}}
$$


> - ***Type I and Type II errors usually involve a trade-off***. Making a model more sensitive may reduce Type II errors but can increase Type I errors. The appropriate balance depends on the application. In medical screening, reducing Type II errors is often important because missing a disease can be more serious than raising a false alarm.

- All classification metrics (Accuracy, Precision, Recall, F1) can be derived directly from these four values.

### Multiclass Confusion Matrix

The binary confusion matrix has two classes, but a **multiclass classification** problem has three or more possible classes. For $K$ classes, the confusion matrix has a size of $K \times K$.

- **Rows** represent the actual classes.
- **Columns** represent the predicted classes.
- The ***diagonal cells*** contain ***correct predictions***.
- The ***off-diagonal*** cells contain ***misclassifications***.

For example, a three-class problem with classes A, B, and C can be represented as:

| Actual / Predicted | Predicted A | Predicted B | Predicted C |
|:---|---:|---:|---:|
| **Actual A** | $C_{AA}$ | $C_{AB}$ | $C_{AC}$ |
| **Actual B** | $C_{BA}$ | $C_{BB}$ | $C_{BC}$ |
| **Actual C** | $C_{CA}$ | $C_{CB}$ | $C_{CC}$ |

Here, 
- ***$C_{ij}$*** is the ***number of observations*** whose ****actual class is $i$*** and whose ****predicted class is $j$***. 
- The ***diagonal values $C_{AA}$, $C_{BB}$, and $C_{CC}$*** are ***correct predictions***.

##### Deriving metrics for one class

Multiclass precision, recall, and F1-score can be calculated for each class by treating that class as **positive** and combining all other classes into a single **negative** group. This is called the **one-vs-rest** approach.

For class $i$:

$$
TP_i = C_{ii}
$$

$$
FN_i = \sum_{j \ne i} C_{ij}
$$

$$
FP_i = \sum_{j \ne i} C_{ji}
$$

$$
TN_i = \sum_{j \ne i} \sum_{k \ne i} C_{jk}
$$

Therefore, the per-class metrics are:

$$
\mathrm{Precision}_i = \frac{TP_i}{TP_i + FP_i}
$$

$$
\mathrm{Recall}_i = \frac{TP_i}{TP_i + FN_i}
$$

$$
F1_i = \frac{2 \times \text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
$$

#### Macro and micro averaging for precision and recall

In multiclass classification, precision and recall are first calculated for each class. The per-class scores can then be combined using **macro averaging** or **micro averaging**.

**Macro-averaged precision:** Calculate precision for every class and give every class equal importance.

$$
\boxed{\mathrm{Precision}_{\mathrm{macro}} = \frac{1}{K} \sum_{i=1}^{K} \mathrm{Precision}_i}
$$

Substituting the per-class formula:

$$
\mathrm{Precision}_{\mathrm{macro}} = \frac{1}{K} \sum_{i=1}^{K} \frac{TP_i}{TP_i + FP_i}
$$

**Macro-averaged recall:** Calculate recall for every class and give every class equal importance.

$$
\boxed{\mathrm{Recall}_{\mathrm{macro}} = \frac{1}{K} \sum_{i=1}^{K} \mathrm{Recall}_i}
$$

Substituting the per-class formula:

$$
\mathrm{Recall}_{\mathrm{macro}} = \frac{1}{K} \sum_{i=1}^{K} \frac{TP_i}{TP_i + FN_i}
$$

Macro averaging is useful when every class matters equally, especially when minority-class performance should not be hidden by a majority class.

**Micro-averaged precision:** Add the true positives and false positives across all classes before calculating precision.

$$
\boxed{\mathrm{Precision}_{\mathrm{micro}} = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} TP_i + \sum_{i=1}^{K} FP_i}}
$$

**Micro-averaged recall:** Add the true positives and false negatives across all classes before calculating recall.

$$
\boxed{\mathrm{Recall}_{\mathrm{micro}} = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} TP_i + \sum_{i=1}^{K} FN_i}}
$$

Micro averaging gives every individual observation equal importance. For a single-label multiclass classification problem, each prediction belongs to exactly one class, so:

$$
\mathrm{Precision}_{\mathrm{micro}} = \mathrm{Recall}_{\mathrm{micro}} = \mathrm{Accuracy}
$$

Use **macro averaging** when class-level fairness is important. Use **micro averaging** when overall prediction performance is the main objective.

##### Averaging metrics across classes

Because a multiclass model has one score for each class, the scores can be combined in different ways:

| Averaging method | Definition | When to use |
|:---|:---|:---|
| **Macro average** | Calculate the metric for each class, then take the unweighted mean | Every class should have equal importance, including minority classes |
| **Weighted average** | Calculate the metric for each class, then weight it by that class's support (number of actual observations) | Class sizes are unequal and larger classes should have proportionally greater influence |
| **Micro average** | Add all class-level $TP$, $FP$, and $FN$ counts first, then calculate one global metric | Overall performance across all individual predictions is important |

For $K$ classes, the macro average of a metric $M$ is:

$$
M_{macro} = \frac{1}{K} \sum_{i=1}^{K} M_i
$$

The weighted average is:

$$
M_{weighted} = \frac{\sum_{i=1}^{K} n_i M_i}{\sum_{i=1}^{K} n_i}
$$

where $n_i$ is the number of actual observations in class $i$. In scikit-learn, these choices are available through the `average` parameter, such as `average='macro'`, `average='weighted'`, or `average='micro'`.

Multiclass accuracy is still calculated as the number of correct predictions divided by the total number of predictions:

$$
	ext{Accuracy} = \frac{\sum_{i=1}^{K} C_{ii}}{\sum_{i=1}^{K}\sum_{j=1}^{K} C_{ij}}
$$

The binary formulas are therefore not discarded for multiclass classification; they are applied one class at a time and then averaged according to the evaluation objective.

---

## Clustering Metrics

In unsupervised learning, the goal is to group similar data points together without ground-truth labels. Evaluating clustering is more challenging than supervised learning since there is no explicit "correct answer." The following metrics measure cluster quality internally.

### Clustering Metrics Map

```mermaid
graph TD
    CLM["Clustering Metrics"]

    CLM --> INT["Internal Metrics\nNo ground truth labels required"]

    INT --> SS["Silhouette Score\nMeasures cohesion vs separation\nRange: -1 to +1\nHigher is better"]
    INT --> DBI["Davies-Bouldin Index\nMeasures average cluster similarity\nRange: 0 to infinity\nLower is better"]

    SS --> SSI["Score near +1\nWell-clustered point"]
    SS --> SSB["Score near 0\nPoint on cluster boundary"]
    SS --> SSN["Score near -1\nLikely in wrong cluster"]

    DBI --> DBIL["Lower value\nCompact and well-separated clusters"]
    DBI --> DBIH["Higher value\nOverlapping or poorly defined clusters"]

    style CLM fill:#8E44AD,color:#fff,stroke:#5D2D73
    style INT fill:#9B59B6,color:#fff,stroke:#6C3483
```

---

### 1. Silhouette Score

The Silhouette Score evaluates how well each data point fits within its assigned cluster — balancing **cohesion** (closeness to its own cluster) and **separation** (distance from other clusters).

$$Silhouette\ Score = \frac{b - a}{\max(a, b)}$$

Where:
- $a$ = average distance between a point and all other points **in the same cluster**
- $b$ = average distance between a point and all points **in the nearest other cluster**

| Score | Interpretation |
|:---:|:---|
| Close to +1 | Well-clustered — point is far from neighbouring clusters |
| Around 0 | Point is on the boundary between two clusters |
| Close to −1 | Likely misclassified — point is closer to another cluster |

> Higher Silhouette Score is better. Use it to compare different values of $k$ in K-Means.

---

### 2. Davies-Bouldin Index

The Davies-Bouldin Index measures the **average similarity** between each cluster and its most similar neighbouring cluster. It combines cluster scatter (compactness) and inter-cluster distance.

$$Davies\text{-}Bouldin\ Index = \frac{1}{N} \sum_{i=1}^{N} \max_{i \neq j} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

Where:
- $\sigma_i$ = average distance of points in cluster $i$ from its centroid (scatter)
- $d(c_i, c_j)$ = distance between centroids of clusters $i$ and $j$

| Index Value | Interpretation |
|:---:|:---|
| Lower | Better clustering — clusters are compact and well-separated |
| Higher | Worse clustering — clusters overlap or are poorly defined |

> Lower Davies-Bouldin Index is better. Unlike Silhouette Score, it does not require computing distances to all points.
