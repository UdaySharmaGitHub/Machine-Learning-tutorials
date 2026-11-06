# ROC Curve & AUC: Important Concepts You Should Know

## Table of Contents

- [ROC-AUC Does Not Measure Probability Calibration](#1-roc-auc-does-not-measure-probability-calibration)
- [ROC-AUC Is Independent of Classification Threshold](#2-roc-auc-is-independent-of-classification-threshold)
- [ROC-AUC Is Invariant to Monotonic Transformations](#3-roc-auc-is-invariant-to-monotonic-transformations)
- [ROC-AUC and Class Imbalance](#4-roc-auc-and-class-imbalance)
- [ROC Curve vs. Precision-Recall Curve](#5-roc-curve-vs-precision-recall-curve-pr-curve)
- [AUC Can Be Misleading When Comparing Models](#6-auc-can-be-misleading-when-comparing-models)
  - [What Is a Confidence Interval?](#what-is-a-confidence-interval)
- [What Happens When AUC Is Below 0.5?](#7-what-happens-when-auc-is-below-05)
- [ROC-AUC for Multiclass Classification](#8-roc-auc-for-multiclass-classification)
  - [One-vs-Rest (OvR)](#a-one-vs-rest-ovr)
  - [One-vs-One (OvO)](#b-one-vs-one-ovo)
- [Use Prediction Scores, Not Just Predicted Labels, for ROC-AUC](#9-use-prediction-scores-not-just-predicted-labels-for-roc-auc)
- [Partial AUC (pAUC) and Operating Regions](#10-partial-auc-pauc-and-operating-regions)
- [ROC-AUC Does Not Guarantee Good Production Performance](#11-roc-auc-does-not-guarantee-good-production-performance)
- [Complete ROC & AUC Learning Checklist](#12-complete-roc--auc-learning-checklist)
  - [Fundamentals](#fundamentals)
  - [AUC Concepts](#auc-concepts)
  - [Threshold Optimization](#threshold-optimization)
  - [Advanced Concepts](#advanced-concepts)
  - [Practical Implementation](#practical-implementation)
- [What Should You Study Next?](#13-what-should-you-study-next)

## 1. ROC-AUC Does Not Measure Probability Calibration

> [!WARNING]
> A **high ROC-AUC** does **not** mean that a model has **well-calibrated probabilities**. This is one of the most common ROC-AUC misunderstandings.

Logistic Regression predicts probabilities. For example:

| Customer | Actual class | Model A | Model B |
|:---|:---|---:|---:|
| A | Positive | 0.90 | 0.99 |
| B | Positive | 0.80 | 0.95 |
| C | Negative | 0.30 | 0.40 |
| D | Negative | 0.10 | 0.20 |

- Both models rank positive examples above negative examples.
- Therefore, both have perfect **AUC = 1.0**.
- Their predicted probabilities are nevertheless different.

**Probability calibration** measures whether predicted probabilities correspond to actual observed frequencies. For example:

- If a model predicts that 100 customers each have a 70% probability of churning, approximately 70 should actually churn for the predictions to be well calibrated.
- ***ROC-AUC does not measure calibration.***

| ROC-AUC | Calibration |
|:---|:---|
| Measures ranking ability | Measures probability reliability |
| Compares positive scores with negative scores | Compares predicted probabilities with actual frequencies |
| Evaluated using the ROC curve | Often evaluated using calibration curves and the Brier score |

> [!IMPORTANT]
> A model can have **excellent AUC** but **poorly calibrated probabilities**. Check **calibration separately** when probability values drive financial decisions, risk estimates, or business planning.

## 2. ROC-AUC Is Independent of Classification Threshold

> [!IMPORTANT]
> **ROC-AUC evaluates ranking across thresholds**; it does not depend on choosing one **classification threshold**.

Suppose Logistic Regression generates:

```text
Customer A: 0.95
Customer B: 0.85
Customer C: 0.65
Customer D: 0.40
Customer E: 0.15
```

You might use:

- Threshold = 0.5
- Threshold = 0.7
- Threshold = 0.9

Changing the threshold changes the final predicted classes, but the underlying probability scores remain the same. ROC-AUC evaluates the ranking of those scores, so changing the classification threshold does not change ROC-AUC.

> [!IMPORTANT]
> **Threshold and AUC answer different questions:**
>
> - The **threshold** determines one operating point and its final class predictions.
> - **ROC-AUC** evaluates overall ranking performance across thresholds.

## 3. ROC-AUC Is Invariant to Monotonic Transformations

> [!NOTE]
> **Key property:** AUC is **invariant to strictly increasing transformations** because they preserve **score ordering**.

Suppose your model produces these scores:

| Customer | Original score |
|:---|---:|
| A | 0.90 |
| B | 0.80 |
| C | 0.60 |
| D | 0.30 |

Now apply a strictly increasing transformation, such as squaring the scores.

| Customer | Original score | Squared score |
|:---|---:|---:|
| A | 0.90 | 0.81 |
| B | 0.80 | 0.64 |
| C | 0.60 | 0.36 |
| D | 0.30 | 0.09 |

- The scores change, but their ranking does not.
- The order remains:

$$A > B > C > D$$

- Therefore, ROC-AUC remains unchanged.
- AUC measures ranking, not the numerical values of the probabilities.
- As a result, AUC alone cannot tell us whether predicted probabilities are well calibrated.

## 4. ROC-AUC and Class Imbalance

> [!NOTE]
> **Class imbalance** is common in real-world machine learning, especially in **fraud detection**, **medical screening**, and **anomaly detection**.

Let's understand this using fraud detection.

Imagine you have 100,000 transactions:

- 99,000 legitimate transactions.
- 1,000 fraudulent transactions.

ROC-AUC is generally less sensitive to class imbalance than accuracy because each rate is calculated within its actual class:

$$
\text{TPR} = \frac{TP}{TP + FN}
$$

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

> [!WARNING]
> Even a **small FPR** can generate **many false positives** when the negative class is very large. Always translate rates into **expected case counts** for the deployment population.

For example:

- 99,000 legitimate transactions.
- FPR = 1%.

The number of false positives is:

$$
FP = 99{,}000 \times 0.01 = 990
$$

If there are:

- 1,000 actual fraudulent transactions.
- TPR = 90%.

then the number of true positives is:

$$
TP = 1{,}000 \times 0.90 = 900
$$

- The system detects 900 fraudulent transactions.
- It also flags 990 legitimate transactions, creating many false alarms.

> [!IMPORTANT]
> **ROC-AUC alone** may not tell the complete story for **highly imbalanced datasets**. Also examine **precision**, **recall**, **PR-AUC**, and the **absolute number of false positives**.

## 5. ROC Curve vs. Precision-Recall Curve (PR Curve)

> [!TIP]
> Use the **Precision-Recall (PR) curve** alongside **ROC-AUC** when the positive class is rare or when the quality of predicted-positive cases is especially important.

The Precision-Recall curve evaluates:

- X-axis: Recall (TPR).
- Y-axis: Precision.

**Recall:**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Precision:**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

In contrast to ROC's TPR and FPR, the PR curve shows the relationship between precision and recall.

| ROC curve | Precision-Recall curve |
|:---|:---|
| X-axis: FPR | X-axis: Recall |
| Y-axis: TPR | Y-axis: Precision |
| Useful for general classification evaluation | Particularly informative for imbalanced positive classes |
| Considers false positives relative to actual negatives | Directly reflects how many predicted positives are correct |

**When is a PR curve useful?**

In fraud detection, for example, you might want to know:

> [!NOTE]
> **Precision question:** If my system flags 100 transactions as fraudulent, how many of those are actually fraudulent?

- That is a precision question.
- ***PR-AUC and ROC-AUC are different metrics.***
- PR-AUC is sensitive to positive-class prevalence, so its baseline depends on the positive-class proportion.

## 6. AUC Can Be Misleading When Comparing Models

Suppose you train two classification models:

| Model | ROC-AUC |
|:---|---:|
| Logistic Regression | 0.91 |
| Random Forest | 0.92 |

We cannot necessarily conclude that the Random Forest is better just because its AUC is 0.01 higher; the difference may be due to random variation in the test dataset.

> [!WARNING]
> A **small AUC difference** does not by itself prove that one model is better. **Quantify uncertainty** and use an appropriate **paired comparison** when models are evaluated on the same cases.

To understand whether the difference is statistically meaningful, we can use:

- Confidence intervals.
- DeLong's test for comparing correlated ROC-AUC values.
- Bootstrap resampling.

### What Is a Confidence Interval?

Suppose your model has:

$$\text{AUC} = 0.90$$

But its 95% confidence interval is:

$$[0.86, 0.94]$$

- The interval indicates uncertainty around the estimated AUC.
- Overlapping confidence intervals alone do not prove that the difference between two models is statistically insignificant. A paired comparison test can help.
- This helps avoid choosing a model based only on a small advantage in one test sample.

## 7. What Happens When AUC Is Below 0.5?

Use these reference points:

- AUC = 1.0 → Perfect ranking.
- AUC = 0.5 → Random ranking.
- AUC below 0.5 → Ranking is worse than random.

For example:

$$\text{AUC} = 0.30$$

- This may indicate that the model consistently ranks negative examples above positive examples.

If we reverse the prediction scores:

$$
\text{AUC}_{\text{reversed}} = 1 - 0.30 = 0.70
$$

> [!CAUTION]
> Do **not automatically reverse a model** just because its **AUC is below 0.5**. First verify the **positive label**, selected **score or probability column**, and expected **score direction**.

Reversing the scores reverses the ranking direction. Before doing so, check:

- Whether positive and negative labels are correct.
- Whether the correct probability column was selected.
- Whether the model's score direction is appropriate.

This checklist is useful when debugging classification models.

## 8. ROC-AUC for Multiclass Classification

So far, we've discussed binary classification:

- Positive (1)
- Negative (0)

But what if you want to classify images into three categories?

For example:

- Cat
- Dog
- Horse

**Can ROC-AUC be used? Yes.** There are two common approaches:

### A. One-vs-Rest (OvR)

We evaluate each class against all the remaining classes.

| Class | Positive | Negative |
|:---|:---|:---|
| Cat | Cat | Dog + Horse |
| Dog | Dog | Cat + Horse |
| Horse | Horse | Cat + Dog |

We calculate ROC-AUC separately for each class.

Then we can calculate an overall score using:

- Macro averaging: Calculate each class's AUC and take their unweighted average.
- Weighted averaging: Average class AUCs using their support (number of actual samples) as weights.

### B. One-vs-One (OvO)

We compare every pair of classes.

For three classes:

- Cat vs Dog.
- Cat vs Horse.
- Dog vs Horse.

We calculate the pairwise AUC values and aggregate them.

This is useful for multiclass classification problems.

## 9. Use Prediction Scores, Not Just Predicted Labels, for ROC-AUC

> [!WARNING]
> **Critical implementation detail:** ROC-AUC requires **continuous prediction scores**, not only **hard class labels**. Hard labels provide only one operating point and cannot produce a complete ROC curve.

Consider:

```python
from sklearn.metrics import roc_auc_score

# Use class labels for classification, but score probabilities for ROC-AUC.
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
```

- `predict()` generally returns final class labels (0 or 1).
- The ROC curve needs continuous scores so it can evaluate different thresholds.
- For SVMs, `decision_function()` can provide scores instead of probabilities, provided higher scores indicate stronger evidence for the positive class.
- ***Ties:*** AUC conventionally gives half credit when a positive and negative example receive exactly the same score.

## 10. Partial AUC (pAUC) and Operating Regions

Partial AUC is useful when evaluation focuses on a particular range of FPR, for example:

$$0 \leq \text{FPR} \leq 0.01$$

This evaluates the classifier only where the false positive rate is at most 1%. Partial AUC compares models in that region instead of across the entire ROC curve.

This is useful for:

- Fraud detection.
- Medical screening.
- Security threat detection.
- Applications where false alarms are particularly expensive.

## 11. ROC-AUC Does Not Guarantee Good Production Performance

> [!WARNING]
> **High ROC-AUC does not guarantee good production performance.** **Prevalence**, **threshold choice**, **calibration**, and **data drift** can all change operational results.

For example, a model may achieve:

$$\text{AUC} = 0.95$$

Although this indicates excellent ranking performance, deployment may reveal issues such as:

- Your positive class is much rarer than in training.
- Your input data distribution has changed.
- Your selected threshold produces too many false alarms.
- The predicted probabilities are poorly calibrated.

Production performance may still be disappointing. Two related concepts are:

- **Data distribution shift:** The characteristics of incoming data differ from those of the training data.
- **Concept drift:** The relationship between input features and the target changes over time.

Therefore, ROC-AUC should be monitored on relevant, representative evaluation data, and periodically reassessed in production.

## 12. Complete ROC & AUC Learning Checklist

Use this checklist to organize your ROC and AUC study:

### Fundamentals

- Confusion matrix: TP, TN, FP, FN
- True Positive Rate (TPR) and False Positive Rate (FPR)
- Sensitivity, specificity, and recall
- ROC space and line of no discrimination
- ROC curve and classification thresholds

### AUC concepts

- Area Under the Curve (AUC)
- Ranking interpretation of AUC
- AUC = 0.5, 1.0, and below 0.5
- AUC vs. accuracy
- AUC vs. threshold

### Threshold optimization

- Minimum sensitivity requirement
- Maximum FPR constraint
- Expected cost minimization
- Youden's J statistic
- Closest point to the top-left corner
- Partial AUC

### Advanced concepts

- ROC-AUC vs. probability calibration
- ROC-AUC invariance to monotonic transformations
- ROC-AUC and class imbalance
- ROC curve vs. Precision-Recall curve
- Confidence intervals and DeLong's test
- Multiclass ROC-AUC: OvR and OvO
- Handling tied prediction scores
- Data distribution shift and concept drift

### Practical implementation

- `roc_curve()` in Scikit-learn
- `roc_auc_score()` in Scikit-learn
- `predict_proba()` vs. `predict()`
- Threshold selection on validation data
- Evaluation on an independent test set

## 13. What Should You Study Next?

I recommend following this sequence:

1. **Precision-Recall curve (PR curve):** Understand why it is particularly useful for imbalanced datasets.
2. **F1 score and F-beta score:** Understand how precision and recall are combined.
3. **Log loss (cross-entropy loss):** Understand how we evaluate the quality of predicted probabilities.
4. **Brier score and calibration curve:** Understand probability calibration.
5. **Confusion matrix and classification metrics:** Connect all the evaluation metrics.

> [!TIP]
> Do not just memorise **ROC-AUC formulas**. Understand **what each metric measures**, **what it ignores**, and **which real-world situation** makes it useful.

That is what will help you move from understanding individual ML algorithms to understanding how to evaluate and deploy machine learning models.