# Multiclass Logistic Regression - Softmax Regression

Logistic Regression is not limited to two classes. This module builds upwards from the binary case through three distinct layers:

1. **Theory** — the Softmax function converts raw logit scores into a valid probability distribution over $K$ classes. Multinomial Logistic Regression (Softmax Regression) is the natural generalisation of the binary model, with the same gradient structure and the same cross-entropy loss.
2. **Non-linear boundaries** — when classes are not linearly separable, polynomial feature expansion transforms a straight-line boundary into a curve. The classic example — concentric circles — is fully solved by adding degree-2 terms ($x_1^2 + x_2^2$) that directly encode the circle equation.
3. **Evaluation** — accuracy alone does not tell the full story for multiclass problems. ROC-AUC with One-vs-Rest (OvR) and One-vs-One (OvO) strategies, Youden's J threshold selection, Precision-Recall AUC, Gini coefficient, and confidence intervals give a complete picture.

---

## Folder Structure

| # | Type | File | Description |
|---|---|---|---|
| 01 | 📝 Notes | [Softmax Function and its mathematical derivation](01%20Softmax%20Function%20and%20its%20mathematical%20derivation.md) | Softmax definition, properties, logits, binary vs. multiclass comparison, matrix form, and full derivation from log-odds |
| 02 | 📝 Notes | [Multinomial Logistic Regression - Softmax Regression](02%20Multinomial%20Logistic%20Regression%20-%20Softmax%20Regression.md) | Comprehensive theory: Softmax pipeline, cross-entropy loss, gradients, regularization, decision boundaries, and temperature scaling |
| 02 | 📓 Notebook | [Multinomial Logistic Regression - Softmax Regression](02%20Multinomial%20Logistic%20Regression%20-%20Softmax%20Regression.ipynb) | Notebook companion for the Softmax Regression theory notes |
| 03 | 📓 Notebook | [Softmax Regression on Iris Dataset](03%20Multinomial%20Logistic%20Regression%20-%20Softmax%20Regression%20Implementation%20on%20Iris%20Dataset.ipynb) | Full Softmax Regression implementation applied to the Iris dataset (3-class classification) |
| 04 | 📓 Notebook | [Polynomial Logistic Regression](04%20polynomial-logistic-regression.ipynb) | Polynomial feature expansion for non-linear decision boundaries on 2D synthetic data |
| 05 | 📓 Notebook | [Polynomial Logistic Regression on Non-Linear (Circles) Data](05%20polynomial-logistic-regression-on-non-linear-data-circles.ipynb) | Why linear boundaries fail on circular data; 48% → 99.88% accuracy with degree-3 polynomial expansion; confusion matrix and probability contour boundary |
| 06 | 📓 Notebook | [Logistic Hyperparameter Tuning](06%20Logistic%20Hyper-parameter.ipynb) | Hyperparameter tuning for logistic regression: regularisation strength, solver, and `max_iter` |
| 07 | 📝 Notes | [Multiclass ROC-AUC (OvR and OvO)](07%20Multiclass%20ROC-AUC%20%28OvR%20and%20OvO%29.md) | Multiclass ROC-AUC with OvR and OvO strategies, threshold selection (Youden's J, EER, cost-sensitive), Precision-Recall AUC, confidence intervals, and Gini coefficient |

---

## Reference Documentation

The following documents from this repository cover the foundational mathematics and statistics used throughout this module.

### Mathematics of Machine Learning

| Concept | Reference | Used in this module |
|---|---|---|
| Softmax Function | [Softmax Function](../Mathematics%20of%20Machine%20Learning/Softmax%20Function/README.md) | §1 — converting logits to probabilities (files 01, 02) |
| Sigmoid Function | [Sigmoid Function](../Mathematics%20of%20Machine%20Learning/Sigmoid%20Function/README.md) | Binary baseline; OvR reduces multiclass to binary sigmoid problems |
| Conditional Probability | [Conditional Probability](../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/README.md) | $P(\text{class} \mid \mathbf{x})$ — the probabilistic output of the model |
| Bayes' Theorem | [Bayes' Theorem](../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/README.md) | Posterior interpretation: $P(k \mid \mathbf{x}) \propto P(\mathbf{x} \mid k)\,P(k)$ |

### Statistics for Machine Learning

| Concept | Reference | Used in this module |
|---|---|---|
| Sensitivity & Specificity | [Sensitivity and Specificity](../Statistics%20for%20Machine%20Learning/Sensitivity%20and%20specificity/Sensitivity%20and%20specificity.md) | §7 — TPR = Sensitivity, Specificity = 1 − FPR in ROC analysis (file 07) |

--- 
## Topics Covered

### Softmax Function and Multinomial Logistic Regression
- Definition: $\sigma(\mathbf{z})_k = e^{z_k} / \sum_j e^{z_j}$ and its three key properties
- What logits are and how they differ from probabilities
- Binary vs. multiclass classification: Sigmoid vs. Softmax
- Mathematical derivation from linear log-odds (reference class → normalization → symmetric form)
- Sigmoid as a special case of Softmax when $K = 2$
- Matrix form of the forward pass: $Z = X_b W$, row-wise Softmax
- Categorical cross-entropy loss and its gradient $\nabla_{W_k}\mathcal{L} = \frac{1}{N}X^\top(\hat{P}_k - Y_k)$
- L1 / L2 regularisation and the `C = 1/λ` convention in scikit-learn
- Decision boundaries and temperature scaling

### Non-Linear Decision Boundaries with Polynomial Features
- Why a linear boundary completely fails on concentric circle data (accuracy ≈ 48%)
- Polynomial feature expansion: `PolynomialFeatures(degree=d)` and the feature count formula $\binom{p+d}{d} - 1$
- Why degree-2 terms $x_1^2 + x_2^2$ directly capture the circle equation $x_1^2 + x_2^2 = r^2$
- Preventing data leakage: `make_pipeline(PolynomialFeatures, LogisticRegression)`
- Accuracy jump from ≈ 48% → ≈ 99.88% with degree-3 expansion
- Probability contour decision boundary visualisation

### Hyperparameter Tuning
- Regularisation strength (`C`), solver selection, and `max_iter`
- Grid search and learning curves for logistic models

### Multiclass Model Evaluation
- ROC-AUC with **One-vs-Rest (OvR)**: macro, weighted, and micro averaging
- ROC-AUC with **One-vs-One (OvO)**: Hand-Till (2001) macro and weighted formulations
- Threshold selection: Youden's J, Equal-Error Rate (EER), cost-sensitive thresholds
- Average Precision (AP) and mean Average Precision (mAP)
- Gini coefficient ($= 2 \cdot \text{AUC} - 1$) and its use in credit scoring
- Confidence intervals for AUC

---

## Mathematics & Statistics Reference

### 1. Softmax Function

Converts raw class scores (logits) into a valid probability distribution over $K$ classes:

$$\sigma(\mathbf{z})_k = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, \dots, K$$

**Properties:** outputs sum to 1, each output is in $(0, 1)$, invariant to constant shift $\sigma(\mathbf{z}) = \sigma(\mathbf{z} - c)$ (used for numerical stability).

---

### 2. Cross-Entropy Loss

Measures the distance between the true one-hot label $\mathbf{y}_i$ and the predicted distribution $\hat{\mathbf{p}}_i$:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log\hat{p}_{ik}$$

For a single sample with true class $c$, this simplifies to $-\log\hat{p}_c$ (negative log-likelihood of the correct class).

---

### 3. Gradient of the Loss

The gradient with respect to the weight matrix $W_k$ for class $k$:

$$\frac{\partial \mathcal{L}}{\partial W_k} = \frac{1}{N} \mathbf{X}^\top (\hat{P}_k - Y_k)$$

Where $\hat{P}_k - Y_k$ is the residual (predicted probability minus true label) — the same form as in binary logistic regression.

---

### 4. Regularization

**L2 (Ridge)** — penalises large weights, encourages small distributed values:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \|W\|_F^2$$

**L1 (Lasso)** — penalises the sum of absolute weights, encourages sparsity:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|W\|_1$$

In scikit-learn, $C = 1/\lambda$ (inverse regularization strength).

---

### 5. Decision Boundary

For two classes $i$ and $j$, the decision boundary is the hyperplane where both class probabilities are equal:

$$\mathbf{w}_i^\top \mathbf{x} = \mathbf{w}_j^\top \mathbf{x} \implies (\mathbf{w}_i - \mathbf{w}_j)^\top \mathbf{x} = 0$$

With polynomial features $\phi(\mathbf{x})$ (degree $d$), the boundary becomes a degree-$d$ curve in the original feature space. For concentric circles the degree-2 boundary directly captures $x_1^2 + x_2^2 = r^2$.

---

### 6. Polynomial Feature Count

For $p$ input features expanded to degree $d$:

$$\text{number of features} = \binom{p + d}{d} - 1 \quad (\text{with } \texttt{include\_bias=False})$$

Example for $p = 2$ (two input features):

| Degree | Total features | Feature names |
|---|---|---|
| 1 | 2 | $x_1,\; x_2$ |
| 2 | 5 | $x_1,\; x_2,\; x_1^2,\; x_1 x_2,\; x_2^2$ |
| 3 | 9 | above + $x_1^3,\; x_1^2 x_2,\; x_1 x_2^2,\; x_2^3$ |

---

### 7. ROC Curve Metrics

For a binary classifier (or one class in OvR):

| Metric | Formula | Meaning |
|---|---|---|
| True Positive Rate (TPR / Recall) | $\text{TPR} = \dfrac{\text{TP}}{\text{TP} + \text{FN}}$ | Fraction of true positives correctly identified |
| False Positive Rate (FPR) | $\text{FPR} = \dfrac{\text{FP}}{\text{FP} + \text{TN}}$ | Fraction of true negatives incorrectly flagged |
| Precision | $P = \dfrac{\text{TP}}{\text{TP} + \text{FP}}$ | Fraction of positive predictions that are correct |
| Specificity | $\text{Spec} = 1 - \text{FPR}$ | Fraction of true negatives correctly identified |

AUC equals the probability that a randomly chosen positive is scored higher than a randomly chosen negative (Wilcoxon-Mann-Whitney statistic).

---

### 8. Multiclass AUC — One-vs-Rest (OvR)

Treat each class $k$ as positive and all others as negative. Three averaging strategies:

$$\text{AUC}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{AUC}(k)$$

$$\text{AUC}_{\text{weighted}} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot \text{AUC}(k)$$

$$\text{AUC}_{\text{micro}} = \text{AUC}\!\left(\bigcup_k \text{scores}_k,\ \bigcup_k \text{labels}_k\right)$$

Where $n_k$ is the number of samples of class $k$ and $N$ is the total sample count.

---

### 9. Multiclass AUC — One-vs-One (OvO, Hand-Till 2001)

Average pairwise AUCs over all $\binom{K}{2}$ class pairs:

$$\text{AUC}_{\text{OvO macro}} = \frac{2}{K(K-1)} \sum_{i < j} \frac{A(i \mid j) + A(j \mid i)}{2}$$

$$\text{AUC}_{\text{OvO weighted}} = \frac{2}{K(K-1)} \sum_{i < j} \frac{n_i + n_j}{2N} \cdot \frac{A(i \mid j) + A(j \mid i)}{2}$$

Where $A(i \mid j)$ is the AUC of class $i$ against class $j$ (the probability that a random $i$ is ranked above a random $j$).

---

### 10. Threshold Selection

Given a trained model, choose an operating threshold $t^*$ using one of:

| Rule | Formula | Use when |
|---|---|---|
| **Youden's J** | $t^* = \arg\max_t \bigl(\text{TPR}(t) - \text{FPR}(t)\bigr)$ | Equal cost for FP and FN |
| **Equal-Error Rate (EER)** | $t^*$ where $\text{FPR}(t) = 1 - \text{TPR}(t)$ | Symmetric error requirement |
| **Cost-sensitive** | $t^* = \arg\min_t \bigl(C_{\text{FP}} \cdot \text{FP}(t) + C_{\text{FN}} \cdot \text{FN}(t)\bigr)$ | FP and FN have different real-world costs |

---

### 11. Precision-Recall AUC & Average Precision

Average Precision (AP) summarises the Precision-Recall curve:

$$\text{AP} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k$$

Macro-averaged over $K$ classes:

$$\text{mAP} = \frac{1}{K} \sum_{k=1}^{K} \text{AP}(k)$$

Preferred over ROC-AUC when the positive class is rare (high class imbalance), because it does not count true negatives.

---

### 12. Gini Coefficient

A linear transform of AUC that maps $[0.5, 1] \to [0, 1]$:

$$\text{Gini} = 2 \cdot \text{AUC} - 1$$

Gini = 0 ↔ random classifier; Gini = 1 ↔ perfect classifier. Widely used in credit scoring.

---

### 13. Temperature Scaling

Controls the "confidence" of the Softmax output:

$$\sigma(\mathbf{z} / T)_k = \frac{e^{z_k / T}}{\displaystyle\sum_{j} e^{z_j / T}}$$

$T > 1$ flattens the distribution (softer, less confident); $T < 1$ sharpens it (harder, more confident); $T \to 0$ approaches one-hot argmax.

---

## Datasets

| File | Samples | Features | Classes | Used in |
|---|---|---|---|---|
| [nonlinear_data_circles.csv](datasets/nonlinear_data_circles.csv) | 10,000 | `x1`, `x2` | 0 = outer ring, 1 = inner circle | Notebooks 04, 05 |
| [ushape.csv](datasets/ushape.csv) | — | `x1`, `x2` | 0 = outer U, 1 = inner U | Notebook 04 |


