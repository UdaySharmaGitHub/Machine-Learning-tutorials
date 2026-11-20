# K-Nearest Neighbours (KNN)

K-Nearest Neighbours is a simple yet powerful machine learning algorithm that makes predictions by finding the **K closest data points** in the training set and aggregating their labels or values. It requires no training phase — all computation happens at prediction time — making it one of the most intuitive examples of ***instance-based*** and ***lazy learning***.

This module builds the algorithm from the ground up:

1. **Introduction & Key Concepts** — what KNN is, its core properties, meaning of K, non-parametric nature, lazy learning, and the neighbourhood principle.
2. **Algorithm and Working** — step-by-step breakdown with math formulas, pseudocode, and visual diagrams.
3. **Distance Metrics** — Euclidean, Manhattan, Minkowski, and other metrics that drive KNN decisions.
4. **Code and Numerical Examples** — hands-on implementation and worked examples to solidify understanding.

---

## Folder Structure

| # | Type | File | Description |
|---|---|---|---|
| 00 | 📝 Notes | [KNN Intro and Key Concepts](./00%20KNN%20Intro%20and%20Key%20Concept.md) | ***What is KNN***, **core properties**, *meaning of K*, ***non-parametric*** nature, ***lazy learning***, ***neighbourhood principle***, and **workflow diagram** |
| 01 | 📝 Notes | [KNN Algorithm and Working Step-by-Step](./01%20KNN%20Algorithm%20and%20Working%20Step-by-Step..md) | ***Step-by-step algorithm*** with **math formulas**, *distance computation*, **neighbor selection**, and *aggregation* |
| 02 | 📝 Notes | [Distance Metrics](./02%20Distance%20Metrics.md) | ***Euclidean***, ***Manhattan***, ***Minkowski***, ***Hamming***, ***Cosine***, ***Mahalanobis***, ***Chebyshev***, ***Jaccard*** — *when to use each*, **comparisons**, and **visualizations** |
| 03 | 📝 Notes | [How to Choose K in KNN: Bias, Variance and Validation](./03%20How%20to%20Choose%20K%20in%20KNN%3A%20Bias%2C%20Variance%20and%20Validation.md) | ***Bias-variance tradeoff***, ***cross-validation***, **K selection strategies**, and *practical guidelines* |
| 04 | 📝 Notes | [KNN Classification and Regression](./04%20KNN%20Classification%20and%20Regression.md) | ***Majority voting*** for **classification**, ***local mean/median*** for **regression**, **mathematical derivations**, and **theoretical foundations** |
| 04.1 | 📓 Notebook | [KNN Classifier From Scratch](./04.1%20KNN%20Classifier%20From%20Scratch.ipynb) | **From-scratch implementation**: *distance computation*, *neighbor selection*, *majority voting*, and *decision boundary visualization* |
| 04.2 | 📓 Notebook | [KNN Classifier Using Scikit-learn](./04.2%20KNN%20Classifier%20Using%20Sickit-learn.ipynb) | **Scikit-learn workflow**: *model instantiation*, *training*, *predictions*, *evaluation*, and *hyperparameter tuning* |
| 04.3 | 📓 Notebook | [KNN Regressor From Scratch](./04.3%20KNN%20Regressor%20From%20Scratch.ipynb) | **From-scratch regression**: *local mean predictions*, *distance-weighted averaging*, and *performance metrics* |
| 04.4 | 📓 Notebook | [KNN Regressor Using Scikit-learn](./04.4%20KNN%20Regressor%20Using%20Sickit-learn.ipynb) | **Scikit-learn regression**: *model training*, *predictions*, *residual analysis*, and *error comparison* |
| 05 | 📝 Notes | [Feature Scaling and Preprocessing](./05%20Feature%20scaling%20and%20preprocessing.md) | ***Standardization***, ***normalization***, ***feature importance***, and **impact on KNN performance** |
| 06 | 📝 Notes | [Numerical Example on KNN](./06%20Numerical%20Example%20on%20KNN.md) | **Step-by-step worked example**: *distance calculation*, *neighbor selection*, and *final prediction* on a concrete dataset |
| 07 | 📝 Notes | [Decision Surface and Boundaries](./07%20Decision%20Surface%20and%20Boundaries.md) | ***Decision boundaries***, **visualization techniques**, *interpretation in high dimensions*, and **overfitting/underfitting patterns** |
| 08 | 📝 Notes | [Weighted KNN](./08%20Weighted%20KNN.md) | ***Distance-weighted voting***, ***kernel functions***, **smooth probability estimates**, and **bias reduction** |
| 08.1 | 📓 Notebook | [Weighted KNN Notebook](./08.1%20Weighted%20KNN%20notebook.ipynb) | **Weighted KNN implementation**: *comparison with uniform weights*, *impact on boundaries*, and **performance evaluation** |
| 09 | 📝 Notes | [Curse of Dimensionality and Special Data Types](./09%20Curse%20of%20Dimensionality%20and%20Special%20Data%20Types.md) | ***High-dimensional breakdown***, ***text & categorical data***, **distance metrics for special types**, and **practical mitigation strategies** |
| 10 | 📝 Notes | [Search Algorithms and Computational Complexity](./10%20Search%20algorithms%20and%20computational%20complexity.md) | ***Brute-force***, ***KD-tree***, ***Ball-tree***, ***HNSW***, ***LSH***, ***IVF+PQ*** — *complexity analysis*, **memory trade-offs**, and **production deployment** |
| 11 | 📝 Notes | [End-to-End KNN Modelling Workflow](./11%20End-to-End%20KNN%20Modelling%20Workflow.md) | **Complete pipeline**: *data preparation*, *cross-validation*, *hyperparameter tuning*, *model evaluation*, and **production deployment checklist** |
| 12 | 📝 Notes | [Evaluation in KNN](./12%20Evaluation%20in%20KNN.md) | ***Classification metrics*** (accuracy, precision, recall, F1, ROC-AUC), ***regression metrics*** (MAE, MSE, RMSE, R²), ***probability calibration***, and **operational concerns** |
| 12.1 | 📓 Notebook | [Evaluation in KNN - Implementation](./12.1%20Evaluation%20in%20KNN%20-%20implementation.ipynb) | **Hands-on evaluation**: *confusion matrices*, *classification reports*, *cross-validation scores*, and **metric comparisons** |
| 13 | 📝 Notes | [Class Imbalance, Ties, and Edge Cases](./13%20Class%20imbalance%2C%20ties%2C%20and%20edge%20cases.md) | ***Imbalanced classes***, ***vote ties***, ***distance ties***, ***duplicate inputs***, ***self-neighbors***, and **adversarial considerations** |
| — | 📂 Folder | [Other Concepts](./Other%20concept/) | ***KD-Tree***, ***Ball-Tree***, **spatial indexing**, and **KNN optimization** techniques |
| — | 📂 Folder | [Images](./images/) | **Visualizations** for all distance metrics, algorithms, decision surfaces, and worked examples |
| — | 📂 Folder | [Datasets](./datasets/) | **Sample datasets** for hands-on learning and experimentation |

---

## Known Issues

| # | Issue | Affected Files | Status |
|---|-------|---------------|--------|
| 1 | **SVG alignment** — several inline SVG diagrams are slightly misaligned; elements such as labels, arrows, and shapes appear off-centre or clipped compared to their intended layout | Multiple `.md` files across this module | 🔧 **Needs Fix** |

---

## Related Mathematics & ML Foundations

This KNN series builds upon foundational concepts from other modules in the Machine Learning tutorials. The table below shows which topics are covered within the KNN module and which reference external Mathematics materials:

| Topic | Module/Folder | Coverage | Status | Notes |
|---|---|---|---|---|
| ***Distance Metrics*** | [02 Distance Metrics](./02%20Distance%20Metrics.md) & [Mathematics / Distance Metrics](../Mathematics%20of%20Machine%20Learning/Distance%20Metrices/) | Euclidean, Manhattan, Minkowski, Hamming, Cosine, Mahalanobis, Chebyshev, Jaccard, and custom metrics | ✅ **Complete** | KNN chapter has formulas & comparisons; Mathematics folder has foundational theory |
| ***Probability Calibration*** | [12 Evaluation in KNN](./12%20Evaluation%20in%20KNN.md) & [Mathematics / Probability Calibration](../Mathematics%20of%20Machine%20Learning/Probability%20Calibration/) | Calibration curves, reliability diagrams, how K controls calibration, probability quality metrics | ✅ **Complete** | KNN chapter covers calibration in context of predictions; Mathematics folder has theoretical foundations |
| ***Decision Boundaries*** | [07 Decision Surface and Boundaries](./07%20Decision%20Surface%20and%20Boundaries.md) | Visualization, interpretation, dimensionality effects, boundary smoothness | ✅ **Complete** | 2D/3D visualizations showing how K affects boundary complexity |

---

### What is KNN?
- A ***supervised***, ***non-parametric***, ***instance-based***, ***lazy*** learning algorithm
- Works for both **classification** (majority voting) and **regression** (average values)
- No training phase — stores the entire dataset and computes at prediction time
- The meaning of ***K*** — how many neighbors influence the final prediction

### Key Concepts
- ***Non-parametric:*** model capacity grows with data, no fixed parameter vector
- ***Lazy learning:*** training is $O(nd)$; all computation deferred to prediction time
- ***Neighbourhood principle:*** nearby points in feature space tend to have similar targets
- ***Distance-based decision:*** choice of metric and feature scaling are critical

### The KNN Algorithm
- Compute distances from query point to all training points
- Select the $k$ nearest neighbors: $N_k(\mathbf{x}) = \{(1),(2),\ldots,(k)\}$
- **Classification:** $\hat{y}(\mathbf{x}) = \operatorname{mode}\{y_i : i \in N_k(\mathbf{x})\}$
- **Regression:** $\hat{f}(\mathbf{x}) = \frac{1}{k}\sum_{i \in N_k(\mathbf{x})} y_i$
- Optimization: partial selection in $O(n)$ vs full sort in $O(n \log n)$

### Distance Metrics
- **Euclidean distance:** $d(x, x_i) = \sqrt{\sum_{j=1}^{n}(x_j - x_{ij})^2}$ — straight-line distance in Euclidean space
- **Manhattan (L₁):** sum of absolute differences — useful in high dimensions
- **Minkowski (Lₚ):** generalized distance with parameter p; p=1 is Manhattan, p=2 is Euclidean, p→∞ is Chebyshev
- **Hamming:** counts position mismatches in binary/categorical data
- **Cosine:** angle-based similarity for text and high-dimensional sparse data
- **Mahalanobis:** weighted Euclidean accounting for feature correlations and covariance
- **Chebyshev (L∞):** maximum absolute difference — "Chessboard distance"
- **Jaccard:** set similarity based on intersection and union
- **Impact of feature scaling and metric choice** on model performance

### KNN Classification and Regression
- **Classification:** Majority voting among K neighbors; estimated class probabilities; handling ties; multiclass vs binary
- **Regression:** Local mean of K neighbors; distance-weighted averaging; local median (robust to outliers); interpolation vs extrapolation
- **Mathematical foundations:** Why local mean minimizes squared error; weighted mean derivation; asymptotic consistency
- **Kernel view:** KNN as adaptive-bandwidth kernel regression; density-dependent neighborhood size
