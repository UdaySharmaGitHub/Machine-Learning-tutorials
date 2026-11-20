# K-Nearest Neighbors (KNN) — Numerical Example

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Problem Statement](#2-problem-statement)
- [3. Training Dataset](#3-training-dataset)
- [4. Step 1: Calculate the Distance Using Euclidean Distance](#4-step-1-calculate-the-distance-using-euclidean-distance)
  - [Euclidean Distance Formula](#euclidean-distance-formula)
  - [Distance Calculations for All Points](#distance-calculations-for-all-points)
- [5. Step 2: Prepare the Distance Table](#5-step-2-prepare-the-distance-table)
- [6. Step 3: Sort the Distances in Ascending Order](#6-step-3-sort-the-distances-in-ascending-order)
- [7. Step 4: Select the K Nearest Neighbors](#7-step-4-select-the-k-nearest-neighbors)
- [8. Step 5: Perform Majority Voting (Classification)](#8-step-5-perform-majority-voting-classification)
- [9. Step 6: Final Prediction](#9-step-6-final-prediction)
- [10. Complete KNN Workflow Summary](#10-complete-knn-workflow-summary)
- [11. Important Concept: Feature Scaling in KNN](#11-important-concept-feature-scaling-in-knn)
- [12. A Complete Classification Example](#12-a-complete-classification-example)
  - [12.1 Calculate Distances](#121-calculate-distances)
  - [12.2 Rank and Vote](#122-rank-and-vote)
  - [12.3 Distance-Weighted Vote](#123-distance-weighted-vote)
- [13. A Complete Regression Example](#13-a-complete-regression-example)
  - [13.1 Uniform Prediction](#131-uniform-prediction)
  - [13.2 Inverse-Distance Prediction](#132-inverse-distance-prediction)
  - [13.3 Local Uncertainty is Not Automatic](#133-local-uncertainty-is-not-automatic)
- [14. Important Concept: How Do We Choose K?](#14-important-concept-how-do-we-choose-k)
- [15. What Happens When K Is Too Small or Too Large?](#15-what-happens-when-k-is-too-small-or-too-large)
- [16. Important Points to Remember for Exams](#16-important-points-to-remember-for-exams)
- [17. Final Conclusion](#17-final-conclusion)

---

## 1. Introduction

***K-Nearest Neighbors (KNN)*** is a ***supervised machine learning algorithm*** used for both ***classification*** and ***regression*** problems.

KNN works by finding the ***K closest data points*** (neighbors) to a new data point and using those neighbors to make a prediction.

- ***Classification:*** Predicts a categorical class using ***majority voting***.
- ***Regression:*** Predicts a numerical value using the ***average*** of neighboring target values.

In this numerical example, we will use KNN for a ***classification problem*** to predict whether a website visitor will purchase something based on their browsing behavior.

---

## 2. Problem Statement

Suppose we have historical data about website visitors.

We want to predict whether a new visitor will purchase something based on two features:

1. ***Pages Visited:*** Number of pages visited by the user.
2. ***Time on Site:*** Time spent by the user on the website, measured in minutes.

The ***target variable*** is:

- `Yes`: The visitor purchased something.
- `No`: The visitor did not purchase anything.

### Given New Data Point

Suppose a new visitor has the following characteristics:

| Feature | Value |
|---|---:|
| Pages Visited | 5 |
| Time on Site | 12 minutes |
| Purchased | Unknown |

Our objective is to predict whether this visitor will purchase something using KNN.

We will use:

$$K = 3$$

This means we will consider the ***three nearest neighbors*** to make our prediction.

---

## 3. Training Dataset

We have the following historical training dataset containing eight visitors.

| Visitor | Pages Visited (X₁) | Time on Site (X₂) | Purchased (Y) |
|---|---:|---:|---|
| P1 | 2 | 5 | No |
| P2 | 3 | 8 | No |
| P3 | 5 | 15 | Yes |
| P4 | 7 | 20 | Yes |
| P5 | 6 | 18 | Yes |
| P6 | 1 | 3 | No |
| P7 | 4 | 10 | No |
| P8 | 8 | 25 | Yes |

#### Visualization: Training Dataset

<div align="center">

![Training dataset](./images/Numerical%20Images/01_training_dataset_Numerical.png)

</div>

### New Data Point

| Feature | Value |
|---|---:|
| Pages Visited | 5 |
| Time on Site | 12 |
| Purchased | ? |

#### Visualization: New Query Point

<div align="center">

![New query point](./images/Numerical%20Images/02_new_query_point_Numerical.png)

</div>

We need to identify the three closest visitors to this new data point.

---

## 4. Step 1: Calculate the Distance Using Euclidean Distance

The first step in KNN is calculating the ***distance*** between the new data point and every existing training data point.

We will use the ***Euclidean Distance Formula***.

### Euclidean Distance Formula

For two-dimensional data:

$$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{(x_1 - x_{i1})^2 + (x_2 - x_{i2})^2}$$

**Where:**

- $\mathbf{x}$ = ***New data point***
- $\mathbf{x}_i$ = ***Existing training data point***
- $x_1, x_2$ = Features of the new data point
- $x_{i1}, x_{i2}$ = Features of the existing data point
- $d(\mathbf{x}, \mathbf{x}_i)$ = ***Distance*** between the two points

Our new data point is:

$$\mathbf{x} = (5, 12)$$

We calculate its distance from all eight visitors.

### Distance Calculations for All Points

#### Distance for P1

P1 = (2, 5)

$$d(P1) = \sqrt{(5-2)^2 + (12-5)^2}$$

$$= \sqrt{3^2 + 7^2}$$

$$= \sqrt{9 + 49}$$

$$= \sqrt{58}$$

$$\boxed{d(P1) = 7.616}$$

#### Distance for P2

P2 = (3, 8)

$$d(P2) = \sqrt{(5-3)^2 + (12-8)^2}$$

$$= \sqrt{2^2 + 4^2}$$

$$= \sqrt{4 + 16}$$

$$= \sqrt{20}$$

$$\boxed{d(P2) = 4.472}$$

#### Distance for P3

P3 = (5, 15)

$$d(P3) = \sqrt{(5-5)^2 + (12-15)^2}$$

$$= \sqrt{0^2 + (-3)^2}$$

$$= \sqrt{0 + 9}$$

$$\boxed{d(P3) = 3.000}$$

#### Distance for P4

P4 = (7, 20)

$$d(P4) = \sqrt{(5-7)^2 + (12-20)^2}$$

$$= \sqrt{(-2)^2 + (-8)^2}$$

$$= \sqrt{4 + 64}$$

$$= \sqrt{68}$$

$$\boxed{d(P4) = 8.246}$$

#### Distance for P5

P5 = (6, 18)

$$d(P5) = \sqrt{(5-6)^2 + (12-18)^2}$$

$$= \sqrt{(-1)^2 + (-6)^2}$$

$$= \sqrt{1 + 36}$$

$$= \sqrt{37}$$

$$\boxed{d(P5) = 6.083}$$

#### Distance for P6

P6 = (1, 3)

$$d(P6) = \sqrt{(5-1)^2 + (12-3)^2}$$

$$= \sqrt{4^2 + 9^2}$$

$$= \sqrt{16 + 81}$$

$$= \sqrt{97}$$

$$\boxed{d(P6) = 9.849}$$

#### Distance for P7

P7 = (4, 10)

$$d(P7) = \sqrt{(5-4)^2 + (12-10)^2}$$

$$= \sqrt{1^2 + 2^2}$$

$$= \sqrt{1 + 4}$$

$$= \sqrt{5}$$

$$\boxed{d(P7) = 2.236}$$

#### Distance for P8

P8 = (8, 25)

$$d(P8) = \sqrt{(5-8)^2 + (12-25)^2}$$

$$= \sqrt{(-3)^2 + (-13)^2}$$

$$= \sqrt{9 + 169}$$

$$= \sqrt{178}$$

$$\boxed{d(P8) = 13.342}$$

---

## 5. Step 2: Prepare the Distance Table

After calculating the distances, we combine all results into one table.

| Visitor | Distance | Purchased |
|---|---:|---|
| P1 | 7.616 | No |
| P2 | 4.472 | No |
| P3 | 3.000 | Yes |
| P4 | 8.246 | Yes |
| P5 | 6.083 | Yes |
| P6 | 9.849 | No |
| P7 | 2.236 | No |
| P8 | 13.342 | Yes |

#### Visualization: Distances to All Points

<div align="center">

![Distances to all points](./images/Numerical%20Images/03_distances_to_all_points_Numerical.png)

</div>

> [!IMPORTANT]
> We do ***not*** select neighbors based on their class labels. We select them ***only based on their distances*** — the class label is ignored at this stage.

---

## 6. Step 3: Sort the Distances in Ascending Order

Now we arrange all distances from ***smallest to largest***.

The ***smallest distance*** represents the ***closest neighbor***.

| Rank | Visitor | Distance | Purchased |
|---:|---|---:|---|
| 1 | P7 | 2.236 | No |
| 2 | P3 | 3.000 | Yes |
| 3 | P2 | 4.472 | No |
| 4 | P5 | 6.083 | Yes |
| 5 | P1 | 7.616 | No |
| 6 | P4 | 8.246 | Yes |
| 7 | P6 | 9.849 | No |
| 8 | P8 | 13.342 | Yes |

This ranking helps us identify the ***nearest neighbors***.

---

## 7. Step 4: Select the K Nearest Neighbors

We have already selected:

$$K = 3$$

Therefore, we select the ***first three data points*** with the ***smallest distances***.

Our three nearest neighbors are:

| Neighbor | Visitor | Distance | Purchased |
|---|---|---:|---|
| 1st | P7 | 2.236 | No |
| 2nd | P3 | 3.000 | Yes |
| 3rd | P2 | 4.472 | No |

These are the three closest visitors to our new visitor.

> [!NOTE]
> **Why do we select only three?** Because $K = 3$ means KNN considers ***only the three nearest training data points*** when making the prediction. The remaining five visitors (P5, P1, P4, P6, P8) are completely ignored in this prediction.

#### Visualization: K=3 Nearest Neighbors Highlighted

<div align="center">

![K=3 nearest neighbors](./images/Numerical%20Images/04_highlight_k3_neighbors_Numerical.png)

</div>

---

## 8. Step 5: Perform Majority Voting (Classification)

Since this is a ***classification problem***, we use ***majority voting***.

We examine the class labels of the three nearest neighbors.

| Neighbor | Purchased |
|---|---|
| P7 | No |
| P3 | Yes |
| P2 | No |

Now count the occurrences of each class.

| Class | Number of Votes |
|---|---:|
| ***No*** | 2 |
| Yes | 1 |

We can see that:

- ***No*** received **2 votes**.
- Yes received 1 vote.

Since ***No*** has the ***majority***, KNN predicts that the new visitor will **not purchase anything**.

#### Visualization: Majority Voting Result

<div align="center">

![Majority voting](./images/Numerical%20Images/05_majority_voting_Numerical.png)

</div>

### Majority Voting Formula

$$\boxed{\hat{y}(\mathbf{x}) = \operatorname{mode}\{y_i : i \in N_k(\mathbf{x})\}}$$

**Where:**

- $\hat{y}(\mathbf{x})$ = ***Predicted class***
- $N_k(\mathbf{x})$ = ***Set of K nearest neighbors***
- $y_i$ = ***Class label*** of each neighbor
- $\operatorname{mode}$ = ***Most frequently occurring class***

---

## 9. Step 6: Final Prediction

Our new visitor has:

- ***Pages Visited*** = 5
- ***Time on Site*** = 12 minutes

After calculating distances, selecting the three nearest neighbors, and performing majority voting, we get:

$$\boxed{\text{Purchased} = \textbf{No}}$$

> [!TIP]
> The KNN algorithm predicts that the new visitor will ***not purchase anything*** because 2 of the 3 nearest neighbors belong to the **No** class. The decision is based entirely on ***proximity*** — the algorithm finds visitors who behaved most similarly and uses their outcomes.

---

## 10. Complete KNN Workflow Summary

The entire numerical example follows these steps:

1. **Prepare the dataset:** Identify the training data, features, and ***target variable***.
2. **Select K:** Choose the number of nearest neighbors. In this example, $K = 3$.
3. **Calculate distances:** Calculate the ***Euclidean distance*** between the new data point and every training data point.
4. **Sort distances:** Arrange the calculated distances in ***ascending order***.
5. **Find nearest neighbors:** Select the first $K$ data points with the ***smallest distances***.
6. **Majority voting:** Count the class labels of the selected neighbors and identify the ***most frequent class***.
7. **Make prediction:** Return the class with the ***highest number of votes***.

### Final Result

$$\boxed{\text{Prediction} = \textbf{No Purchase}}$$

---

## 11. Important Concept: Feature Scaling in KNN

One ***critical*** consideration when implementing KNN is ***feature scaling***.

Our dataset contains two features:

- ***Pages Visited:*** Values range from 1 to 8.
- ***Time on Site:*** Values range from 3 to 25.

Notice that the numerical range of ***Time on Site*** is much larger than that of ***Pages Visited***.

Since ***Euclidean distance*** depends on numerical differences between features, a feature with a ***larger scale*** can have a ***disproportionate influence*** on the distance calculation.

### Why Is Feature Scaling Important?

The critical insight is that ***in distance-based algorithms like KNN, a feature's numerical scale acts as an implicit weight***, regardless of its actual predictive importance.

> [!NOTE]
> For a ***comprehensive deep dive*** into feature scaling methods, preprocessing pipelines, outlier handling, and sklearn implementation, see the dedicated guide: [Feature scaling and preprocessing.md](./Feature%20scaling%20and%20preprocessing.md)

#### The Distance Weighting Problem

Suppose one feature has values between 1 and 10, while another has values between 1 and 10,000.

The second feature may ***dominate the distance calculation*** even when both features are equally important.

**Concrete Example: Age vs Income**

Consider two features with very different scales:
- ***Age (in years):*** typically ranges from 20 to 80
- ***Annual Income (in rupees):*** typically ranges from 100,000 to 10,000,000

For two people with differences:
- Age difference: $\Delta \text{age} = 10$ years
- Income difference: $\Delta \text{income} = 100,000$ rupees

Their contribution to ***squared Euclidean distance*** is:

$$(\Delta \text{age})^2 = 10^2 = 100$$

$$(\Delta \text{income})^2 = (100,000)^2 = 10,000,000,000 = 10^{10}$$

The ***income difference dominates by a factor of 100,000,000***! Even if age is ***more predictive*** of the target, the model will essentially ignore it.

| Feature | Scale | Distance Contribution | Influence |
|---------|-------|----------------------|-----------|
| ***Age (years)*** | 20–80 | $10^2 = 100$ | ~0.0001% |
| ***Income (rupees)*** | 100K–10M | $(100K)^2 = 10^{10}$ | ~99.9999% |

The distance metric is essentially decided by income alone, rendering age information invisible.

> [!WARNING]
> In this numerical example, the unscaled distances are computed correctly as given. However, in ***real machine learning practice***, you should apply ***feature scaling before computing distances***. Without scaling, features with larger numerical ranges will unfairly dominate.

To avoid this issue, we commonly use:

**1. Min-Max Normalization**

$$X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

This scales feature values to a specified range, commonly between **0 and 1**.

**After scaling both age and income to [0, 1]:**

| Feature | Scale | Distance Contribution | Influence |
|---------|-------|----------------------|-----------|
| ***Age (scaled)*** | 0–1 | ~0.01 | ~50% |
| ***Income (scaled)*** | 0–1 | ~0.01 | ~50% |

Both features now contribute ***equally*** to the distance calculation, as intended.

> [!TIP]
> Min-Max scaling is simple and interpretable. Use it when data has ***no extreme outliers*** and you need values in a fixed range.

**2. Standardization (Z-score Normalization)**

$$Z = \frac{X - \mu}{\sigma}$$

**Where:**

- $X$ = Original value
- $\mu$ = ***Mean*** of the feature
- $\sigma$ = ***Standard deviation***

This transforms features to have a ***mean of 0*** and a ***standard deviation of 1***.

> [!NOTE]
> Standardization is ***symmetric*** and works well for normally distributed features. Outliers affect both the mean and standard deviation, so it's less robust than other methods.

**3. Robust Scaling (Median and IQR)**

For data with ***significant outliers***, use robust scaling:

$$X' = \frac{X - \text{median}(X)}{IQR}$$

Where $IQR = Q_{75\%} - Q_{25\%}$ (interquartile range).

> [!IMPORTANT]
> ***Robust scaling is the best choice for real-world data*** because it uses the median and IQR, which are ***resistant to extreme outliers***. Unlike mean and standard deviation, these statistics are barely affected by outliers.

> [!IMPORTANT]
> The ***scaling parameters*** (mean, standard deviation, min, max) must be ***learned from the training data only*** and then applied to the new data point. Never fit the scaler on the entire dataset including test data — that causes ***data leakage***.

---

## 12. A Complete Classification Example

Let's walk through a simpler ***classification*** example to solidify the KNN process.

Suppose the ***query point*** is $\mathbf{x} = (4, 4)$ and the training data are:

| Point | Coordinates | Class |
|---|---:|---|
| A | $(2, 3)$ | Red |
| B | $(5, 4)$ | Blue |
| C | $(3, 5)$ | Red |
| D | $(6, 6)$ | Blue |
| E | $(4, 7)$ | Blue |
| F | $(1, 1)$ | Red |

We use ***Euclidean distance*** and $k = 3$.

### 12.1 Calculate Distances

Distance from query $(4, 4)$ to each point:

$$d_A = \sqrt{(4-2)^2 + (4-3)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.236$$

$$d_B = \sqrt{(4-5)^2 + (4-4)^2} = \sqrt{1 + 0} = 1.000$$

$$d_C = \sqrt{(4-3)^2 + (4-5)^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.414$$

$$d_D = \sqrt{(4-6)^2 + (4-6)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.828$$

$$d_E = \sqrt{(4-4)^2 + (4-7)^2} = \sqrt{0 + 9} = 3.000$$

$$d_F = \sqrt{(4-1)^2 + (4-1)^2} = \sqrt{9 + 9} = \sqrt{18} \approx 4.243$$

### 12.2 Rank and Vote

Sort by distance and select the ***3 nearest neighbors***:

| Rank | Point | Distance | Class |
|---:|---|---:|---|
| 1 | B | $1.000$ | Blue |
| 2 | C | $1.414$ | Red |
| 3 | A | $2.236$ | Red |
| 4 | D | $2.828$ | Blue |
| 5 | E | $3.000$ | Blue |
| 6 | F | $4.243$ | Red |

The nearest three neighbors are ***B (Blue), C (Red), and A (Red)***.

***Majority voting***: 2 Red votes vs 1 Blue vote.

$$\boxed{\hat{y}(4, 4) = \text{Red}}$$

***Estimated class probabilities***:

$$\hat{P}(\text{Red} \mid (4, 4)) = \frac{2}{3} \approx 0.667$$

$$\hat{P}(\text{Blue} \mid (4, 4)) = \frac{1}{3} \approx 0.333$$

### 12.3 Distance-Weighted Vote

With ***inverse-distance weighting*** $w_i = 1/d_i$, the weighted scores are:

$$S_{\text{Blue}} = \frac{1}{d_B} = \frac{1}{1} = 1.000$$

$$S_{\text{Red}} = \frac{1}{d_C} + \frac{1}{d_A} = \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{5}} \approx 0.707 + 0.447 = 1.154$$

> [!TIP]
> Even with ***weighted voting***, ***Red*** still wins because it has the ***higher weighted score***. However, the margin is narrower (1.154 vs 1.000) compared to unweighted voting (2 vs 1). This shows that ***weighting can change the confidence of the prediction*** even when the predicted class remains the same.

---

## 13. A Complete Regression Example

Now let's see how KNN works for ***regression***.

Suppose we're predicting the price of a home, and the ***three nearest homes*** to our query have the following distances and sale prices:

| Neighbour | Distance | Price (lakh) |
|---|---:|---:|
| 1 | $0.5$ | $60$ |
| 2 | $1.0$ | $72$ |
| 3 | $2.0$ | $90$ |

### 13.1 Uniform Prediction

With ***unweighted (uniform) KNN regression***, the prediction is the ***simple mean*** of the target values:

$$\boxed{\hat{y} = \frac{60 + 72 + 90}{3} = \frac{222}{3} = 74 \text{ lakh}}$$

### 13.2 Inverse-Distance Prediction

With ***inverse-distance weighting***, the weights are $w_i = 1/d_i$:

- $w_1 = 1/0.5 = 2.000$
- $w_2 = 1/1.0 = 1.000$
- $w_3 = 1/2.0 = 0.500$

The ***weighted mean*** is:

$$\hat{y} = \frac{w_1 \cdot 60 + w_2 \cdot 72 + w_3 \cdot 90}{w_1 + w_2 + w_3}$$

$$= \frac{2(60) + 1(72) + 0.5(90)}{2 + 1 + 0.5}$$

$$= \frac{120 + 72 + 45}{3.5}$$

$$= \frac{237}{3.5}$$

$$\boxed{\hat{y} \approx 67.714 \text{ lakh}}$$

> [!NOTE]
> The ***nearest neighbor*** (distance 0.5) has the lowest price ($60), so it exerts the greatest influence with weighting. This pulls the weighted prediction ***lower*** (67.714) compared to the uniform mean (74.000). ***Distance weighting*** makes sense in regression when nearby data points are expected to be more predictive.

### 13.3 Local Uncertainty is Not Automatic

The ***spread of neighbor targets*** — here ranging from 60 to 90 — can be reported as a local diagnostic to assess ***variability*** in the neighborhood.

> [!IMPORTANT]
> Ordinary KNN does ***not*** automatically provide a valid ***prediction interval*** with statistical coverage guarantees. If you need uncertainty quantification for regression, consider:
> - ***Quantile regression*** (predicting quantiles instead of means)
> - ***Conformal prediction*** (wrapping KNN with a conformal predictor)
> - ***Bootstrapping*** (sampling neighbors with replacement to estimate variance)
> - ***Probabilistic local models*** (fitting a local regression model around the query point)

---

## 14. Important Concept: How Do We Choose K?

The value of $K$ is an important ***hyperparameter*** in KNN.

There are two common approaches to selecting $K$.

### Method 1: Heuristic Approach

We use a ***rule of thumb*** to get an initial estimate of $K$.

One commonly used heuristic is:

$$K = \sqrt{N}$$

**Where:**

- $K$ = Number of neighbors
- $N$ = Total number of training samples

For example, if we have 100 training samples:

$$K = \sqrt{100} = 10$$

Therefore, we can initially consider $K = 10$.

> [!NOTE]
> In ***binary classification***, an ***odd K*** is sometimes preferred to reduce the possibility of ties. However, the square-root rule is only a rough guideline, not a guarantee of the best K. Its performance depends on many factors including dimensionality, noise level, and class overlap.

### Method 2: Experimental Approach (Cross-Validation)

In this approach, we test multiple $K$ values and evaluate their performance using ***cross-validation***.

For example:

| K | Accuracy |
|---:|---:|
| 1 | 85% |
| 3 | 89% |
| 5 | 92% |
| 7 | 95% |
| 9 | 93% |
| 11 | 91% |

Here, $K = 7$ gives the highest accuracy.

> [!TIP]
> For ***reliable results***, use ***cross-validation*** (such as k-fold or LOOCV) rather than relying on a single train-test split. Cross-validation evaluates the model's performance across multiple subsets of the data, giving a more stable and unbiased estimate of the optimal K.

---

## 15. What Happens When K Is Too Small or Too Large?

The choice of $K$ affects the ***complexity*** of the KNN decision boundary.

### Small K — Overfitting

When $K$ is very small:

- The model becomes ***sensitive to noise***.
- A single nearby data point can strongly influence the prediction.
- The model may ***overfit*** the training data.
- Decision boundaries become ***irregular and jagged***.

Example: $K = 1$

> [!WARNING]
> With $K = 1$, the model predicts based on a ***single nearest neighbor***. Any noisy or mislabeled point near the decision boundary can cause incorrect predictions. This is the most extreme case of ***overfitting*** in KNN.

### Large K — Underfitting

When $K$ is very large:

- The model considers many ***distant data points***.
- ***Local patterns*** may be ignored.
- The model may become too generalized and ***underfit*** the data.
- Decision boundaries become ***overly smooth***.

Example: $K = N$ (all training points)

> [!WARNING]
> With $K = N$, every query point returns the same prediction — the ***most common class*** in the entire training set (global majority). The model completely ignores any local structure in the data. This is the most extreme case of ***underfitting*** in KNN.

### Optimal K

A suitable $K$ ***balances overfitting and underfitting***.

***Cross-validation*** is the standard method to identify a suitable value that minimizes validation error.

---

## 16. Important Points to Remember for Exams

1. KNN is a ***supervised machine learning algorithm***.
2. KNN can be used for both ***classification*** and ***regression***.
3. KNN is a ***non-parametric algorithm*** — no assumption about the data distribution.
4. KNN is also called a ***lazy learning algorithm*** because it does not build an explicit model during training; all computation happens at prediction time.
5. KNN calculates ***distances*** between a new data point and existing training data points.
6. ***Euclidean distance*** is one of the commonly used distance metrics.
7. KNN selects the ***K nearest data points*** based on distance.
8. ***Classification*** uses majority voting.
9. ***Regression*** commonly uses the average of neighboring target values.
10. ***Feature scaling*** is important because KNN is distance-based.
11. A very ***small K*** can lead to ***overfitting***.
12. A very ***large K*** can lead to ***underfitting***.
13. $K$ can be initially estimated using the ***square-root heuristic*** and selected more reliably using ***cross-validation***.

---

## 17. Final Conclusion

In this numerical example, we used the ***K-Nearest Neighbors*** algorithm to predict whether a new website visitor would purchase something.

We followed the complete KNN classification process:

1. Calculated ***Euclidean distances*** from the new point to all 8 training points.
2. Sorted distances in ***ascending order***.
3. Selected the ***3 nearest neighbors***: P7, P3, P2.
4. Applied ***majority voting***: No (2 votes) vs Yes (1 vote).
5. Obtained the ***final prediction***.

Since two out of the three nearest neighbors belong to the **No Purchase** class, the final prediction is:

$$\boxed{\text{No Purchase}}$$

> [!NOTE]
> This example demonstrates how KNN uses the ***similarity between existing data points*** and a new data point to make predictions. KNN first finds the nearest neighbors using ***distance*** and then uses their target labels or values to make a prediction. The algorithm is simple to understand and implement, but requires careful attention to ***feature scaling*** and ***K selection*** in practice.

---

<p align="center">
  ⬅️ <strong>Previous:</strong> <a href="./05%20Feature%20scaling%20and%20preprocessing.md">05 Feature scaling and preprocessing</a> &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Next:</strong> <a href="./07%20Decision%20Surface%20and%20Boundaries.md">07 Decision Surface and Boundaries</a> ➡️
</p>
