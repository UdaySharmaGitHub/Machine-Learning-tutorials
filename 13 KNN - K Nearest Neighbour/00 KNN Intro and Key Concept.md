# K-Nearest Neighbours (KNN): Introduction & Key Concepts

## Table of Contents

- [Introduction](#introduction)
- [What is 'K' in K-Nearest Neighbour?](#what-is-k-in-k-nearest-neighbour)
- [Key Concepts](#key-concepts)
  - [Key Vocabulary and Notation](#key-vocabulary-and-notation)
  - [1.1 What KNN Learns During Fit](#11-what-knn-learns-during-fit)
  - [1.2 The Neighbourhood Principle](#12-the-neighbourhood-principle)
  - [1.3 Instance-Based / Lazy Learning](#13-instance-based--lazy-learning)
  - [1.4 Non-Parametric Nature](#14-non-parametric-nature)
  - [1.5 What "Non-parametric" Does and Does Not Mean](#15-what-non-parametric-does-and-does-not-mean)
  - [1.6 Distance-Based Decision](#16-distance-based-decision)

---

## Introduction

- **K‑Nearest Neighbor (KNN)** is a simple and widely used machine learning technique for **classification** and **regression** tasks.
- It works by identifying the **K closest data points** to a given input and making predictions based on the **majority class** or **average value** of those neighbors.
- **KNN** is a **supervised machine learning algorithm** used for both **classification** and **regression** tasks.
- It predicts the label or value of a new data point by looking at the **K closest data points** in the training set.
- KNN is known as a **lazy learner** and a **non-parametric algorithm**.
- Instead of building a mathematical formula during a training phase, it simply stores the **entire dataset** in memory and performs all calculations when a prediction is requested.

KNN is:

- **supervised** because its neighbours have known targets;
- **non-parametric** because it does not assume a fixed finite-parameter form such as a line or Gaussian class distribution;
- **instance-based** because stored training instances directly participate in predictions;
- **lazy** because expensive model work is largely deferred until prediction time;
- **local** because a prediction is based on a neighbourhood around the query rather than one global equation.

## What is 'K' in K-Nearest Neighbour?

In the KNN algorithm, ***K is a number*** that tells the algorithm how many ***nearby points*** or ***neighbors*** to look at when it makes a ***decision***.

> [!IMPORTANT]
> ***K is a hyperparameter*** — a ***value you choose before training*** that ***controls the model's behavior***. ***Different values of K produce different results***, so ***selecting the right K*** is ***crucial for model performance***.

**Example:** Imagine you're deciding which fruit it is based on its shape and size. You compare it to fruits you already know.

- If ***k = 3***, the algorithm looks at the ***3 closest fruits*** to the new one.
- If ***2 of those 3 fruits*** are ***apples*** and ***1 is a banana***, the algorithm says the ***new fruit is an apple*** because ***most of its neighbors are apples***.

This principle is called ***majority voting*** and is fundamental to how KNN makes predictions.

## Key Concepts

# 2. Key vocabulary and notation

| Symbol | Meaning |
|---|---|
| $n$ | number of training examples |
| $d$ | number of features/dimensions |
| $C$ | number of classes |
| $\mathbf x_i$ | training feature vector $i$ |
| $y_i$ | label or numerical target for example $i$ |
| $\mathbf x$ | query point to predict |
| $k$ | number of neighbours used |
| $d(\mathbf x,\mathbf z)$ | distance between two feature vectors |
| $N_k(\mathbf x)$ | index set of the $k$ nearest training examples to $\mathbf x$ |
| $w_i(\mathbf x)$ | optional weight for neighbour $i$ |
| $\mathbb 1(A)$ | indicator: $1$ if statement $A$ is true, else $0$ |
| $\hat y(\mathbf x)$ | prediction for query $\mathbf x$

### 1.1 What KNN Learns During Fit

**Conceptually**, classical KNN does not estimate coefficients. **Fitting stores** training feature vectors and targets, plus any search index. Library implementations may build a **KD-tree** or **ball tree**, but this is an acceleration structure rather than a learned predictive equation.

### 1.2 The Neighbourhood Principle

KNN rests on a **smoothness assumption**:

> [!NOTE]
> ***Inputs that are close*** under a ***meaningful representation*** and ***metric*** tend to have ***similar targets***.

The qualification is essential. Geographic coordinates may make distance meaningful for house prices. Arbitrary customer identifiers do not. 

> [!WARNING]
> ***A KNN model cannot repair*** a ***representation*** in which ***irrelevant scale***, ***noise***, or ***poor encoding*** makes the ***wrong observations appear close***.

### 1.3 Instance-Based / Lazy Learning

**KNN is a lazy learner** — no model fitting beyond storing data. All computation happens at query time. This means:

- **Training is fast:** $O(nd)$ to store $n$ points in $d$ dimensions
- **Prediction is costly:** $O(nd)$ per query (computing distances to all training points)
- **Acceleration possible** with data structures like **KD-tree** or **ball tree** to reduce query-time complexity

### 1.4 Non-Parametric Nature

**KNN makes no strong assumptions** about the form of the decision boundary; the "model" grows with data. Key implications:

- **Can capture complex shapes:** Given enough data, KNN can learn arbitrarily complex decision boundaries
- **Risk of overfitting:** High if $k$ is too small — model can become too flexible and fit noise
- **Model grows with data:** Adding more training examples makes the decision surface more detailed

> [!CAUTION]
> ***Overfitting Risk:*** ***If K is too small*** (***especially K=1***), ***KNN becomes highly sensitive*** to ***noise*** and ***individual outliers***, leading to ***poor generalization*** on ***new data***.

### 1.5 What "Non-parametric" Does and Does Not Mean

**Non-parametric does not mean** that KNN has no settings. It has important hyperparameters: $k$, **distance metric**, **feature transformation**, **voting weights**, and **search algorithm**. The term means that **model capacity is not restricted to a fixed-size parameter vector** independent of the amount of data. Storing more examples can make the represented decision surface more detailed.

### 1.6 Distance-Based Decision

**The only inductive bias is that nearby points in feature space should have similar labels or outputs.** Therefore:

- **Choice of distance metric is crucial:** Different metrics (Euclidean, Manhattan, Cosine, etc.) can produce very different results
- **Feature scaling matters:** Unscaled features with different ranges can dominate the distance calculation
- **Representation is critical:** Poor feature encoding, irrelevant noise, or bad scaling makes the wrong observations appear close

> [!TIP]
> ***Best Practice:*** ***Always normalize or scale*** your ***features*** before using ***KNN***. ***Features with larger numerical ranges*** (***e.g., age in years vs. income in thousands***) can ***disproportionately influence*** distance calculations if ***not properly scaled***.

Given a labelled training set

$$
\mathcal D=\{(\mathbf x_i,y_i)\}_{i=1}^{n},\qquad \mathbf x_i\in\mathbb R^d,
$$

and a new query $\mathbf x$, KNN performs ***three conceptual operations***:

- ***measure*** the ***distance*** from $\mathbf x$ to ***training observations***;
- ***select*** the $k$ ***smallest distances***;
- ***combine*** the corresponding ***targets*** $y_i$.

---

<p align="center">
  <strong>Next:</strong> <a href="./01%20KNN%20Algorithm%20and%20Working%20Step-by-Step..md">01 KNN Algorithm and Working Step-by-Step</a> ➡️
</p>



