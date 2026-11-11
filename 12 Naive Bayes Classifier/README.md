# Naive Bayes Classifier

Naive Bayes is a family of probabilistic classifiers built entirely on **Bayes' Theorem** with one strong assumption: every feature is **conditionally independent** given the class label. Despite this "naive" assumption, it works surprisingly well in practice — particularly for text classification, spam filtering, and medical diagnosis — and serves as an excellent foundation for understanding probabilistic machine learning.

This module builds the classifier from the ground up:

1. **Probability foundations** — conditional probability, independent events, and mutually exclusive events lay the mathematical groundwork.
2. **Bayes' Theorem** — deriving and applying the theorem that drives the classifier, reinforced with worked problems.
3. **The classifier itself** — intuition, full mathematical derivation, and implementation from scratch.
4. **Practical extensions** — handling numerical features with Gaussian Naive Bayes.

---

## Folder Structure

| # | Type | File | Description |
|---|---|---|---|
| 01 | 📝 Notes | [Conditional Probability](./Conditional%20Probability.md) | ***Conditional probability*** *definition*, **formula**, and **key properties** |
| 02 | 📝 Notes | [Independent and Dependent Events in Probability](./Independent%20and%20dependent%20Events%20in%20Probability.md) | ***Independence*** and ***dependence***, **multiplication rules**, *with vs without replacement* |
| 03 | 📝 Notes | [Mutually Exclusive Events in Probability](./Mutually%20Exclusive%20Events%20in%20Probability.md) | ***Mutually exclusive*** vs **independent events**, **addition rule**, and *Venn diagram* intuition |
| 04 | 📝 Notes | [Bayes Theorem in Probability](./Bayes%20Theorem%20in%20Probability.md) | ***Prior***, ***likelihood***, ***posterior***, ***evidence*** — **full derivation** and intuition behind ***Bayes' Theorem*** |
| 05 | 📝 Notes | [Problem Based upon Bayes' Theorem](./Problem%20Based%20upon%20Bayes%20Theorem.md) | **Worked problems** applying ***Bayes' Theorem***: *medical tests*, *spam*, and *classic urn problems* |
| 06 | 📝 Notes | [Intuition Behind Naive Bayes Classifier](./Intuition%20Behind%20Naive%20Bayes%20Classifier.md) | Why the ***conditional independence assumption*** *works in practice*; **spam and text classification** intuition |
| 07 | 📝 Notes | [Naive Bayes Classifier](./Naive%20Bayes%20Classifier.md) | **Full derivation**: ***Bayes' Theorem*** → **chain rule** → ***independence assumption*** → **decision rule** → **parameter estimation** |
| 08 | 📝 Notes | [Types of Naive Bayes Classifier](./Types%20of%20Naive%20Bayes%20Classifier.md) | ***Gaussian***, ***Multinomial***, ***Bernoulli***, ***Categorical***, and ***Complement*** Naive Bayes — **likelihood formulas**, **parameter estimation**, and *when to use each* |
| 09 | 📝 Notes | [Numerical Example on Naive Bayes Classifier](./Numerical%20Example%20on%20Naive%20Bayes%20Classifier.md) | **Step-by-step worked example**: *prior estimation*, *likelihood calculation*, and *posterior prediction* applied to a concrete dataset |
| 10 | 📓 Notebook | [Simple Code Example of Naive Bayes Classifier](./Simple%20Code%20Example%20of%20Naive%20Bayes%20Classifier.ipynb) | **End-to-end** notebook: build and train ***Naive Bayes*** *from scratch*, compare with *sklearn*, *visualize results* |
| 11 | 📓 Notebook | [Simple Code Example of Naive Bayes Classifier (Detailed Each Step)](./Simple%20Code%20Example%20of%20Naive%20Bayes%20Classifier%20%28Detailed%20Each%20Step%29.ipynb) | **Step-by-step** *from-scratch* notebook: **explicit** *prior*, *likelihood*, and *posterior* computation at each stage with *zero-frequency handling* |
| 12 | 📓 Notebook | [Naive-Bayes Classifier In-Depth](./Naive-Bayes%20classifier%20In-Depth.ipynb) | **In-depth** ***sklearn*** notebook: *Gaussian Naive Bayes* on the ***Iris*** dataset, **decision boundaries**, **confusion matrix**, and *class probability visualization* |

---

## Reference Documentation

The following documents from this repository cover the foundational mathematics used throughout this module.

### Mathematics of Machine Learning

| Concept | Reference | Used in this module |
|---|---|---|
| Conditional Probability | [Conditional Probability](../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/README.md) | $P(C \mid \mathbf{x})$ — the posterior probability the classifier maximizes (files 06–09) |
| Bayes' Theorem | [Bayes' Theorem](../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/README.md) | Core theorem: $P(C \mid \mathbf{x}) \propto P(\mathbf{x} \mid C)\,P(C)$ (files 04–09) |

---

## Topics Covered

### Probability Foundations
- Conditional probability: $P(A \mid B) = P(A \cap B) / P(B)$ and its properties
- Independent events: $P(A \cap B) = P(A) \cdot P(B)$ and the independence test
- Mutually exclusive events: $P(A \cap B) = 0$ and the addition rule $P(A \cup B) = P(A) + P(B)$
- Key distinction: independent events can co-occur; mutually exclusive events cannot

### Bayes' Theorem
- Derivation from conditional probability: prior × likelihood / evidence
- Posterior update: how new evidence shifts the probability estimate
- Applied problems: disease testing, spam classification, urn problems

### The Naive Bayes Classifier
- The "naive" conditional independence assumption: $P(\mathbf{x} \mid C) = \prod_i P(x_i \mid C)$
- Why this assumption, despite being wrong, leads to strong classifiers
- Three variants: **Gaussian** (continuous), **Multinomial** (counts), **Bernoulli** (binary)
- MAP decision rule: $\hat{C} = \arg\max_C P(C) \prod_i P(x_i \mid C)$
- Log-likelihood form to avoid numerical underflow
- Laplace (add-1) smoothing for zero-frequency features

### Numerical Features — Gaussian Naive Bayes
- Modeling each feature with a per-class Gaussian: $P(x_i \mid C) = \mathcal{N}(\mu_{iC},\, \sigma^2_{iC})$
- Parameter estimation: sample mean and variance per class
- Decision boundary geometry and comparison with LDA
