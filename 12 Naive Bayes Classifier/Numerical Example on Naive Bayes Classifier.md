# Numerical Example on Naive Bayes Classifier

This document walks through a **complete, step-by-step numerical example** of the Bernoulli Naive Bayes Classifier on a 14-email spam detection problem. Every probability — prior, likelihood, posterior — is computed by hand. The example demonstrates the **zero-frequency problem**, **Laplace smoothing**, and the **log-space** computation technique.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Problem Statement](#2-problem-statement)
3. [Training Dataset (14 Emails)](#3-training-dataset-14-emails)
4. [Step 1 — Class Prior Probabilities](#4-step-1--class-prior-probabilities)
5. [Step 2 — Count Feature Occurrences per Class](#5-step-2--count-feature-occurrences-per-class)
6. [Step 3 — Conditional Likelihood Tables (Without Smoothing)](#6-step-3--conditional-likelihood-tables-without-smoothing)
7. [Step 4 — Predict Test Email A: No Offer (Clean Case)](#7-step-4--predict-test-email-a-no-offer-clean-case)
8. [Step 5 — Predict Test Email B: Contains Offer (Zero-Frequency Problem)](#8-step-5--predict-test-email-b-contains-offer-zero-frequency-problem)
9. [Step 6 — The Zero-Frequency Problem Analysed](#9-step-6--the-zero-frequency-problem-analysed)
10. [Step 7 — Laplace Smoothing (α = 1)](#10-step-7--laplace-smoothing-α--1)
11. [Step 8 — Smoothed Likelihood Tables](#11-step-8--smoothed-likelihood-tables)
12. [Step 9 — Predict Test Email A (With Smoothing)](#12-step-9--predict-test-email-a-with-smoothing)
13. [Step 10 — Predict Test Email B (With Smoothing)](#13-step-10--predict-test-email-b-with-smoothing)
14. [Step 11 — Predict Test Email C (With Smoothing)](#14-step-11--predict-test-email-c-with-smoothing)
15. [Step 12 — Log-Space Computation](#15-step-12--log-space-computation)
16. [Step 13 — Full Results Summary](#16-step-13--full-results-summary)
17. [Key Takeaways](#17-key-takeaways)
18. [References](#18-references)

---

## 1. Prerequisites

Make sure you are comfortable with the following before continuing:

| # | Topic | Why It Matters |
|---|---|---|
| 1 | [Bayes' Theorem in Probability](Bayes%20Theorem%20in%20Probability.md) | The core formula this entire example is built on |
| 2 | [Conditional Probability](Conditional%20Probability.md) | What $P(A \mid B)$ means — used in every likelihood calculation |
| 3 | [Naive Bayes Classifier](Naive%20Bayes%20Classifier.md) | The full mathematical derivation and decision rule |
| 4 | [Types of Naive Bayes Classifier](Types%20of%20Naive%20Bayes%20Classifier.md) | Why we use **Bernoulli NB** for binary features |

---

## 2. Problem Statement

**Task:** Build an **email spam classifier** using the Bernoulli Naive Bayes Classifier.

Each email is described by **4 binary features** — whether or not it contains the words:

| Feature | Values | Description |
|---|---|---|
| **Free** | Yes / No | Does the email contain the word *"Free"*? |
| **Win** | Yes / No | Does the email contain the word *"Win"*? |
| **Offer** | Yes / No | Does the email contain the word *"Offer"*? |
| **Discount** | Yes / No | Does the email contain the word *"Discount"*? |

**Target classes:**

| Class | Label | Meaning |
|---|---|---|
| **Spam** | 1 | The email is unwanted spam |
| **Ham** | 0 | The email is legitimate |

**Test emails we will classify:**

| Email | Free | Win | Offer | Discount | True Class |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Test A** | Yes | Yes | No | No | Spam |
| **Test B** | Yes | Yes | Yes | No | Spam |
| **Test C** | No | No | Yes | Yes | Ham |

> [!NOTE]
> Test Email B is the most interesting case — it contains "Free" and "Win" (both strong spam indicators) but also "Offer" (which never appears in Spam training emails). This triggers the **zero-frequency problem**.

---

## 3. Training Dataset (14 Emails)

The training set has **14 emails** — 7 Spam and 7 Ham — each described by 4 binary features.

| Email | Free | Win | Offer | Discount | **Class** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| E01 | Yes | Yes | No | Yes | **Spam** |
| E02 | Yes | No | No | No | **Spam** |
| E03 | No | Yes | No | Yes | **Spam** |
| E04 | Yes | Yes | No | No | **Spam** |
| E05 | Yes | No | No | Yes | **Spam** |
| E06 | No | Yes | No | No | **Spam** |
| E07 | Yes | Yes | No | Yes | **Spam** |
| E08 | Yes | No | Yes | No | Ham |
| E09 | No | No | Yes | Yes | Ham |
| E10 | No | Yes | Yes | No | Ham |
| E11 | Yes | No | Yes | Yes | Ham |
| E12 | No | No | Yes | No | Ham |
| E13 | No | Yes | Yes | Yes | Ham |
| E14 | No | No | No | Yes | Ham |

> [!IMPORTANT]
> Notice that **"Offer" never appears in any Spam email** (E01–E07 all have Offer = No). This will create the **zero-frequency problem** in Step 5.

**Dataset summary:**

| Class | Count | Emails |
|---|---|---|
| **Spam** | 7 | E01, E02, E03, E04, E05, E06, E07 |
| **Ham** | 7 | E08, E09, E10, E11, E12, E13, E14 |
| **Total** | **14** | — |

---

## 4. Step 1 — Class Prior Probabilities

The **prior probability** $P(C_k)$ is the fraction of training examples that belong to class $C_k$:

$$\boxed{\widehat{P}(C_k) = \frac{N_k}{N}}$$

**Where:**
- $N_k$ = number of training examples in class $C_k$
- $N$ = total number of training examples

**Computing priors:**

$$\boxed{P(\text{Spam}) = \frac{N_\text{Spam}}{N} = \frac{7}{14} = 0.5}$$

$$\boxed{P(\text{Ham}) = \frac{N_\text{Ham}}{N} = \frac{7}{14} = 0.5}$$

| Class | Count | Prior |
|---|:---:|:---:|
| **Spam** | 7 | $7/14 = 0.5$ |
| **Ham** | 7 | $7/14 = 0.5$ |
| **Total** | **14** | $1.0$ |

> [!TIP]
> Both classes have equal prior probability (0.5 each) because the dataset is perfectly **balanced** — 7 Spam and 7 Ham. If the dataset were imbalanced, the priors would differ, and the more common class would get a head start in the scoring.

---

## 5. Step 2 — Count Feature Occurrences per Class

For each **feature** and each **class**, count how many training examples have that feature set to "Yes".

### 5.1 Counting for Spam (N_Spam = 7)

Go through E01–E07 and tally each feature:

| Email | Free | Win | Offer | Discount |
|:---:|:---:|:---:|:---:|:---:|
| E01 | **Yes** | **Yes** | No | **Yes** |
| E02 | **Yes** | No | No | No |
| E03 | No | **Yes** | No | **Yes** |
| E04 | **Yes** | **Yes** | No | No |
| E05 | **Yes** | No | No | **Yes** |
| E06 | No | **Yes** | No | No |
| E07 | **Yes** | **Yes** | No | **Yes** |
| **Count (Yes)** | **5** | **5** | **0** | **4** |
| **Count (No)** | **2** | **2** | **7** | **3** |

### 5.2 Counting for Ham (N_Ham = 7)

Go through E08–E14 and tally each feature:

| Email | Free | Win | Offer | Discount |
|:---:|:---:|:---:|:---:|:---:|
| E08 | **Yes** | No | **Yes** | No |
| E09 | No | No | **Yes** | **Yes** |
| E10 | No | **Yes** | **Yes** | No |
| E11 | **Yes** | No | **Yes** | **Yes** |
| E12 | No | No | **Yes** | No |
| E13 | No | **Yes** | **Yes** | **Yes** |
| E14 | No | No | No | **Yes** |
| **Count (Yes)** | **2** | **2** | **6** | **4** |
| **Count (No)** | **5** | **5** | **1** | **3** |

> [!WARNING]
> **⚠ Zero Count Detected**
>
> ***"Offer" appeared 0 times in Spam emails.*** This is the setup for the **zero-frequency problem** — any test email containing "Offer" will get $P(\text{Offer} = \text{Yes} \mid \text{Spam}) = 0$, which will **collapse the entire Spam score to zero**.

---

## 6. Step 3 — Conditional Likelihood Tables (Without Smoothing)

For each feature $X_i$ in each class $C_k$, the **conditional likelihood** is:

$$\boxed{P(X_i = v \mid C_k) = \frac{\text{Count}(X_i = v \text{ in } C_k)}{N_k}}$$

### 6.1 Likelihoods for Spam

| Feature | $P(\text{Yes} \mid \text{Spam})$ | Fraction | $P(\text{No} \mid \text{Spam})$ | Fraction |
|---|:---:|:---:|:---:|:---:|
| **Free** | $5/7$ | $\approx 0.7143$ | $2/7$ | $\approx 0.2857$ |
| **Win** | $5/7$ | $\approx 0.7143$ | $2/7$ | $\approx 0.2857$ |
| **Offer** | $\mathbf{0/7}$ | $\mathbf{= 0}$ | $7/7$ | $= 1.0000$ |
| **Discount** | $4/7$ | $\approx 0.5714$ | $3/7$ | $\approx 0.4286$ |

### 6.2 Likelihoods for Ham

| Feature | $P(\text{Yes} \mid \text{Ham})$ | Fraction | $P(\text{No} \mid \text{Ham})$ | Fraction |
|---|:---:|:---:|:---:|:---:|
| **Free** | $2/7$ | $\approx 0.2857$ | $5/7$ | $\approx 0.7143$ |
| **Win** | $2/7$ | $\approx 0.2857$ | $5/7$ | $\approx 0.7143$ |
| **Offer** | $6/7$ | $\approx 0.8571$ | $1/7$ | $\approx 0.1429$ |
| **Discount** | $4/7$ | $\approx 0.5714$ | $3/7$ | $\approx 0.4286$ |

---

## 7. Step 4 — Predict Test Email A: No Offer (Clean Case)

> **Test Email A:** Free = **Yes**, Win = **Yes**, Offer = **No**, Discount = **No**

This email has the classic spam-like keywords "Free" and "Win" but does **not** contain "Offer" — so no zero-frequency issue.

### The Naive Bayes Decision Rule

$$\boxed{\hat{y} = \arg\max_{C_k}\; P(C_k) \cdot \prod_{i=1}^{n} P(x_i \mid C_k)}$$

> [!NOTE]
> We are using **Bernoulli NB**, so we substitute the **actual feature value** (Yes or No) into each $P(x_i \mid C_k)$ term. For example, Free = Yes → use $P(\text{Free=Yes} \mid C_k)$; Offer = No → use $P(\text{Offer=No} \mid C_k)$.

### 7.1 Score for Spam

$$\text{Score(Spam)} = P(\text{Spam}) \times P(\text{Free=Yes} \mid \text{Spam}) \times P(\text{Win=Yes} \mid \text{Spam}) \times P(\text{Offer=No} \mid \text{Spam}) \times P(\text{Discount=No} \mid \text{Spam})$$

Substituting values:

$$= \frac{1}{2} \;\times\; \frac{5}{7} \;\times\; \frac{5}{7} \;\times\; \frac{7}{7} \;\times\; \frac{3}{7}$$

$$= \frac{1}{2} \;\times\; \frac{5 \times 5 \times 7 \times 3}{7^4}$$

$$= \frac{1}{2} \;\times\; \frac{525}{2401}$$

$$= \frac{1}{2} \;\times\; 0.21866 = \boxed{0.10933}$$

### 7.2 Score for Ham

$$\text{Score(Ham)} = P(\text{Ham}) \times P(\text{Free=Yes} \mid \text{Ham}) \times P(\text{Win=Yes} \mid \text{Ham}) \times P(\text{Offer=No} \mid \text{Ham}) \times P(\text{Discount=No} \mid \text{Ham})$$

Substituting values:

$$= \frac{1}{2} \;\times\; \frac{2}{7} \;\times\; \frac{2}{7} \;\times\; \frac{1}{7} \;\times\; \frac{3}{7}$$

$$= \frac{1}{2} \;\times\; \frac{2 \times 2 \times 1 \times 3}{7^4}$$

$$= \frac{1}{2} \;\times\; \frac{12}{2401}$$

$$= \frac{1}{2} \;\times\; 0.004998 = \boxed{0.002499}$$

### 7.3 Normalise to Get Posterior Probabilities

$$\text{Total} = 0.10933 + 0.002499 = 0.11183$$

$$P(\text{Spam} \mid \text{Email A}) = \frac{0.10933}{0.11183} = \boxed{97.8\%}$$

$$P(\text{Ham} \mid \text{Email A}) = \frac{0.002499}{0.11183} = \boxed{2.2\%}$$

### 7.4 Prediction

| Class | Unnormalised Score | Posterior Probability |
|---|:---:|:---:|
| **Spam** | **0.10933** | **97.8%** |
| Ham | 0.002499 | 2.2% |

**Prediction: Spam** ✓ — The classifier confidently identifies this as spam. "Free" and "Win" are both very common in Spam emails, while being rare in Ham. "Offer" is absent (which helps Spam since Offer=No is even more common in Spam). The evidence strongly favours Spam.

---

## 8. Step 5 — Predict Test Email B: Contains Offer (Zero-Frequency Problem)

> **Test Email B:** Free = **Yes**, Win = **Yes**, Offer = **Yes**, Discount = **No**

This email also contains "Free" and "Win" — yet it **also** contains "Offer". Since "Offer" was never observed in any Spam training email, this triggers the zero-frequency problem.

### 8.1 Score for Spam

$$\text{Score(Spam)} = \frac{1}{2} \;\times\; \frac{5}{7} \;\times\; \frac{5}{7} \;\times\; \underbrace{\frac{0}{7}}_{P(\text{Offer=Yes}|\text{Spam})} \;\times\; \frac{3}{7}$$

$$= \frac{1}{2} \;\times\; \frac{5 \times 5 \times \mathbf{0} \times 3}{7^4} = \boxed{0}$$

### 8.2 Score for Ham

$$\text{Score(Ham)} = \frac{1}{2} \;\times\; \frac{2}{7} \;\times\; \frac{2}{7} \;\times\; \frac{6}{7} \;\times\; \frac{3}{7}$$

$$= \frac{1}{2} \;\times\; \frac{2 \times 2 \times 6 \times 3}{7^4}$$

$$= \frac{1}{2} \;\times\; \frac{72}{2401} = \frac{1}{2} \;\times\; 0.02999 = \boxed{0.01499}$$

### 8.3 Prediction (Without Smoothing)

| Class | Unnormalised Score | Posterior Probability |
|---|:---:|:---:|
| **Spam** | **0** | **0%** |
| Ham | 0.01499 | 100% |

**Prediction: Ham** ✗ — Despite containing both "Free" and "Win", the classifier labels it as Ham. A single zero probability for "Offer=Yes in Spam" has **destroyed the entire Spam score**.

> [!IMPORTANT]
> This is the ***zero-frequency problem*** in action. Even though "Free" ($P = 5/7 \approx 71\%$) and "Win" ($P = 5/7 \approx 71\%$) are very strong Spam indicators, a **single zero** in the product makes the entire Spam score zero — no matter how strong the other evidence is.

---

## 9. Step 6 — The Zero-Frequency Problem Analysed

### Why Does One Zero Destroy Everything?

Naive Bayes computes a **product** of probabilities:

$$\text{Score}(C_k) = P(C_k) \;\times\; P(x_1 \mid C_k) \;\times\; P(x_2 \mid C_k) \;\times\; \cdots \;\times\; P(x_n \mid C_k)$$

If **any single factor is zero**, the entire product becomes zero — regardless of how large all other factors are:

$$0.5 \;\times\; \underbrace{\frac{5}{7}}_{0.714} \;\times\; \underbrace{\frac{5}{7}}_{0.714} \;\times\; \underbrace{0}_{P(\text{Offer=Yes}|\text{Spam})} \;\times\; \underbrace{\frac{3}{7}}_{0.429} = 0$$

This is **mathematically correct** given the raw counts — "Offer" never appeared in Spam. But it is **epistemologically wrong**: not seeing something in 7 training examples does not mean it is ***impossible***. It means it is ***unobserved*** — and we need a technique to handle this distinction.

### Summary of the Problem

| Condition | Probability |
|---|:---:|
| $P(\text{Offer=Yes} \mid \text{Spam})$ | $0/7 = \mathbf{0}$ |
| $P(\text{Spam} \mid \text{Email B})$ | $\mathbf{0\%}$ — entire Spam score zeroed |
| $P(\text{Ham} \mid \text{Email B})$ | $100\%$ — false certainty |

The classifier is forced into an **incorrect prediction with false confidence** — not because Ham is more likely, but because a single count-of-zero collapsed the Spam score.

**Solution: Laplace Smoothing.**

---

## 10. Step 7 — Laplace Smoothing (α = 1)

**Laplace smoothing** (also called *additive smoothing* or *add-one smoothing*) prevents zero probabilities by adding a small constant $\alpha$ to every count before computing probabilities.

### The Smoothed Likelihood Formula (Bernoulli NB)

For a **binary feature** $X_i \in \{0, 1\}$ with $\alpha = 1$ and $K = 2$ (two possible values: Yes / No):

$$\boxed{\widehat{P}(X_i = v \mid C_k) = \frac{\text{Count}(X_i = v \text{ in } C_k) + \alpha}{N_k + \alpha \cdot K}}$$

**Where:**

| Symbol | Meaning | Value in this example |
|---|---|---|
| $\text{Count}(X_i = v,\, C_k)$ | Times feature $X_i$ had value $v$ in class $C_k$ | Raw count from Step 2 |
| $N_k$ | Total training examples in class $C_k$ | 7 (for both classes) |
| $\alpha$ | Smoothing parameter | 1 (Laplace) |
| $K$ | Number of distinct values per feature | 2 (binary: Yes / No) |

**New denominator:**

$$N_k + \alpha \cdot K = 7 + 1 \times 2 = \mathbf{9}$$

> [!NOTE]
> **Why K = 2?** Each feature is binary — it can take exactly two values: Yes (1) or No (0). So $K = 2$. The denominator grows by $\alpha \cdot K = 2$ to keep all probabilities summing to 1.

> [!IMPORTANT]
> After adding $\alpha = 1$ to every count, no count can stay at zero. The smallest possible smoothed count is $0 + 1 = 1$, giving a probability of at least $1/9 \approx 0.111$ — ***never zero***.

---

## 11. Step 8 — Smoothed Likelihood Tables

### 11.1 Smoothed Likelihoods for Spam

$$\text{Smoothed denominator} = N_\text{Spam} + \alpha \cdot K = 7 + 1 \times 2 = 9$$

| Feature | Raw Count (Yes) | Smoothed Count (Yes) | $P(\text{Yes} \mid \text{Spam})$ | Smoothed Count (No) | $P(\text{No} \mid \text{Spam})$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Free** | 5 | $5 + 1 = 6$ | $6/9 \approx \mathbf{0.6667}$ | $2 + 1 = 3$ | $3/9 \approx 0.3333$ |
| **Win** | 5 | $5 + 1 = 6$ | $6/9 \approx \mathbf{0.6667}$ | $2 + 1 = 3$ | $3/9 \approx 0.3333$ |
| **Offer** | **0** | $0 + 1 = 1$ | $1/9 \approx \mathbf{0.1111}$ ← *fixed!* | $7 + 1 = 8$ | $8/9 \approx 0.8889$ |
| **Discount** | 4 | $4 + 1 = 5$ | $5/9 \approx \mathbf{0.5556}$ | $3 + 1 = 4$ | $4/9 \approx 0.4444$ |

### 11.2 Smoothed Likelihoods for Ham

$$\text{Smoothed denominator} = N_\text{Ham} + \alpha \cdot K = 7 + 1 \times 2 = 9$$

| Feature | Raw Count (Yes) | Smoothed Count (Yes) | $P(\text{Yes} \mid \text{Ham})$ | Smoothed Count (No) | $P(\text{No} \mid \text{Ham})$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Free** | 2 | $2 + 1 = 3$ | $3/9 \approx \mathbf{0.3333}$ | $5 + 1 = 6$ | $6/9 \approx 0.6667$ |
| **Win** | 2 | $2 + 1 = 3$ | $3/9 \approx \mathbf{0.3333}$ | $5 + 1 = 6$ | $6/9 \approx 0.6667$ |
| **Offer** | 6 | $6 + 1 = 7$ | $7/9 \approx \mathbf{0.7778}$ | $1 + 1 = 2$ | $2/9 \approx 0.2222$ |
| **Discount** | 4 | $4 + 1 = 5$ | $5/9 \approx \mathbf{0.5556}$ | $3 + 1 = 4$ | $4/9 \approx 0.4444$ |

### 11.3 Before vs. After Smoothing — Comparison

| Feature | $P(\text{Yes} \mid \text{Spam})$ Before | $P(\text{Yes} \mid \text{Spam})$ After | Change |
|---|:---:|:---:|---|
| Free | $5/7 \approx 0.7143$ | $6/9 \approx 0.6667$ | Slightly lower |
| Win | $5/7 \approx 0.7143$ | $6/9 \approx 0.6667$ | Slightly lower |
| **Offer** | $\mathbf{0/7 = 0}$ | $\mathbf{1/9 \approx 0.1111}$ | **Fixed — no longer zero!** |
| Discount | $4/7 \approx 0.5714$ | $5/9 \approx 0.5556$ | Slightly lower |

> [!TIP]
> Smoothing slightly **reduces** probabilities for features with high counts (from $5/7 \approx 0.714$ to $6/9 \approx 0.667$), and **increases** probabilities for features with low or zero counts. This is the trade-off: we prevent catastrophic zeros at the cost of a small amount of shrinkage in high-count probabilities.

---

## 12. Step 9 — Predict Test Email A (With Smoothing)

> **Test Email A:** Free = **Yes**, Win = **Yes**, Offer = **No**, Discount = **No**

### 12.1 Score for Spam (Smoothed)

$$\text{Score(Spam)} = \frac{1}{2} \;\times\; \frac{6}{9} \;\times\; \frac{6}{9} \;\times\; \frac{8}{9} \;\times\; \frac{4}{9}$$

$$= \frac{1}{2} \;\times\; \frac{6 \times 6 \times 8 \times 4}{9^4}$$

$$= \frac{1}{2} \;\times\; \frac{1152}{6561}$$

$$= \frac{1}{2} \;\times\; 0.17559 = \boxed{0.08779}$$

### 12.2 Score for Ham (Smoothed)

$$\text{Score(Ham)} = \frac{1}{2} \;\times\; \frac{3}{9} \;\times\; \frac{3}{9} \;\times\; \frac{2}{9} \;\times\; \frac{4}{9}$$

$$= \frac{1}{2} \;\times\; \frac{3 \times 3 \times 2 \times 4}{9^4}$$

$$= \frac{1}{2} \;\times\; \frac{72}{6561}$$

$$= \frac{1}{2} \;\times\; 0.010975 = \boxed{0.005487}$$

### 12.3 Posterior Probabilities (Smoothed)

$$\text{Total} = 0.08779 + 0.005487 = 0.09328$$

$$P(\text{Spam} \mid \text{Email A}) = \frac{0.08779}{0.09328} = \boxed{94.1\%}$$

$$P(\text{Ham} \mid \text{Email A}) = \frac{0.005487}{0.09328} = \boxed{5.9\%}$$

**Prediction: Spam** ✓ (still correct, slightly less confident than without smoothing: 94.1% vs 97.8%)

---

## 13. Step 10 — Predict Test Email B (With Smoothing)

> **Test Email B:** Free = **Yes**, Win = **Yes**, Offer = **Yes**, Discount = **No**

Now $P(\text{Offer=Yes} \mid \text{Spam}) = 1/9 \approx 0.111$ — **no longer zero!**

### 13.1 Score for Spam (Smoothed)

$$\text{Score(Spam)} = \frac{1}{2} \;\times\; \frac{6}{9} \;\times\; \frac{6}{9} \;\times\; \underbrace{\frac{1}{9}}_{P(\text{Offer=Yes}|\text{Spam})\text{ fixed!}} \;\times\; \frac{4}{9}$$

$$= \frac{1}{2} \;\times\; \frac{6 \times 6 \times 1 \times 4}{9^4}$$

$$= \frac{1}{2} \;\times\; \frac{144}{6561}$$

$$= \frac{1}{2} \;\times\; 0.021946 = \boxed{0.010973}$$

### 13.2 Score for Ham (Smoothed)

$$\text{Score(Ham)} = \frac{1}{2} \;\times\; \frac{3}{9} \;\times\; \frac{3}{9} \;\times\; \frac{7}{9} \;\times\; \frac{4}{9}$$

$$= \frac{1}{2} \;\times\; \frac{3 \times 3 \times 7 \times 4}{9^4}$$

$$= \frac{1}{2} \;\times\; \frac{252}{6561}$$

$$= \frac{1}{2} \;\times\; 0.038410 = \boxed{0.019205}$$

### 13.3 Posterior Probabilities (Smoothed)

$$\text{Total} = 0.010973 + 0.019205 = 0.030178$$

$$P(\text{Spam} \mid \text{Email B}) = \frac{0.010973}{0.030178} = \boxed{36.4\%}$$

$$P(\text{Ham} \mid \text{Email B}) = \frac{0.019205}{0.030178} = \boxed{63.6\%}$$

**Prediction: Ham**

### 13.4 Comparing Without vs. With Smoothing

| | Without Smoothing | With Laplace Smoothing |
|---|:---:|:---:|
| **Score(Spam)** | $\mathbf{0}$ — Spam is impossible | **0.010973** |
| **Score(Ham)** | 0.01499 | 0.019205 |
| $P(\text{Spam} \mid \text{Email B})$ | **0%** | **36.4%** |
| $P(\text{Ham} \mid \text{Email B})$ | **100%** | **63.6%** |
| **Prediction** | Ham (false certainty) | Ham (honest uncertainty) |

> [!TIP]
> Smoothing did **not** flip the prediction — the email is still classified as Ham. But the Spam probability jumped from **0% → 36.4%**, which is a far more honest reflection of the evidence: "Free" and "Win" are strongly present (spam signals), and only the higher prior likelihood of "Offer" in Ham tips the balance. A real spam filter would treat 36.4% as a borderline case requiring caution.

---

## 14. Step 11 — Predict Test Email C (With Smoothing)

> **Test Email C:** Free = **No**, Win = **No**, Offer = **Yes**, Discount = **Yes**

This email lacks both "Free" and "Win" (the main spam keywords) and contains "Offer" and "Discount", which appear more often in Ham. We expect it to classify as Ham.

### 14.1 Score for Spam (Smoothed)

$$\text{Score(Spam)} = \frac{1}{2} \;\times\; \underbrace{\frac{3}{9}}_{P(\text{Free=No}|\text{Spam})} \;\times\; \underbrace{\frac{3}{9}}_{P(\text{Win=No}|\text{Spam})} \;\times\; \underbrace{\frac{1}{9}}_{P(\text{Offer=Yes}|\text{Spam})} \;\times\; \underbrace{\frac{5}{9}}_{P(\text{Discount=Yes}|\text{Spam})}$$

$$= \frac{1}{2} \;\times\; \frac{3 \times 3 \times 1 \times 5}{9^4}$$

$$= \frac{1}{2} \;\times\; \frac{45}{6561}$$

$$= \frac{1}{2} \;\times\; 0.006860 = \boxed{0.003430}$$

### 14.2 Score for Ham (Smoothed)

$$\text{Score(Ham)} = \frac{1}{2} \;\times\; \underbrace{\frac{6}{9}}_{P(\text{Free=No}|\text{Ham})} \;\times\; \underbrace{\frac{6}{9}}_{P(\text{Win=No}|\text{Ham})} \;\times\; \underbrace{\frac{7}{9}}_{P(\text{Offer=Yes}|\text{Ham})} \;\times\; \underbrace{\frac{5}{9}}_{P(\text{Discount=Yes}|\text{Ham})}$$

$$= \frac{1}{2} \;\times\; \frac{6 \times 6 \times 7 \times 5}{9^4}$$

$$= \frac{1}{2} \;\times\; \frac{1260}{6561}$$

$$= \frac{1}{2} \;\times\; 0.19205 = \boxed{0.09602}$$

### 14.3 Posterior Probabilities (Smoothed)

$$\text{Total} = 0.003430 + 0.09602 = 0.099450$$

$$P(\text{Spam} \mid \text{Email C}) = \frac{0.003430}{0.099450} = \boxed{3.4\%}$$

$$P(\text{Ham} \mid \text{Email C}) = \frac{0.09602}{0.099450} = \boxed{96.6\%}$$

**Prediction: Ham** ✓ — The classifier is very confident. Absent "Free" and "Win" (the main spam signals) and the presence of "Offer" (which is very common in Ham: 7/9 ≈ 78%) together make Ham overwhelmingly more likely.

---

## 15. Step 12 — Log-Space Computation

### Why Log-Space?

When multiplying many probabilities (each between 0 and 1), the product becomes extremely small and can **underflow to zero** in floating-point arithmetic. For example:

$$0.5 \;\times\; 0.667 \;\times\; 0.667 \;\times\; 0.111 \;\times\; 0.444 = 0.01097$$

This is still representable — but with hundreds or thousands of features, the product becomes `0.0` on any machine.

**Solution:** Take the logarithm. Since $\log$ is strictly increasing, the class with the highest score is unchanged:

$$\boxed{\hat{y} = \arg\max_k \underbrace{P(C_k) \prod_i P(x_i \mid C_k)}_{\text{product}} = \arg\max_k \underbrace{\Bigl[\log P(C_k) + \sum_i \log P(x_i \mid C_k)\Bigr]}_{\text{sum of logs}}}$$

**Multiplication becomes addition** — far more numerically stable.

### Reference: Natural Logarithm Values Used

| Fraction | Decimal | $\ln(\text{value})$ |
|---|:---:|:---:|
| $1/9$ | 0.1111 | $-2.1972$ |
| $2/9$ | 0.2222 | $-1.5041$ |
| $3/9$ | 0.3333 | $-1.0986$ |
| $4/9$ | 0.4444 | $-0.8109$ |
| $5/9$ | 0.5556 | $-0.5878$ |
| $6/9$ | 0.6667 | $-0.4055$ |
| $7/9$ | 0.7778 | $-0.2513$ |
| $8/9$ | 0.8889 | $-0.1178$ |
| $1/2$ | 0.5000 | $-0.6931$ |

### 15.1 Log Scores for Test Email A (Free=Yes, Win=Yes, Offer=No, Discount=No)

**Spam log score:**

$$g(\text{Spam} \mid A) = \ln(0.5) + \ln\!\left(\tfrac{6}{9}\right) + \ln\!\left(\tfrac{6}{9}\right) + \ln\!\left(\tfrac{8}{9}\right) + \ln\!\left(\tfrac{4}{9}\right)$$

$$= (-0.6931) + (-0.4055) + (-0.4055) + (-0.1178) + (-0.8109) = \boxed{-2.4328}$$

**Ham log score:**

$$g(\text{Ham} \mid A) = \ln(0.5) + \ln\!\left(\tfrac{3}{9}\right) + \ln\!\left(\tfrac{3}{9}\right) + \ln\!\left(\tfrac{2}{9}\right) + \ln\!\left(\tfrac{4}{9}\right)$$

$$= (-0.6931) + (-1.0986) + (-1.0986) + (-1.5041) + (-0.8109) = \boxed{-5.2053}$$

**Decision:** $g(\text{Spam} \mid A) = -2.4328 \;\;>\;\; g(\text{Ham} \mid A) = -5.2053$

→ **Spam wins** ✓

### 15.2 Log Scores for Test Email B (Free=Yes, Win=Yes, Offer=Yes, Discount=No)

**Spam log score:**

$$g(\text{Spam} \mid B) = \ln(0.5) + \ln\!\left(\tfrac{6}{9}\right) + \ln\!\left(\tfrac{6}{9}\right) + \ln\!\left(\tfrac{1}{9}\right) + \ln\!\left(\tfrac{4}{9}\right)$$

$$= (-0.6931) + (-0.4055) + (-0.4055) + (-2.1972) + (-0.8109) = \boxed{-4.5122}$$

**Ham log score:**

$$g(\text{Ham} \mid B) = \ln(0.5) + \ln\!\left(\tfrac{3}{9}\right) + \ln\!\left(\tfrac{3}{9}\right) + \ln\!\left(\tfrac{7}{9}\right) + \ln\!\left(\tfrac{4}{9}\right)$$

$$= (-0.6931) + (-1.0986) + (-1.0986) + (-0.2513) + (-0.8109) = \boxed{-3.9525}$$

**Decision:** $g(\text{Spam} \mid B) = -4.5122 \;\;<\;\; g(\text{Ham} \mid B) = -3.9525$

→ **Ham wins**

### 15.3 Log Scores for Test Email C (Free=No, Win=No, Offer=Yes, Discount=Yes)

**Spam log score:**

$$g(\text{Spam} \mid C) = \ln(0.5) + \ln\!\left(\tfrac{3}{9}\right) + \ln\!\left(\tfrac{3}{9}\right) + \ln\!\left(\tfrac{1}{9}\right) + \ln\!\left(\tfrac{5}{9}\right)$$

$$= (-0.6931) + (-1.0986) + (-1.0986) + (-2.1972) + (-0.5878) = \boxed{-5.6753}$$

**Ham log score:**

$$g(\text{Ham} \mid C) = \ln(0.5) + \ln\!\left(\tfrac{6}{9}\right) + \ln\!\left(\tfrac{6}{9}\right) + \ln\!\left(\tfrac{7}{9}\right) + \ln\!\left(\tfrac{5}{9}\right)$$

$$= (-0.6931) + (-0.4055) + (-0.4055) + (-0.2513) + (-0.5878) = \boxed{-2.3432}$$

**Decision:** $g(\text{Spam} \mid C) = -5.6753 \;\;<\;\; g(\text{Ham} \mid C) = -2.3432$

→ **Ham wins** ✓

### 15.4 Recovering Probabilities from Log Scores (Optional)

When you need the actual posterior probabilities (not just the winning class), use the **log-sum-exp trick** to avoid overflow:

$$\text{LSE} = m + \ln\!\left(e^{g_\text{Spam} - m} + e^{g_\text{Ham} - m}\right), \qquad m = \max(g_\text{Spam},\, g_\text{Ham})$$

**For Test Email B:**

$$m = \max(-4.5122,\; -3.9525) = -3.9525$$

$$\text{LSE} = -3.9525 + \ln\!\left(e^{-4.5122 - (-3.9525)} + e^{0}\right)$$

$$= -3.9525 + \ln\!\left(e^{-0.5597} + 1\right)$$

$$= -3.9525 + \ln(0.5714 + 1.000) = -3.9525 + \ln(1.5714)$$

$$= -3.9525 + 0.4520 = -3.5005$$

$$P(\text{Spam} \mid B) = e^{g_\text{Spam} - \text{LSE}} = e^{-4.5122 - (-3.5005)} = e^{-1.0117} \approx 0.364 = \boxed{36.4\%}$$

$$P(\text{Ham} \mid B) = e^{g_\text{Ham} - \text{LSE}} = e^{-3.9525 - (-3.5005)} = e^{-0.4520} \approx 0.636 = \boxed{63.6\%}$$

These match the direct calculation in Step 10 exactly. ✓

---

## 16. Step 13 — Full Results Summary

### 16.1 All Log Scores

| Test Email | $g(\text{Spam})$ | $g(\text{Ham})$ | Winner |
|:---:|:---:|:---:|:---:|
| **A** (Free, Win, no Offer, no Disc) | $-2.4328$ | $-5.2053$ | **Spam** |
| **B** (Free, Win, Offer, no Disc) | $-4.5122$ | $-3.9525$ | **Ham** |
| **C** (no Free, no Win, Offer, Disc) | $-5.6753$ | $-2.3432$ | **Ham** |

### 16.2 All Posterior Probabilities

| Test Email | Features | $P(\text{Spam})$ (no smooth) | $P(\text{Spam})$ (smoothed) | $P(\text{Ham})$ (smoothed) | Prediction |
|:---:|---|:---:|:---:|:---:|:---:|
| **A** | Free=Y, Win=Y, Offer=N, Disc=N | 97.8% | 94.1% | 5.9% | **Spam** ✓ |
| **B** | Free=Y, Win=Y, Offer=Y, Disc=N | **0%** (zero-freq!) | 36.4% | 63.6% | **Ham** |
| **C** | Free=N, Win=N, Offer=Y, Disc=Y | **0%** (zero-freq!) | 3.4% | 96.6% | **Ham** ✓ |

### 16.3 Complete Likelihood Reference Table (Smoothed)

| Feature | Value | $P(\cdot \mid \text{Spam})$ | $P(\cdot \mid \text{Ham})$ |
|---|:---:|:---:|:---:|
| **Free** | Yes | $6/9 \approx 0.6667$ | $3/9 \approx 0.3333$ |
| **Free** | No | $3/9 \approx 0.3333$ | $6/9 \approx 0.6667$ |
| **Win** | Yes | $6/9 \approx 0.6667$ | $3/9 \approx 0.3333$ |
| **Win** | No | $3/9 \approx 0.3333$ | $6/9 \approx 0.6667$ |
| **Offer** | Yes | $1/9 \approx 0.1111$ | $7/9 \approx 0.7778$ |
| **Offer** | No | $8/9 \approx 0.8889$ | $2/9 \approx 0.2222$ |
| **Discount** | Yes | $5/9 \approx 0.5556$ | $5/9 \approx 0.5556$ |
| **Discount** | No | $4/9 \approx 0.4444$ | $4/9 \approx 0.4444$ |

> [!NOTE]
> **Discount is a neutral feature.** $P(\text{Discount=Yes} \mid \text{Spam}) = P(\text{Discount=Yes} \mid \text{Ham}) = 5/9 \approx 0.556$. It carries **no discriminative power** for this dataset. In a real classifier, such features can safely be dropped without affecting accuracy.

---

## 17. Key Takeaways

1. **Prior probabilities** reflect class frequencies in training data. Equal priors (0.5 each) mean the features alone determine the classification.

2. **Likelihood** $P(x_i \mid C_k)$ is estimated by counting feature occurrences per class. For Bernoulli NB with binary features: $\text{count} / N_k$.

3. **Posterior = Prior × Product of Likelihoods** (unnormalised). Divide by the sum across all classes to normalise.

4. **The zero-frequency problem** occurs when a feature is never observed in a class during training. The resulting probability is exactly zero, which collapses the entire class score — no matter how strong the other evidence is.

5. **Laplace smoothing** (add-one smoothing, $\alpha = 1$) adds a pseudo-count of 1 to every feature count. The denominator increases by $\alpha \cdot K$ (here: $2$) to maintain valid probabilities. This guarantees ***no probability is ever exactly zero***.

6. **Smoothing can change confidence levels but not necessarily the prediction.** In Test Email B, the Spam probability rose from 0% → 36.4%, but Ham was still predicted — because "Offer" is genuinely much more common in Ham.

7. **Log-space computation** replaces the product of small probabilities with a sum of log-probabilities. The class with the highest log-score wins. This prevents numerical underflow and produces identical predictions.

8. **Neutral features** (where $P(x_i \mid C_k)$ is equal across all classes) contribute no information and can be safely ignored.

---

## 18. References

### Within This Module

| Document | What You Will Find |
|---|---|
| [Naive Bayes Classifier](Naive%20Bayes%20Classifier.md) | Full mathematical derivation, Laplace smoothing, log-space form |
| [Types of Naive Bayes Classifier](Types%20of%20Naive%20Bayes%20Classifier.md) | Bernoulli vs. Multinomial vs. Gaussian NB — when to use each |
| [Bayes Theorem in Probability](Bayes%20Theorem%20in%20Probability.md) | The theorem this classifier is built on |
| [Problem Based upon Bayes Theorem](Problem%20Based%20upon%20Bayes%20Theorem.md) | Simpler worked examples applying Bayes' Theorem |
| [Intuition Behind Naive Bayes Classifier](Intuition%20Behind%20Naive%20Bayes%20Classifier.md) | Conceptual understanding before the math |

### Code Examples

| Notebook | What You Will Find |
|---|---|
| [Simple Code Example](Simple%20Code%20Example%20of%20Naive%20Bayes%20Classifier.ipynb) | Concise sklearn implementation |
| [Detailed Step-by-Step Code](Simple%20Code%20Example%20of%20Naive%20Bayes%20Classifier%20%28Detailed%20Each%20Step%29.ipynb) | Fully annotated from-scratch implementation |
| [In-Depth Analysis](Naive-Bayes%20classifier%20In-Depth.ipynb) | Gaussian, Multinomial, Bernoulli NB with real datasets |

---

<p align="center">
  ⬅️ <strong>Previous:</strong> <a href="Naive%20Bayes%20Classifier.md">Naive Bayes Classifier</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>Next:</strong> <a href="Types%20of%20Naive%20Bayes%20Classifier.md">Types of Naive Bayes Classifier</a> ➡️
</p>
