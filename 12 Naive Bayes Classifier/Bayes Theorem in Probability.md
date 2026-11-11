# Bayes' Theorem in Probability

**Bayes' theorem** (also called **Bayes' law** or **Bayes' rule**) is one of the most important results in probability.
- It determines the **conditional probability** of event $A$ when event $B$ has already occurred.
- It gives a mathematical way to **invert** conditional probabilities — to find the probability of a **cause** given its **effect**.
- It is named after the Reverend **Thomas Bayes** (c. 1701–1761) and was developed independently by **Pierre-Simon Laplace**.
- It is the **mathematical backbone** of the Naive Bayes Classifier, Bayesian inference, spam filters, and medical diagnostic models.

---

## Table of Contents

1. [The Formula at a Glance](#1-the-formula-at-a-glance)
2. [Statement of the Theorem](#2-statement-of-the-theorem)
3. [Derivation](#3-derivation)
4. [The Core Intuition — Updating Beliefs](#4-the-core-intuition--updating-beliefs)
5. [Key Terms](#5-key-terms)
   - [Hypotheses](#hypotheses)
   - [Prior Probability](#prior-probability)
   - [Likelihood](#likelihood)
   - [Posterior Probability](#posterior-probability)
   - [Evidence](#evidence)
   - [Joint Probability](#joint-probability)
6. [The Extended Form — Law of Total Probability](#6-the-extended-form--law-of-total-probability)
7. [Useful Forms](#7-useful-forms)
   - [Proportional Form](#proportional-form)
   - [Odds Form and Bayes Factor](#odds-form-and-bayes-factor)
   - [Continuous Random Variables](#continuous-random-variables)
8. [Worked Examples](#8-worked-examples)
   - [Example 1 — Medical Diagnosis](#example-1--medical-diagnosis)
   - [Example 2 — Drug Testing and the Base-Rate Fallacy](#example-2--drug-testing-and-the-base-rate-fallacy)
   - [Example 3 — Bent Coins](#example-3--bent-coins)
9. [Bayesian vs Frequentist Interpretations](#9-bayesian-vs-frequentist-interpretations)
10. [Generalisation to Three Events](#10-generalisation-to-three-events)
11. [Practice Problems](#11-practice-problems)
12. [Quick Revision — Important Formulas](#12-quick-revision--important-formulas)
13. [Connection to Naive Bayes Classifier](#13-connection-to-naive-bayes-classifier)
14. [References](#14-references)

---

## 1. The Formula at a Glance

For two events $A$ and $B$ with $P(B) \neq 0$:

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/images/bayes-formula-anatomy.svg" alt="Anatomy of Bayes' theorem: posterior, likelihood, prior, evidence colour-coded" />
</p>

| Term | Name | Meaning |
|---|---|---|
| $P(A \mid B)$ | **Posterior** | Updated probability of hypothesis $A$ *after* seeing evidence $B$ |
| $P(B \mid A)$ | **Likelihood** | Probability of observing evidence $B$ *if* $A$ is true |
| $P(A)$ | **Prior** | Initial probability of $A$ *before* any evidence |
| $P(B)$ | **Evidence** (marginal likelihood) | Total probability of observing $B$ across all hypotheses; normalising constant |

---

## 2. Statement of the Theorem

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}, \qquad P(B) \neq 0$$

where $A$ and $B$ are events.

- $P(A \mid B)$ — **conditional probability**: the probability of $A$ occurring *given that* $B$ is true. This is the ***posterior*** probability.
- $P(B \mid A)$ — also **conditional**: the probability of $B$ *given that* $A$ is true. Viewed as a function of $A$ for fixed $B$, it is the ***likelihood***.
- $P(A)$ and $P(B)$ — the **marginal** (unconditioned) probabilities of each event on its own.

---

## 3. Derivation

Bayes' theorem follows directly from the definition of conditional probability.

The **joint probability** $P(A \cap B)$ can be written **two equivalent ways**:

$$P(A \cap B) = P(A \mid B) \cdot P(B)$$

$$P(A \cap B) = P(B \mid A) \cdot P(A)$$

Since both expressions equal $P(A \cap B)$, they are equal to each other:

$$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Dividing both sides by $P(B)$ (assuming $P(B) \neq 0$):

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

The diagram below shows this **geometrically** — the overlap of regions $A$ and $B$ is the same whether computed "through $A$" or "through $B$", and equating the two routes is exactly Bayes' rule.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/images/image.png" alt="Visual proof of Bayes' theorem using overlapping Venn regions" width="480"/>
</p>

---

## 4. The Core Intuition — Updating Beliefs

The single most important idea: **Bayes' theorem tells you how to change your mind when you get new data.**

You start with a ***prior*** belief, observe some ***evidence***, and end with a revised ***posterior*** belief.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/images/prior-posterior-update.svg" alt="Bayesian updating pipeline: prior + likelihood + evidence → posterior" />
</p>

| Perspective | Description |
|---|---|
| **Updating beliefs** | Formalises how a rational observer moves from prior to posterior as evidence accumulates. Today's posterior becomes tomorrow's prior. |
| **Flipping conditional probabilities** | Lets you compute $P(A \mid B)$ when you can only easily measure the reverse $P(B \mid A)$ — e.g. "probability of disease given a positive test" from "probability of a positive test given disease." |
| **Weighing evidence against base rates** | Strong evidence can still leave a hypothesis unlikely if the hypothesis was very rare to begin with (see §8.2). |

---

## 5. Key Terms

### Hypotheses

A ***hypothesis*** is a possible event or outcome being evaluated.

- In Bayes' theorem, hypotheses are written as $A_1, A_2, \ldots, A_n$, where each represents a distinct scenario or cause.
- Must be **mutually exclusive** and **exhaustive** — exactly one must be true.
- **Example:** *"The patient has disease A"*, *"has disease B"*, or *"is healthy".*

### Prior Probability

The ***prior probability*** $P(A)$ is the **initial probability** assigned to a hypothesis *before* any new evidence is observed.

- Represents **pre-existing knowledge** or **belief** about how likely a hypothesis is.
- Also called the **"a priori" probability** (from Latin: *"from what comes before"*).
- **Example:** The *prevalence* of a disease in the general population — how many people have it *before* any test is run.

### Likelihood

The ***likelihood*** $P(B \mid A)$ is the probability of observing evidence $B$ *if* hypothesis $A$ is true.

- Tells you how well the hypothesis **explains the observed evidence**.
- A high likelihood means the evidence is very compatible with the hypothesis.
- **Example:** The sensitivity of a medical test — $P(\text{positive test} \mid \text{has disease})$.

### Posterior Probability

The ***posterior probability*** $P(A \mid B)$ is the **updated probability** of a hypothesis *after* incorporating new evidence via Bayes' theorem.

- The *output* of Bayes' theorem — the **revised belief** once evidence has been factored in.
- Also called the **"a posteriori" probability** (from Latin: *"from what comes after"*).
- **Example:** The probability that a patient *has a disease* given that their **test result was positive**.

### Evidence

The ***evidence*** $P(B)$ is the **total probability** of observing evidence $B$ across all possible hypotheses.

- Acts as a **normalising constant** so that posterior probabilities sum to 1.
- Often computed using the **Law of Total Probability** (see §6).
- **Example:** The total probability of a positive test result, regardless of whether the patient has the disease.

### Joint Probability

The ***joint probability*** $P(A \cap B)$ is the probability of **two events occurring together** — simultaneously.

- $P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$ — equating these two expressions **is** the derivation of Bayes' rule.
- **Example:** The probability that a randomly chosen person is *both* a smoker *and* has lung disease.

> [!NOTE]
> **Numerical example — rolling a die:**
>
> | Event | Description | Outcomes | Probability |
> |:---:|---|:---:|:---:|
> | $B$ | Roll an **even** number | $\{2, 4, 6\}$ | $P(B) = 1/2$ |
> | $C$ | Roll a number **> 3** | $\{4, 5, 6\}$ | $P(C) = 1/2$ |
> | $B \cap C$ | **Even AND > 3** | $\{4, 6\}$ | $P(B \cap C) = 1/3$ |
>
> Only $\{4, 6\}$ satisfy both conditions — that is the intersection $B \cap C$.

---

## 6. The Extended Form — Law of Total Probability

Often $P(B)$ cannot be measured directly, but it can be decomposed.

If $A_1, A_2, \ldots, A_n$ are **mutually exclusive and exhaustive** (exactly one must occur), the **law of total probability** gives:

$$P(B) = \sum_{i} P(B \mid A_i) \cdot P(A_i)$$

Substituting into the denominator yields the **extended form**:

$$\boxed{P(A_i \mid B) = \frac{P(A_i) \cdot P(B \mid A_i)}{\displaystyle\sum_{j} P(A_j) \cdot P(B \mid A_j)}}$$

The most common case uses a hypothesis $A$ and its complement $\lnot A$ — which always form a mutually exclusive, exhaustive pair:

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid \lnot A) \cdot P(\lnot A)}}$$

This **two-term version** is the workhorse for most real applications (medical tests, spam detection, urn problems).

---

## 7. Useful Forms

### Proportional Form

When evidence $P(B)$ is fixed and we vary only the hypothesis, Bayes' theorem reduces to:

$$\boxed{P(A \mid B) \propto P(A) \cdot P(B \mid A)}$$

> [!IMPORTANT]
> **Posterior $\propto$ Prior $\times$ Likelihood**

The denominator $P(B)$ is just a normalising constant. This proportional form is used throughout Bayesian inference and is the basis of the Naive Bayes Classifier.

### Odds Form and Bayes Factor

Writing Bayes' theorem for both $A$ and $\lnot A$ and dividing, $P(B)$ cancels:

$$\boxed{\underbrace{\frac{P(A \mid B)}{P(\lnot A \mid B)}}_{\text{posterior odds}} = \underbrace{\frac{P(A)}{P(\lnot A)}}_{\text{prior odds}} \times \underbrace{\frac{P(B \mid A)}{P(B \mid \lnot A)}}_{\text{Bayes factor}}}$$

**Posterior odds = Prior odds × Bayes factor.**

The **Bayes factor** $\frac{P(B \mid A)}{P(B \mid \lnot A)}$ measures how much the evidence shifts the odds.

**Example:** Disease prevalence 1%, test accuracy 99%.

$$O(\text{infected}) = \frac{0.01}{0.99} = \frac{1}{99}, \qquad \text{Bayes factor} = \frac{0.99}{0.01} = \frac{99}{1}$$

$$O(\text{infected} \mid +) = \frac{1}{99} \times \frac{99}{1} = \frac{1}{1} \implies P = 50\%$$

Even with a 99%-accurate test, a positive result gives only 50% certainty — because the disease is rare.

### Continuous Random Variables

For continuous random variables $X$ and $Y$ with probability densities:

$$f_{X \mid Y=y}(x) = \frac{f_{Y \mid X=x}(y) \cdot f_X(x)}{f_Y(y)}$$

Here $f_X$ is the **prior density** and $f_{X \mid Y=y}$ is the **posterior density**. This is the version used throughout Bayesian statistics and probabilistic machine learning.

---

## 8. Worked Examples

### Example 1 — Medical Diagnosis

**Setup:** A doctor tests a patient for a disease. Let $E$ = *"patient has the disease"*, $F$ = *"test is positive."*

Suppose pancreatic cancer has incidence $1/100{,}000$, while $10/99{,}999$ healthy people also show the symptom. Per 100,000 people:

| | Cancer: Yes | Cancer: No | Total |
|---|:---:|:---:|:---:|
| **Symptom: Yes** | 1 | 10 | 11 |
| **Symptom: No** | 0 | 99,989 | 99,989 |
| **Total** | 1 | 99,999 | 100,000 |

$$P(E \mid F) = \frac{P(F \mid E) \cdot P(E)}{P(F \mid E) \cdot P(E) + P(F \mid \lnot E) \cdot P(\lnot E)} = \frac{1 \times 0.00001}{1 \times 0.00001 + (10/99999) \times 0.99999} = \frac{1}{11} \approx 9.1\%$$

> [!NOTE]
> Even with symptoms, the probability of cancer is only **9.1%** — because the disease is so rare, the **prior dominates**.

---

### Example 2 — Drug Testing and the Base-Rate Fallacy

**Setup:** A drug test is **99% sensitive** (true-positive rate = 0.99) and **99% specific** (false-positive rate = 0.01). Only **0.3%** of people actually use the drug.

$$P(\text{User} \mid +) = \frac{0.99 \times 0.003}{0.99 \times 0.003 + 0.01 \times 0.997} = \frac{0.00297}{0.01294} \approx 23\%$$

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/images/base-rate-tree.svg" alt="Natural-frequency tree: of 100,000 people, 297 true positives vs 997 false positives" />
</p>

> [!WARNING]
> Even though the test is "99% accurate," **most positive results are false**. The reason is the ***base rate***: users are so rare that the 1% error applied to the huge non-user group produces **more false positives than true positives**.
>
> **Takeaway:** The accuracy of a test is not the same as the probability that a positive result is correct. The **prior always matters**.

---

### Example 3 — Bent Coins

**Setup:** An urn holds coins of three types:

| Type | $P(\text{Heads})$ | Count | Prior $P(\text{type})$ |
|:---:|:---:|:---:|:---:|
| A (fair) | 0.5 | 2 | 2/5 |
| B (biased) | 0.6 | 2 | 2/5 |
| C (biased) | 0.9 | 1 | 1/5 |

Draw a coin at random, flip it — it lands **Heads**. What is $P(\text{type A} \mid \text{Heads})$?

**Step 1 — Evidence via law of total probability:**

$$P(H) = 0.5 \times \tfrac{2}{5} + 0.6 \times \tfrac{2}{5} + 0.9 \times \tfrac{1}{5} = 0.20 + 0.24 + 0.18 = 0.62$$

**Step 2 — Posterior for type A:**

$$P(A \mid H) = \frac{P(H \mid A) \cdot P(A)}{P(H)} = \frac{0.5 \times 0.4}{0.62} = \frac{0.20}{0.62} \approx 32\%$$

> [!NOTE]
> Observing Heads **lowered** the probability of fair coin A (from prior 40% down to ~32%), because Heads is better explained by the biased coins B and C.

---

## 9. Bayesian vs Frequentist Interpretations

The same equation can be read in two philosophically different ways.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/images/bayesian-vs-frequentist.svg" alt="Bayesian vs Frequentist interpretations of probability" />
</p>

| | **Bayesian (epistemic)** | **Frequentist** |
|---|---|---|
| Probability is… | a rational **degree of belief** | a long-run **proportion** of outcomes |
| $P(A)$ means | how plausible hypothesis $A$ is | the fraction of trials with property $A$ |
| Single events | can have a probability | only meaningful across many repetitions |
| Role of Bayes | links belief **before** and **after** evidence | relates proportions among sub-populations |

Under the **Bayesian view**:
- $P(A)$ is the **prior** — initial degree of belief.
- $P(B \mid A)$ and $P(B \mid \lnot A)$ are **likelihoods** — how expected the evidence is under each hypothesis.
- $P(A \mid B)$ is the **posterior** — the updated belief after accounting for evidence $B$.

---

## 10. Generalisation to Three Events

Bayes' theorem extends to additional conditioning events. With a third event $C$ (where $P(C) > 0$), every probability is conditioned on $C$:

$$\boxed{P(A \mid B \cap C) = \frac{P(B \mid A \cap C) \cdot P(A \mid C)}{P(B \mid C)}}$$

**Derivation:** The joint probability $P(A \cap B \cap C)$ can be written two ways using the chain rule:

$$P(A \cap B \cap C) = P(A \mid B \cap C) \cdot P(B \mid C) \cdot P(C)$$

$$P(A \cap B \cap C) = P(B \mid A \cap C) \cdot P(A \mid C) \cdot P(C)$$

Equating them and dividing by $P(B \mid C) \cdot P(C)$ gives the result above.

---

## 11. Practice Problems

**Problem 1:** A factory has two machines. Machine A produces 60% of all parts with a 2% defect rate. Machine B produces 40% with a 5% defect rate. A randomly selected part is defective. What is the probability it came from Machine A?

<details>
<summary>Solution</summary>

Let $D$ = defective, $A$ = from Machine A.

$$P(D) = P(D \mid A) \cdot P(A) + P(D \mid B) \cdot P(B) = 0.02 \times 0.6 + 0.05 \times 0.4 = 0.012 + 0.020 = 0.032$$

$$P(A \mid D) = \frac{P(D \mid A) \cdot P(A)}{P(D)} = \frac{0.02 \times 0.6}{0.032} = \frac{0.012}{0.032} = \mathbf{37.5\%}$$

</details>

---

**Problem 2:** An email spam filter knows 30% of emails are spam. The word "free" appears in 80% of spam and 10% of legitimate emails. An email containing "free" arrives — what is the probability it is spam?

<details>
<summary>Solution</summary>

Let $S$ = spam, $F$ = contains "free".

$$P(F) = P(F \mid S) \cdot P(S) + P(F \mid \lnot S) \cdot P(\lnot S) = 0.80 \times 0.30 + 0.10 \times 0.70 = 0.24 + 0.07 = 0.31$$

$$P(S \mid F) = \frac{P(F \mid S) \cdot P(S)}{P(F)} = \frac{0.80 \times 0.30}{0.31} = \frac{0.24}{0.31} \approx \mathbf{77.4\%}$$

</details>

---

**Problem 3:** Three bags each contain red and blue balls:

| Bag | Red | Blue |
|---|:---:|:---:|
| Bag 1 | 3 | 7 |
| Bag 2 | 5 | 5 |
| Bag 3 | 8 | 2 |

A bag is chosen at random (equal probability) and a ball drawn is **Red**. What is the probability it came from Bag 3?

<details>
<summary>Solution</summary>

$$P(R) = \frac{3}{10} \times \frac{1}{3} + \frac{5}{10} \times \frac{1}{3} + \frac{8}{10} \times \frac{1}{3} = \frac{1}{30}(3 + 5 + 8) = \frac{16}{30} = \frac{8}{15}$$

$$P(B_3 \mid R) = \frac{P(R \mid B_3) \cdot P(B_3)}{P(R)} = \frac{\tfrac{8}{10} \times \tfrac{1}{3}}{\tfrac{8}{15}} = \frac{\tfrac{8}{30}}{\tfrac{8}{15}} = \frac{8}{30} \times \frac{15}{8} = \mathbf{\frac{1}{2} = 50\%}$$

</details>

---

## 12. Quick Revision — Important Formulas

| Concept | Formula |
|---|---|
| **Bayes' theorem** | $P(A \mid B) = \dfrac{P(B \mid A) \cdot P(A)}{P(B)}$ |
| **Derivation identity** | $P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$ |
| **Law of total probability** | $P(B) = \displaystyle\sum_i P(B \mid A_i) \cdot P(A_i)$ |
| **Extended Bayes** | $P(A_i \mid B) = \dfrac{P(A_i) \cdot P(B \mid A_i)}{\sum_j P(A_j) \cdot P(B \mid A_j)}$ |
| **Two-term form** | $P(A \mid B) = \dfrac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid \lnot A) \cdot P(\lnot A)}$ |
| **Proportional form** | $P(A \mid B) \propto P(A) \cdot P(B \mid A)$ |
| **Odds form** | $O(A \mid B) = O(A) \cdot \dfrac{P(B \mid A)}{P(B \mid \lnot A)}$ |
| **Three-event form** | $P(A \mid B \cap C) = \dfrac{P(B \mid A \cap C) \cdot P(A \mid C)}{P(B \mid C)}$ |

---

## 13. Connection to Naive Bayes Classifier

Bayes' theorem is the **direct mathematical foundation** of the Naive Bayes Classifier.

### From Bayes' Theorem to the Classifier

Given a data point $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ and class label $C$, the goal is to compute the **posterior**:

$$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$

This is exactly Bayes' theorem with $A = C$ (the class hypothesis) and $B = \mathbf{x}$ (the observed features as evidence).

### The "Naive" Independence Assumption

Computing $P(\mathbf{x} \mid C)$ exactly requires all joint conditional distributions — intractable for many features. Naive Bayes assumes features are **conditionally independent** given the class:

$$P(\mathbf{x} \mid C) = \prod_{i=1}^{n} P(x_i \mid C)$$

This gives the **MAP decision rule**:

$$\boxed{\hat{C} = \arg\max_{C} P(C) \prod_{i=1}^{n} P(x_i \mid C)}$$

### Mapping Bayes' Terms to the Classifier

| Component | Bayes' theorem term | Role in Naive Bayes |
|---|---|---|
| $P(C)$ | **Prior** | Class frequency in training data |
| $P(x_i \mid C)$ | **Likelihood** | Per-feature conditional probability given class |
| $P(C \mid \mathbf{x})$ | **Posterior** | The quantity we maximise to classify |
| $P(\mathbf{x})$ | **Evidence** | Same for all classes — dropped from $\arg\max$ |

> [!NOTE]
> **Why the evidence is ignored:** Since $P(\mathbf{x})$ is identical for all candidate classes, it does not affect which class has the highest posterior. The classifier only needs $P(C) \cdot P(\mathbf{x} \mid C)$.

---

## 14. References

> [!NOTE]
> **Primary reference:** The content and all diagrams in this chapter are adapted from [Bayes' Theorem](../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/README.md) in the *Mathematics of Machine Learning* module.


### Within This Module

- [Conditional Probability](Conditional%20Probability.md)
- [Independent and Dependent Events in Probability](Independent%20and%20dependent%20Events%20in%20Probability.md)
- [Mutually Exclusive Events in Probability](Mutually%20Exclusive%20Events%20in%20Probability.md)
- [Problem Based upon Bayes Theorem](Problem%20Based%20upon%20Bayes%20Theorem.md)
- [Naive Bayes Classifier](Naive%20Bayes%20Classifier.md)

### Mathematics of Machine Learning

- [Bayes' Theorem — Full Reference](../Mathematics%20of%20Machine%20Learning/Bayes%27%20theorem/README.md)

### External References

- [Wikipedia — Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [GeeksforGeeks — Bayes' Theorem](https://www.geeksforgeeks.org/maths/bayes-theorem/)
- Thomas Bayes & Richard Price, *An Essay towards solving a Problem in the Doctrine of Chances* (1763)

---

<p align="center">
  ⬅️ <strong>Previous:</strong> <a href="Mutually%20Exclusive%20Events%20in%20Probability.md">Mutually Exclusive Events in Probability</a> &nbsp&nbsp|&nbsp&nbsp <strong>Next:</strong> <a href="Problem%20Based%20upon%20Bayes%20Theorem.md">Problem Based upon Bayes Theorem</a> ➡️
</p>
