# Bayes' Theorem

**Bayes' theorem** (also called **Bayes' law** or **Bayes' rule**) is one of the most *important* results in **probability**.
- **Bayes' theorem** (also known as the **Bayes Rule** or **Bayes Law**) is used to determine the ***conditional probability*** of event **A** when event **B** has *already occurred*.
- It is a **fundamental rule of probability** that describes how to ***update*** the probability of a **hypothesis** when ***new evidence*** arrives.
- It is named after the Reverend ***Thomas Bayes*** (c. 1701–1761) and was developed independently by ***Pierre-Simon Laplace***.
- It gives a mathematical way to ***invert*** **conditional probabilities** — to find the probability of a **cause** given its **effect**.

> **Main reference:** This note follows the structure and examples of the [Wikipedia: Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) article.

---

## Table of Contents

- **[1. The Formula at a Glance](#1-the-formula-at-a-glance)**
- **[2. Statement of the Theorem](#2-statement-of-the-theorem)**
- **[3. Visual Proof / Derivation](#3-visual-proof--derivation)**
- **[4. The Core Intuition: Updating Beliefs](#4-the-core-intuition-updating-beliefs)**
- **[5. The Extended Form (Law of Total Probability)](#5-the-extended-form-law-of-total-probability)**
- **[6. Worked Examples](#6-worked-examples)**
  - [6.1 Medical Diagnosis](#61-medical-diagnosis)
  - [6.2 Drug Testing and the Base-Rate Fallacy](#62-drug-testing-and-the-base-rate-fallacy)
  - [6.3 Bent Coins](#63-bent-coins)
- **[7. Interpretations: Bayesian vs. Frequentist](#7-interpretations-bayesian-vs-frequentist)**
- **[8. Useful Forms](#8-useful-forms)**
  - [8.1 Prior and Likelihood (Proportionality)](#81-prior-and-likelihood-proportionality)
  - [8.2 Continuous Random Variables](#82-continuous-random-variables)
  - [8.3 The Odds Form and the Bayes Factor](#83-the-odds-form-and-the-bayes-factor)
- **[9. Generalization to Three Events](#9-generalization-to-three-events)**
- **[10. Why Bayes' Theorem Matters in Machine Learning](#10-why-bayes-theorem-matters-in-machine-learning)**
- **[11. Bayes' Theorem Related Terms](#11-bayes-theorem-related-terms)**
  - [11.1 Hypotheses](#111-hypotheses)
  - [11.2 Prior Probability (Priori Probability)](#112-prior-probability-priori-probability)
  - [11.3 Posterior Probability](#113-posterior-probability)
  - [11.4 Likelihood](#114-likelihood)
  - [11.5 Evidence](#115-evidence)
  - [11.6 Conditional Probability](#116-conditional-probability)
  - [11.7 Joint Probability](#117-joint-probability)
  - [11.8 Random Variables](#118-random-variables)
- **[12. Key Concepts Checklist](#12-key-concepts-checklist)**
- **[13. References](#13-references)**

---

## 1. The Formula at a Glance

For two events $A$ and $B$ with $P(B) \neq 0$:

```math
\boxed{\,P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}\,}
```

Each piece has a name and a role:

![Anatomy of Bayes' theorem: posterior equals likelihood times prior divided by evidence, with each term colour-coded and labelled](images/bayes-formula-anatomy.svg)

| Term | Name | Meaning |
|---|---|---|
| $P(A \mid B)$ | **Posterior** or **Conditional Probability** | Updated probability of hypothesis $A$ *after* seeing evidence $B$. |
| $P(B \mid A)$ | **Likelihood** | Probability of observing evidence $B$ *if* $A$ is true. |
| $P(A)$ | **Prior** | Initial probability of $A$ *before* any evidence. |
| $P(B)$ | **Evidence** (marginal likelihood) | Total probability of observing $B$ across all hypotheses; acts as a normalising constant. |

---

## 2. Statement of the Theorem

Bayes' theorem is stated as:

```math
\boxed{\,P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}\,}, \qquad P(B) \neq 0
```

where $A$ and $B$ are **events**.

- $P(A \mid B)$ is a **conditional probability**: the probability of $A$ occurring *given that* $B$ is true. It is the ***posterior*** probability of $A$ given $B$.
- $P(B \mid A)$ is also **conditional**: the probability of $B$ occurring *given that* $A$ is true. Viewed as a function of $A$ for fixed $B$, it is the ***likelihood***.
- $P(A)$ and $P(B)$ are the probabilities of observing $A$ and $B$ *on their own*, with no conditioning. These are the ***marginal*** (or *unconditioned*) probabilities.

---

## 3. Visual Proof / Derivation

- Bayes' theorem **follows directly from the relationship** between **joint** and **conditional probabilities**. 
- The **joint probability** of $A$ and $B$ both happening can be **written two equivalent ways:**

```math
P(A \cap B) = P(A \mid B)\,P(B)
```

```math
P(A \cap B) = P(B \mid A)\,P(A)
```

Since both equal $P(A \cap B)$, they equal each other:

```math
P(A \mid B)\,P(B) = P(B \mid A)\,P(A)
```

Dividing both sides by $P(B)$ (assuming $P(B) \neq 0$) gives Bayes' theorem:

```math
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
```

The diagram below shows this **geometrically**: the overlap of regions $A$ and $B$ is the *same* whether you compute it "down from $A$" or "down from $B$," and equating the two routes is *exactly* Bayes' rule.

![Visual proof of Bayes' theorem using overlapping regions for events A and B, showing that the joint probability computed two ways must be equal](images/image.png)

---

## 4. The Core Intuition: Updating Beliefs

The single ***most important*** idea is this: **Bayes' theorem tells you how to change your mind when you get new data.** You start with a ***prior*** belief, you observe some ***evidence***, and you end with a revised ***posterior*** belief.

![Bayesian updating pipeline: a prior belief enters, is combined with the likelihood of the evidence and normalised by the total evidence, and leaves as a higher posterior belief](images/prior-posterior-update.svg)

Three ways to think about it:

- **Updating beliefs.** It formalises how a rational observer moves from a prior to a posterior as evidence accumulates. Today's posterior becomes tomorrow's prior.
- **Flipping conditional probabilities.** It lets you compute $P(A \mid B)$ when you can only easily measure the *reverse*, $P(B \mid A)$ — e.g. "probability of disease given a positive test" from "probability of a positive test given disease."
- **Weighing evidence against base rates.** A strong piece of evidence can still leave a hypothesis unlikely if the hypothesis was very rare to begin with (see §6.2).

---

## 5. The Extended Form (Law of Total Probability)

- Often we cannot measure $P(B)$ directly, but we *can* break it down. 
- If events $A_1, A_2, \ldots$ are **mutually exclusive and exhaustive** (exactly one must occur), the **law of total probability** gives:

```math
P(B) = \sum_i P(B \mid A_i)\,P(A_i)
```

Substituting this into the denominator yields the extended form:

```math
P(A_i \mid B) = \frac{P(A_i)\,P(B \mid A_i)}{\sum_j P(A_j)\,P(B \mid A_j)}
```

The *simplest* and **most common** case uses just a **hypothesis** $A$ and its **complement** $\lnot A$ (*"not $A$"*), which *always* form a ***mutually exclusive, exhaustive*** pair:

```math
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B \mid A)\,P(A) + P(B \mid \lnot A)\,P(\lnot A)}
```

This ***two-term version*** is the *workhorse* for the examples below.

---

## 6. Worked Examples

### 6.1 Medical Diagnosis

- A doctor tests a patient for a **disease**. 
- Let $E$ be the event *"patient has the disease"* and $F$ the event *"test is positive."* We want $P(E \mid F)$:

```math
P(E \mid F) = \frac{P(F \mid E)\,P(E)}{P(F \mid E)\,P(E) + P(F \mid \lnot E)\,P(\lnot E)}
```

Here $P(E)$ is the **prevalence** of the disease and $P(F \mid E)$ is the **sensitivity** (true-positive rate).

- **A concrete case.** Suppose pancreatic cancer has incidence $1/100{,}000$, while $10/99{,}999$ healthy people also show the relevant symptoms. Per 100,000 people:

| | Cancer: Yes | Cancer: No | Total |
|---|:---:|:---:|:---:|
| **Symptom: Yes** | 1 | 10 | 11 |
| **Symptom: No** | 0 | 99,989 | 99,989 |
| **Total** | 1 | 99,999 | 100,000 |

```math
P(\text{Cancer} \mid \text{Symptoms}) = \frac{1 \times 0.00001}{1 \times 0.00001 + (10/99999) \times 0.99999} = \frac{1}{11} \approx 9.1\%
```

- Even with symptoms, the probability of cancer is only about **9.1%** — because the disease is *so rare* to begin with.

### 6.2 Drug Testing and the Base-Rate Fallacy

A drug test is **99% sensitive** (true-positive rate $= 0.99$) and **99% specific** (true-negative rate $= 0.99$, so the false-positive rate is $0.01$). Suppose **0.3%** of people actually use the drug. What is the probability that a random person who tests positive is really a user?

```math
P(\text{User} \mid \text{+}) = \frac{P(\text{+} \mid \text{User})\,P(\text{User})}{P(\text{+} \mid \text{User})\,P(\text{User}) + P(\text{+} \mid \text{Non-user})\,P(\text{Non-user})}
```

```math
= \frac{0.99 \times 0.003}{0.99 \times 0.003 + 0.01 \times 0.997} \approx 23\%
```

Even though the test is *"99% accurate,"* **most positive results are false.** The reason is the ***base rate***: users are *so rare* that the small **1% error** applied to the *huge* non-user group produces **more false positives than true positives**.

![Natural-frequency tree for the drug test: of 100,000 people, 300 users yield 297 true positives while 99,700 non-users still yield 997 false positives, so only 297 of 1,294 positives are real — about 23 percent](images/base-rate-tree.svg)

> **Takeaway:** Accuracy of a test is *not* the same as the probability that a positive result is correct. The prior (base rate) always matters.

### 6.3 Bent Coins

An urn holds coins of three types that land heads with different probabilities:

| Type | $P(\text{Heads})$ | Count | Prior $P(\text{type})$ |
|:---:|:---:|:---:|:---:|
| A (fair) | 0.5 | 2 | 2/5 |
| B (biased) | 0.6 | 2 | 2/5 |
| C (biased) | 0.9 | 1 | 1/5 |

Draw a coin at random, flip it, and it comes up **heads**. What is the probability it was **type A**?

First the **evidence** via the *law of total probability*:

```math
P(H) = P(H \mid A)P(A) + P(H \mid B)P(B) + P(H \mid C)P(C)
```

```math
P(H) = 0.5 \times \tfrac{2}{5} + 0.6 \times \tfrac{2}{5} + 0.9 \times \tfrac{1}{5} = 0.20 + 0.24 + 0.18 = 0.62
```

Then the **posterior** for type A:

```math
P(A \mid H) = \frac{P(H \mid A)\,P(A)}{P(H)} = \frac{0.5 \times 0.4}{0.62} = \frac{0.2}{0.62} \approx 32\%
```

Observing heads ***lowered*** the probability of the **fair coin A** (from a prior of **40%** down to about **32%**), because heads is *better explained* by the more heads-prone coins **B** and **C**.

---

## 7. Interpretations: Bayesian vs. Frequentist

The *same equation* can be read in two **philosophically different** ways, depending on what *"probability"* means to you.

![Two interpretations: Bayesian probability as a rational degree of belief updated by evidence, versus frequentist probability as a long-run proportion of outcomes](images/bayesian-vs-frequentist.svg)

| | **Bayesian (epistemic)** | **Frequentist** |
|---|---|---|
| Probability is… | a rational **degree of belief** | a long-run **proportion** of outcomes |
| $P(A)$ means | how plausible hypothesis $A$ is | the fraction of trials with property $A$ |
| Single events | can have a probability | only meaningful across many repetitions |
| Role of Bayes | links belief **before** and **after** evidence | relates proportions among sub-populations |

Under the Bayesian view, if $A$ is a proposition and $B$ is evidence:

- $P(A)$ is the **prior** — the initial degree of belief in $A$.
- $P(\lnot A) = 1 - P(A)$ is the initial belief that $A$ is false.
- $P(B \mid A)$ and $P(B \mid \lnot A)$ are **likelihoods** — how expected the evidence is under each hypothesis.
- $P(A \mid B)$ is the **posterior** — the belief in $A$ after accounting for $B$.

---

## 8. Useful Forms

### 8.1 Prior and Likelihood (Proportionality)

- In many applications (especially **Bayesian inference**) the evidence $B$ is fixed and we only vary the hypothesis $A$. 
Then $P(B)$ is a constant, and Bayes' theorem reduces to a proportionality:

```math
P(A \mid B) \;\propto\; P(A) \cdot P(B \mid A)
```

> **In words:** the posterior is proportional to the prior times the likelihood. The denominator $P(B)$ just rescales everything so the posterior sums (or integrates) to 1.

### 8.2 Continuous Random Variables

- For two continuous random variables $X$ and $Y$ with densities, the theorem takes the analogous form:

```math
f_{X \mid Y=y}(x) = \frac{f_{Y \mid X=x}(y)\,f_X(x)}{f_Y(y)}
```

valid wherever $f_X(x) > 0$ and $f_Y(y) > 0$. This is the version used throughout **Bayesian statistics**, where $f_X$ is a ***prior density*** and $f_{X \mid Y=y}$ is the ***posterior density***. (The fully general *measure-theoretic* statement rests on the **Radon–Nikodym theorem**, formulated by **Kolmogorov** in 1933.)

### 8.3 The Odds Form and the Bayes Factor

Writing Bayes' theorem for both $A$ and $\lnot A$ and dividing, the term $P(B)$ cancels:

```math
\underbrace{\frac{P(A \mid B)}{P(\lnot A \mid B)}}_{\text{posterior odds}} = \underbrace{\frac{P(A)}{P(\lnot A)}}_{\text{prior odds}} \times \underbrace{\frac{P(B \mid A)}{P(B \mid \lnot A)}}_{\text{Bayes factor}}
```

Or compactly, with $O(\cdot)$ denoting odds:

```math
O(A \mid B) = O(A) \cdot \frac{P(B \mid A)}{P(B \mid \lnot A)}
```

The ratio $\frac{P(B \mid A)}{P(B \mid \lnot A)}$ is the **Bayes factor** (or likelihood ratio). In short: **posterior odds = prior odds × Bayes factor.**

**Example.** A disease has prevalence 1%, and a test is correct 99% of the time.

```math
O(\text{infected}) = \frac{0.01}{0.99} = \frac{1}{99}, \qquad \text{Bayes factor} = \frac{0.99}{0.01} = \frac{99}{1}
```

```math
O(\text{infected} \mid \text{+}) = \frac{1}{99} \times \frac{99}{1} = \frac{1}{1}
```

Posterior odds of **1-to-1** mean a posterior probability of **50%** — the *same surprising* base-rate lesson from §6.2, seen through **odds**.

---

## 9. Generalization to Three Events

Bayes' theorem extends to *additional conditioning events*. With a **third event** $C$ (where $P(C) > 0$), **every probability** is conditioned on $C$:

```math
P(A \mid B \cap C) = \frac{P(B \mid A \cap C)\,P(A \mid C)}{P(B \mid C)}
```

This follows from the **chain rule**. Writing the joint probability $P(A \cap B \cap C)$ — *the probability that all three events $A$, $B$, and $C$ occur together* — two ways,

```math
P(A \cap B \cap C) = P(A \mid B \cap C)\,P(B \mid C)\,P(C)
```

```math
P(A \cap B \cap C) = P(B \mid A \cap C)\,P(A \mid C)\,P(C)
```

equating them and solving for $P(A \mid B \cap C)$ gives the result above.

---

## 10. Why Bayes' Theorem Matters in Machine Learning

Bayes' rule is the ***mathematical backbone*** of a large family of methods:

- **Naive Bayes classifiers** — spam detection, text and document classification. They apply Bayes' theorem with a "naive" conditional-independence assumption between features.
- **Bayesian inference** — updating model parameters as data arrives; the foundation of probabilistic modelling.
- **Spam filters** — estimating $P(\text{spam} \mid \text{words})$ from $P(\text{words} \mid \text{spam})$.
- **Medical / diagnostic models** — combining test likelihoods with disease base rates.
- **Markov Chain Monte Carlo (MCMC)** — modern sampling methods that make Bayesian posteriors computable even for complex or improper priors.
- **Finance and risk** — re-evaluating credit risk or market states as new data comes in.

The common thread: **start with a prior, observe data, compute a posterior** — exactly the loop in §4.

---

## 11. Bayes' Theorem Related Terms

After learning about Bayes' theorem in detail, let us understand some ***important terms*** related to the concepts we covered in the **formula and derivation**.

### 11.1 Hypotheses

- A ***hypothesis*** is a **possible event** or **outcome** that we are trying to evaluate.
- In Bayes' theorem, hypotheses are typically written as $E_1, E_2, \ldots, E_n$, where each $E_i$ represents a **distinct scenario** or *cause*.
- The hypotheses must be ***mutually exclusive*** and ***exhaustive*** — exactly one of them must be true.
- **Example:** *"The patient has disease A"*, *"the patient has disease B"*, or *"the patient is healthy."*

### 11.2 Prior Probability (Priori Probability)

- The ***prior probability*** $P(E_i)$ is the **initial probability** assigned to a hypothesis *before* any new evidence is observed.
- It represents our ***pre-existing knowledge*** or ***belief*** about how likely each hypothesis is.
- Also called the **"a priori" probability** (from Latin: *"from what comes before"*).
- **Example:** The *prevalence* of a disease in the general population — how many people have it *before* any test is run.

### 11.3 Posterior Probability

- The ***posterior probability*** $P(E_i \mid A)$ is the **updated probability** of a hypothesis *after* incorporating **new information** via Bayes' theorem.
- It is the *output* of Bayes' theorem — the ***revised belief*** once evidence has been factored in.
- Also called the **"a posteriori" probability** (from Latin: *"from what comes after"*).
- **Example:** The probability that a patient *has a disease* given that their **test result was positive**.

### 11.4 Likelihood

- The ***likelihood*** $P(B \mid A)$ is the probability of observing evidence $B$ *if* hypothesis $A$ is true.
- Viewed as a function of $A$ for a fixed observed $B$, it measures how well the hypothesis **explains** the evidence.
- A **high likelihood** means the evidence is very compatible with the hypothesis.
- **Note:** The likelihood is *not* a probability over hypotheses — it is a probability over evidence values for a fixed hypothesis.
- **Example:** The *sensitivity* of a medical test — $P(\text{positive test} \mid \text{has disease})$. A test with 99% sensitivity has a likelihood of 0.99 for the hypothesis *"patient has disease."*

### 11.5 Evidence

- The ***evidence*** $P(B)$ (also called the ***marginal likelihood***) is the **total probability** of observing evidence $B$ across *all* possible hypotheses.
- It acts as a **normalising constant** that ensures the posterior probabilities sum to 1.
- It is computed using the **law of total probability**: $P(B) = \sum_i P(B \mid A_i)\,P(A_i)$.
- In practice it is often the hardest term to compute, which is why the **proportional form** $P(A \mid B) \propto P(A) \cdot P(B \mid A)$ is widely used.
- **Example:** The total probability of a positive test result — accounting for both people who *have* the disease and those who *do not*.

### 11.6 Conditional Probability

- ***Conditional probability*** $P(A \mid B)$ is the probability of event $A$ occurring ***given that*** event $B$ has *already occurred*.
- It describes how the **occurrence of one event** changes the likelihood of **another event**.
- The vertical bar "$\mid$" is read as **"given"**: $P(A \mid B)$ is read *"the probability of A given B."*
- **Example:** The probability that it *rains* today, given that the **sky is cloudy**.

### 11.7 Joint Probability

- ***Joint probability*** $P(A \cap B)$ is the probability of **two or more events** occurring ***together*** — simultaneously.
- The symbol $\cap$ is read as **"and"** or **"intersection"** — $P(B \cap C)$ means *"the probability that **both** $B$ **and** $C$ occur."*
- It extends naturally to **three events**: $P(A \cap B \cap C)$ means *"the probability that $A$, $B$, **and** $C$ all occur at the same time."*
- It is the *intersection* of the events in **probability space** — only the outcomes that belong to *every* event at once are counted.
- The connection to Bayes' theorem: $P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$ — *equating these two expressions **is** the derivation of Bayes' rule.*
- **Example:** The probability that a randomly chosen person is *both* **a smoker** ***and*** **has lung disease** is $P(\text{smoker} \cap \text{lung disease})$.

> **Numerical Example — Rolling a Die**
>
> Roll a fair six-sided die once. Define two events:
>
> | Event | Description | Outcomes | Probability |
> |:---:|---|:---:|:---:|
> | $B$ | Roll an **even** number | $\{2, 4, 6\}$ | $P(B) = \tfrac{3}{6} = \tfrac{1}{2}$ |
> | $C$ | Roll a number **greater than 3** | $\{4, 5, 6\}$ | $P(C) = \tfrac{3}{6} = \tfrac{1}{2}$ |
> | $B \cap C$ | Roll a number that is **both even AND > 3** | $\{4, 6\}$ | $P(B \cap C) = \tfrac{2}{6} = \tfrac{1}{3}$ |
>
> Only the outcomes $\{4, 6\}$ satisfy **both** conditions at the same time — that is the intersection $B \cap C$.
>
> Notice: $P(B \cap C) = \tfrac{1}{3} < P(B) = \tfrac{1}{2}$, because demanding *two* conditions at once can only *reduce* (or at most equal) the probability.

### 11.8 Random Variables

- A ***random variable*** is a **real-valued variable** whose value is determined by the outcome of a **random experiment**.
- Each possible value of a random variable has an associated **probability**, found via **experimental** or **theoretical probability**.
- Bayes' theorem applies to both ***discrete*** random variables (e.g. coin flips, disease/no-disease) and ***continuous*** random variables (e.g. a patient's blood pressure, a model's output score).
- **Example:** Let $X$ be the outcome of rolling a die — $X$ can take values $1, 2, 3, 4, 5, 6$, each with probability $\tfrac{1}{6}$.

---

## 12. Key Concepts Checklist

- [ ] I can state Bayes' theorem and name all four terms (posterior, likelihood, prior, evidence).
- [ ] I can derive it from the joint-probability identity $P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)$.
- [ ] I can expand the denominator using the law of total probability.
- [ ] I understand why a 99%-accurate test can still produce mostly false positives (the **base-rate fallacy**).
- [ ] I can use the two-term complement form for a hypothesis and its negation.
- [ ] I know the proportional form: posterior $\propto$ prior $\times$ likelihood.
- [ ] I can convert between probability and odds and apply the **Bayes factor**.
- [ ] I can distinguish the **Bayesian** and **frequentist** interpretations.

---

## 13. References

- [Wikipedia: Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) — the main reference for this note, including the medical, drug-testing, bent-coin, and odds-form examples.
- [GeeksforGeeks: Bayes' Theorem](https://www.geeksforgeeks.org/maths/bayes-theorem/) — accessible walkthrough of the formula, derivation, and worked examples.
- Thomas Bayes & Richard Price, *An Essay towards solving a Problem in the Doctrine of Chances* (1763).
- Pierre-Simon Laplace, *Théorie analytique des probabilités* (1812).
