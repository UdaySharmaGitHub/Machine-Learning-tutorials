# Conditional Probability

- **Conditional probability** is the probability of an event occurring ***given that another event has already happened***.
- It is the core mathematical idea behind the Naive Bayes Classifier — every prediction the classifier makes is a conditional probability of the form:

$$P(\text{class} \mid \text{features})$$

---

## Table of Contents

1. [Definition](#1-definition)
2. [The Formula](#2-the-formula)
3. [Key Concepts and Notation](#3-key-concepts-and-notation)
4. [Properties](#4-properties)
5. [Steps to Calculate](#5-steps-to-calculate)
6. [Worked Examples](#6-worked-examples)
   - [Example 1 — Tossing Two Coins](#example-1--tossing-two-coins)
   - [Example 2 — Drawing Cards Without Replacement](#example-2--drawing-cards-without-replacement)
   - [Example 3 — Medical Test (False Positives)](#example-3--medical-test-false-positives)
   - [Example 4 — Rolling Two Dice](#example-4--rolling-two-dice)
7. [Independent vs Dependent Events](#7-independent-vs-dependent-events)
8. [Multiplication Rule](#8-multiplication-rule)
   - [Chain Rule of Probability](#chain-rule-of-probability)
   - [As an Axiom of Probability](#as-an-axiom-of-probability)
9. [Relationship Between Conditional Probability and Bayes' Theorem](#9-relationship-between-conditional-probability-and-bayes-theorem)
10. [Why Is Conditional Probability Important in Machine Learning?](#10-why-is-conditional-probability-important-in-machine-learning)
11. [Applications in Machine Learning](#11-applications-in-machine-learning)
12. [Practice Problems](#12-practice-problems)
13. [Quick Revision — Important Formulas](#13-quick-revision--important-formulas)
14. [Connection to Naive Bayes Classifier](#14-connection-to-naive-bayes-classifier)
15. [References](#15-references)

---

## 1. Definition

- **Conditional probability** is the probability of an event $A$ occurring, ***given that event $B$ has already occurred***.

***Consider a simple example:*** the probability that any given person has a cough on any given day might be only **5%**. But if we know the person is **sick**, then they are ***much more likely*** to be coughing — say, **75%**. This is the conditional probability:

$$P(\text{Cough}) = 0.05 \qquad \text{vs} \qquad P(\text{Cough} \mid \text{Sick}) = 0.75$$

The second probability is ***conditioned*** on the knowledge that the person is sick. That **extra information** *changes* — and typically ***sharpens*** — our estimate.

---

## 2. The Formula

- If $A$ and $B$ are two events from a sample space, and $P(B) > 0$, the **conditional probability of $A$ given $B$** is:

$$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}}$$

where:

| Symbol | Meaning |
|---|---|
| $P(A \mid B)$ | Probability of event $A$ occurring given that $B$ has occurred |
| $P(A \cap B)$ | Probability of both $A$ and $B$ occurring simultaneously |
| $P(B)$ | Probability of event $B$ occurring (must be $> 0$) |

> **Note:** The formula is only defined when $P(B) > 0$. If $P(B) = 0$, the conditional probability $P(A \mid B)$ is undefined.

**Equivalent form (Multiplication Rule):**

$$\boxed{P(A \cap B) = P(B) \cdot P(A \mid B)}$$

- This rearrangement is often more intuitive:
- ***"the probability of both happening equals the probability of $B$ times the probability of $A$ given $B$ has occurred."***

---

## 3. Key Concepts and Notation

| Concept | Explanation |
|---|---|
| $P(A \mid B)$ | Read as "the probability of $A$ given $B$". The vertical bar means "given". |
| **Order matters** | $P(A \mid B) \neq P(B \mid A)$ in general |
| **Reduced sample space** | Conditioning on $B$ restricts the universe to outcomes where $B$ is true |
| **Symmetry breaks** | Swapping which event is "given" changes the answer completely |

**Classic asymmetry example:**

- Suppose someone has dengue fever. The chance they test positive might be 90%:

$$P(\text{Positive} \mid \text{Dengue}) = 0.90$$

- But if someone tests positive, the chance they actually have dengue (a rare disease) could be only 15% due to high false positive rates:

$$P(\text{Dengue} \mid \text{Positive}) = 0.15$$

- Confusing these ***two probabilities is a common logical error*** known as the **base rate fallacy**.

---

## 4. Properties

Let $A$ and $B$ be events in a sample space $S$.

**Property 1 — Certainty when conditioning on itself:**

$$\boxed{P(S \mid A) = P(A \mid A) = 1}$$

**Property 2 — Addition rule carries over:**

- For events $A$, $B$, and conditioning event $X$ with $P(X) \neq 0$:

$$\boxed{P((A \cup B) \mid X) = P(A \mid X) + P(B \mid X) - P((A \cap B) \mid X)}$$

**Property 3 — Order is not commutative:**

$$\boxed{P(A \mid B) \neq P(B \mid A) \quad \text{in general}}$$

**Property 4 — Complement (only in first argument):**

$$\boxed{P(A' \mid B) = 1 - P(A \mid B)}$$

> **Warning:** $P(A \mid B') \neq 1 - P(A \mid B)$. The complement rule only flips the first argument.

**Property 5 — Independent events:**

- For independent events $A$ and $B$:

$$\boxed{P(A \cap B) = P(A) \cdot P(B)}$$

- For three mutually independent events $A$, $B$, $C$:

$$\boxed{P(A \cap B \cap C) = P(A) \cdot P(B) \cdot P(C)}$$

---

## 5. Steps to Calculate

To find the conditional probability $P(A \mid B)$:

| Step | Action |
|---|---|
| **Step 1** | ***Identify the Events*** — label them **Event A** (what you want to find) and **Event B** (the given condition) |
| **Step 2** | ***Determine the Probability of Event A*** — find $P(A)$ |
| **Step 3** | ***Determine the Probability of Event B*** — find $P(B)$ |
| **Step 4** | ***Determine the Probability of both A and B*** — find $P(A \cap B)$ |
| **Step 5** | ***Apply the Conditional Probability Formula*** — calculate $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$ |

---

## 6. Worked Examples

### Example 1 — Tossing Two Coins

**Setup:** Toss two fair coins. Let:
- $A$ = getting a head on the first coin
- $B$ = getting a head on the second coin

**Sample space:** $S = \{HH, HT, TH, TT\}$

Since the coins are independent, the outcome of one does not affect the other:

$$P(B \mid A) = P(B) = 0.5$$

The condition on $A$ provides no information about $B$, so the conditional probability equals the unconditional one.

---

### Example 2 — Drawing Cards Without Replacement

**Setup:** A standard 52-card deck. Draw two cards without replacing the first. Let:
- $A$ = the first card drawn is red
- $B$ = the second card drawn is red

**Unconditional probability of $B$:** $P(B) = 26/52 = 0.5$

**After conditioning on $A$:** After drawing a red card, 25 red cards and 51 total cards remain:

$$P(B \mid A) = \frac{25}{51} \approx 0.490$$

The events are **dependent** — drawing a red card first slightly reduces the chance the second is also red.

---

### Example 3 — Medical Test (False Positives)

**Setup:** A disease affects 1% of the population. A test for the disease is:
- 95% sensitive: $P(\text{Positive} \mid \text{Disease}) = 0.95$
- 90% specific: $P(\text{Negative} \mid \text{No Disease}) = 0.90$

**Question:** If a person tests positive, what is $P(\text{Disease} \mid \text{Positive})$?

$$P(\text{Positive}) = P(+ \mid \text{Disease}) \cdot P(\text{Disease}) + P(+ \mid \text{No Disease}) \cdot P(\text{No Disease})$$

$$= (0.95)(0.01) + (0.10)(0.99) = 0.0095 + 0.099 = 0.1085$$

$$P(\text{Disease} \mid \text{Positive}) = \frac{0.0095}{0.1085} \approx 8.8\%$$

Even with a positive test, there is only an ~8.8% chance of having the disease because the disease is rare. This counterintuitive result is why **base rates** always matter.

---

### Example 4 — Rolling Two Dice

**Setup:** Roll two fair six-sided dice, $D_1$ and $D_2$. The **sample space** contains $6 \times 6 = 36$ equally likely outcomes. Let:
- **Event A** = $D_1 = 2$ (the first die shows 2)
- **Event B** = $D_1 + D_2 \leq 5$ (the sum of both dice is at most 5)

**Question:** Given that the sum is at most 5, what is the probability that the first die showed a 2?

$$\boxed{P(D_1 = 2 \mid D_1 + D_2 \leq 5) = \frac{P(D_1 = 2 \cap D_1 + D_2 \leq 5)}{P(D_1 + D_2 \leq 5)}}$$

---

**Step 1 — Find $P(A)$: Probability that $D_1 = 2$**

There are exactly **6 outcomes** where $D_1 = 2$: $(2,1), (2,2), (2,3), (2,4), (2,5), (2,6)$.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/images/dice-table-1-d1-equals-2.svg" alt="Dice sum table with D1=2 row highlighted in red" />
</p>

$$P(D_1 = 2) = \frac{6}{36} = \frac{1}{6}$$

---

**Step 2 — Find $P(B)$: Probability that the sum $\leq 5$**

The outcomes where $D_1 + D_2 \leq 5$ are:

| $D_1$ | Valid $D_2$ values | Count |
|---|---|---|
| 1 | 1, 2, 3, 4 | 4 |
| 2 | 1, 2, 3 | 3 |
| 3 | 1, 2 | 2 |
| 4 | 1 | 1 |
| 5, 6 | none | 0 |

Total: $4 + 3 + 2 + 1 = $ **10 outcomes**.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/images/dice-table-2-sum-le-5.svg" alt="Dice sum table with sum ≤ 5 cells highlighted in blue" />
</p>

$$P(D_1 + D_2 \leq 5) = \frac{10}{36}$$

---

**Step 3 — Find $P(A \cap B)$: Both $D_1 = 2$ ***and*** sum $\leq 5$**

From the table above, when $D_1 = 2$, only $D_2 \in \{1, 2, 3\}$ give a sum $\leq 5$:

Outcomes: $(2,1), (2,2), (2,3)$ → **3 outcomes** in the intersection.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/images/dice-table-3-intersection.svg" alt="Dice sum table showing three-level color coding: orange for intersection, light red for D1=2 only, light blue for sum≤5 only" />
</p>

$$P(D_1 = 2 \cap D_1 + D_2 \leq 5) = \frac{3}{36}$$

---

**Step 4 — Apply the Formula**

$$P(D_1 = 2 \mid D_1 + D_2 \leq 5) = \frac{P(D_1 = 2 \cap D_1 + D_2 \leq 5)}{P(D_1 + D_2 \leq 5)} = \frac{3/36}{10/36} = \frac{3}{10} = 0.3$$

**Interpretation:** Without any condition, the probability of $D_1 = 2$ was $1/6 \approx 0.167$. Knowing the sum is at most 5 ***increases*** this probability to $3/10 = 0.3$, because a low sum makes it ***more likely*** that $D_1$ was small.

---

## 7. Independent vs Dependent Events

- Conditional probability cleanly ***distinguishes*** these ***two types of relationships***.

### Independent Events

- Two events are ***independent*** if *knowledge* of one gives ***no information*** about the other:

$$P(A \mid B) = P(A) \quad \text{and} \quad P(B \mid A) = P(B)$$

- *Examples:* **coin flips**, **dice rolls** (separate trials).

### Dependent Events

- Two events are ***dependent*** if the occurrence of one ***changes the probability*** of the other:

$$P(A \mid B) \neq P(A)$$

- *Examples:* **drawing cards without replacement**, **weather on consecutive days**.

**Summary table:**

| Type | Condition | Example |
|---|---|---|
| Independent | $P(A \mid B) = P(A)$ | Flipping two coins |
| Dependent | $P(A \mid B) \neq P(A)$ | Drawing cards without replacement |

> The Naive Bayes Classifier *assumes* all features are independent given the class — the "naive" assumption. In reality features are often dependent, but the classifier still performs remarkably well.

---

## 8. Multiplication Rule

- The Multiplication Rule follows directly from the definition of conditional probability:

$$\boxed{P(A \cap B) = P(A) \cdot P(B \mid A) = P(B) \cdot P(A \mid B)}$$

**For three events:**

$$\boxed{P(A \cap B \cap C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A \cap B)}$$

### Chain Rule of Probability

The pattern extends to ***any number of events*** — this generalization is called the **Chain Rule**:

$$\boxed{P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2 \mid A_1) \cdot P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \cdots \cap A_{n-1})}$$

Each factor *conditions on all previously seen events*, progressively narrowing the sample space.

<p align="center">
  <img src="../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/images/conditional%20Probability%20Chain%20Rule.png" alt="Conditional Probability Chain Rule" />
</p>

> **Why it matters in ML:** The Chain Rule is the backbone of **autoregressive language models** — the probability of a sentence is factored as the product of each word's probability given all preceding words: $P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})$.

**When events are independent**, this simplifies to:

$$\boxed{P(A \cap B) = P(A) \cdot P(B)}$$

This simplification is exactly what Naive Bayes exploits.

### As an Axiom of Probability

Some authors, such as ***de Finetti***, prefer to introduce conditional probability as an **axiom** rather than a derived definition:

$$\boxed{P(A \cap B) = P(A \mid B) \cdot P(B)}$$

This equation, although ***mathematically equivalent***, may be *intuitively easier to understand*. It can be read as:

> *"The probability of $B$ occurring, multiplied by the probability of $A$ occurring **provided that** $B$ has occurred, equals the probability of $A$ and $B$ occurring together."*

This is ***preferred philosophically*** under major probability interpretations such as the **subjective theory**, where conditional probability is treated as a *primitive entity* rather than a ratio.

**Connection to the Poincaré (Inclusion-Exclusion) Formula:**

The multiplication rule introduces a ***symmetry*** with the summation axiom:

$$\boxed{P(A \cup B) = P(A) + P(B) - P(A \cap B)}$$

Combining the two equations gives a ***new representation*** of the intersection:

$$P(A \cap B) = P(A) + P(B) - P(A \cup B) = P(A \mid B) \cdot P(B)$$

And equivalently, the union can be rewritten as:

$$\boxed{P(A \cup B) = P(A) + P(B) - P(A \mid B) \cdot P(B)}$$

---

## 9. Relationship Between Conditional Probability and Bayes' Theorem

Conditional probability is the ***foundation*** of Bayes' Theorem. Bayes' Theorem allows us to ***reverse*** a conditional probability — if we know $P(A \mid B)$, we can compute $P(B \mid A)$.

### Step-by-Step Derivation

**Step 1 — Start with the two conditional probability equations:**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

**Step 2 — Rearrange the second equation to express the joint probability:**

$$P(A \cap B) = P(B \mid A) \times P(A)$$

**Step 3 — Substitute the joint probability expression into the first equation:**

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$$

**Result — Bayes' Theorem:**

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

### Components

| Term | Name | Role |
|---|---|---|
| $P(A \mid B)$ | ***Posterior*** | *Updated belief in $A$ after observing $B$* |
| $P(B \mid A)$ | ***Likelihood*** | *How probable is $B$ if $A$ is true* |
| $P(A)$ | ***Prior*** | *Belief in $A$ before observing any evidence* |
| $P(B)$ | ***Evidence*** | *Normalizing constant — total probability of $B$* |

**Bayes' Theorem** is the ***mathematical foundation*** of *Bayesian statistics* and is widely used in **spam filtering**, **medical diagnosis**, and **probabilistic machine learning**.

> For a full treatment with worked examples, see [Bayes Theorem in Probability](Bayes%20Theorem%20in%20Probability.md).

---

## 10. Why Is Conditional Probability Important in Machine Learning?

Conditional probability is ***fundamental*** to several machine learning algorithms — whenever a model makes a prediction, it is almost always computing ***the probability of an output given an input***.

| **Machine Learning Concept** | **How Conditional Probability is Used** |
|---|---|
| ***Naive Bayes*** | Calculates the probability of a class given input features: $P(\text{class} \mid \mathbf{x})$ |
| ***Bayesian Networks*** | Models dependencies between variables as a graph of conditional probabilities |
| ***Hidden Markov Models*** | Models probabilities of hidden states given a sequence of observations |
| ***Bayesian Inference*** | Updates prior beliefs using new evidence via Bayes' Theorem |
| ***Probabilistic Classification*** | Predicts class probabilities given input data (logistic regression, softmax, etc.) |
| ***Language Models (GPT, etc.)*** | Generates the next token using $P(\text{word}_n \mid \text{word}_1, \ldots, \text{word}_{n-1})$ |
| ***Reinforcement Learning*** | Estimates $P(\text{reward} \mid \text{state, action})$ to guide policy learning |

> Virtually every supervised learning algorithm outputs $P(y \mid \mathbf{x})$ — the probability of label $y$ ***given*** input features $\mathbf{x}$. That vertical bar is conditional probability.

---

## 11. Applications in Machine Learning

### Naive Bayes Classifier

Naive Bayes classifies documents by computing:

$$P(C \mid \mathbf{x}) \propto P(C) \cdot \prod_{i=1}^{n} P(x_i \mid C)$$

where $C$ is the class and $\mathbf{x} = (x_1, \ldots, x_n)$ are features. Every feature probability $P(x_i \mid C)$ is a conditional probability. This is used in spam detection, sentiment analysis, and text categorization.

### Classification with Probabilistic Outputs

Any classifier that outputs a probability score — logistic regression, softmax in neural networks, random forests — is estimating:

$$P(\text{class} \mid \text{features})$$

This is conditional probability: the probability of the label *given* the input features.

### Recommendation Systems

Collaborative filtering estimates:

$$P(\text{user likes item} \mid \text{user's past behavior})$$

Knowing what a user has liked before conditions the probability of liking new items.

### Natural Language Processing

Language models learn:

$$P(\text{next word} \mid \text{previous words})$$

Autoregressive generation (GPT-style) is simply repeated application of conditional probability over a vocabulary.

### Other Domains

| Domain | Conditional Probability Used |
|---|---|
| **Finance** | $P(\text{default} \mid \text{credit score, income})$ |
| **Healthcare** | $P(\text{disease} \mid \text{test result, symptoms})$ |
| **Weather** | $P(\text{rain tomorrow} \mid \text{today's conditions})$ |
| **Marketing** | $P(\text{purchase} \mid \text{browsing history})$ |

---

## 12. Practice Problems

**Problem 1:** A bag contains 3 red balls and 5 blue balls. Two balls are drawn without replacement. What is the probability that the second ball is red, given the first was red?

<details>
<summary>Solution</summary>

After removing one red ball: 2 red remain, 7 total remain.

$$P(\text{2nd red} \mid \text{1st red}) = \frac{2}{7} \approx 0.286$$

</details>

---

**Problem 2:** In a class, 60% of students study math and 40% study both math and science. If a student studies math, what is the probability they also study science?

<details>
<summary>Solution</summary>

$$P(\text{Science} \mid \text{Math}) = \frac{P(\text{Math} \cap \text{Science})}{P(\text{Math})} = \frac{0.40}{0.60} = \frac{2}{3} \approx 0.667$$

</details>

---

**Problem 3:** A test is 99% accurate (both sensitivity and specificity). A disease has prevalence 0.1% (1 in 1000). If you test positive, what is the probability you have the disease?

<details>
<summary>Solution</summary>

$$P(\text{Disease}) = 0.001, \quad P(+ \mid \text{Disease}) = 0.99, \quad P(+ \mid \text{No Disease}) = 0.01$$

$$P(+) = (0.99)(0.001) + (0.01)(0.999) = 0.00099 + 0.00999 = 0.01098$$

$$P(\text{Disease} \mid +) = \frac{0.00099}{0.01098} \approx 9\%$$

Even with a 99%-accurate test, a positive result only means ~9% chance of having a rare disease. Base rates dominate.

</details>

---

**Problem 4 — Multiple Choice:** A college has 100 students. 50 know Python, 40 know Java, and 20 know both. What is $P(\text{Python} \mid \text{Java})$?

- A) 20%
- B) 40%
- C) 50%
- D) 80%

<details>
<summary>Solution</summary>

**Correct: C) 50%**

$$P(\text{Python} \mid \text{Java}) = \frac{P(\text{Python} \cap \text{Java})}{P(\text{Java})} = \frac{20/100}{40/100} = \frac{20}{40} = 0.5 = 50\%$$

Students who know **both** form the intersection. Conditioning on Java shrinks the sample space to just those 40 Java students, and 20 of them also know Python.

</details>

---

**Problem 5 — Multiple Choice:** If $P(A) = 0.6$ and $P(B) = 0.5$, and $A$ and $B$ are **independent**, what is $P(A \cap B)$?

- A) 0.1
- B) 0.3
- C) 0.5
- D) 1.1

<details>
<summary>Solution</summary>

**Correct: B) 0.3**

For independent events, the joint probability is simply the product of the individual probabilities:

$$P(A \cap B) = P(A) \times P(B) = 0.6 \times 0.5 = 0.3$$

Note: option D (1.1) is impossible — probabilities can never exceed 1.

</details>

---

## 13. Quick Revision — Important Formulas

> Use this table as a **cheat sheet** — all the key formulas in one place.

| Concept | Formula |
|---|---|
| ***Conditional Probability*** | $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$ |
| ***Multiplication Rule*** | $P(A \cap B) = P(A \mid B) \cdot P(B)$ |
| ***Symmetric Form*** | $P(A \cap B) = P(A) \cdot P(B \mid A) = P(B) \cdot P(A \mid B)$ |
| ***Independent Events*** | $P(A \cap B) = P(A) \cdot P(B)$ |
| ***Independence Condition*** | $P(A \mid B) = P(A)$ |
| ***Complement Rule*** | $P(A' \mid B) = 1 - P(A \mid B)$ |
| ***Bayes' Theorem*** | $P(A \mid B) = \dfrac{P(B \mid A) \cdot P(A)}{P(B)}$ |
| ***Law of Total Probability*** | $P(A) = \sum_n P(A \mid B_n) \cdot P(B_n)$ |
| ***Chain Rule*** | $P(A_1 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2 \mid A_1) \cdot P(A_3 \mid A_1 \cap A_2) \cdots$ |
| ***Addition Rule (Conditional)*** | $P((A \cup B) \mid X) = P(A \mid X) + P(B \mid X) - P(A \cap B \mid X)$ |

---

## 14. Connection to Naive Bayes Classifier

The Naive Bayes Classifier classifies a new data point $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ by finding the class $C$ that maximizes the **posterior probability**:

$$\hat{C} = \arg\max_C P(C \mid \mathbf{x})$$

Applying Bayes' Theorem:

$$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$

Since $P(\mathbf{x})$ is constant across all classes, it is dropped:

$$\hat{C} = \arg\max_C P(C) \cdot P(\mathbf{x} \mid C)$$

Applying the **conditional independence assumption** (the "naive" part):

$$\hat{C} = \arg\max_C P(C) \cdot \prod_{i=1}^{n} P(x_i \mid C)$$

Every term $P(x_i \mid C)$ is a **conditional probability** — the probability of observing feature $x_i$ given the class $C$.

| Component | Mathematical Object | Where it comes from |
|---|---|---|
| $P(C)$ | Prior | Frequency of each class in training data |
| $P(x_i \mid C)$ | Likelihood | Conditional probability of each feature given class |
| $P(C \mid \mathbf{x})$ | Posterior | What we want to maximize |

> For the full derivation, see 
> [***Naive Bayes Classifier — §2 Mathematical Derivation***](Naive%20Bayes%20Classifier.md#2-mathematical-derivation-of-naive-bayes-classifier).

---

## 15. References

### Within This Module

- **Next Part**: [Independent and Dependent Events in Probability](Independent%20and%20dependent%20Events%20in%20Probability.md)
- [Mutually Exclusive Events in Probability](Mutually%20Exclusive%20Events%20in%20Probability.md)
- [Bayes Theorem in Probability](Bayes%20Theorem%20in%20Probability.md)
- [Naive Bayes Classifier — Complete Theory with Mathematical Derivation](Naive%20Bayes%20Classifier.md#2-mathematical-derivation-of-naive-bayes-classifier)

---

### Full Reference — Mathematics of Machine Learning

For the complete deep-dive — including all visual diagrams, advanced properties, the full Chain Rule derivation, and all ML applications — see:

> **[Conditional Probability — Mathematics of Machine Learning](../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/README.md)**

This is the primary reference document for all theory covered in this section. The folder also contains:

- [Conditional Probability Advanced Concepts](../Mathematics%20of%20Machine%20Learning/Conditional%20Probability/Conditional%20Probability%20Advanced%20Concept.md)

### External References

- [Wikipedia — Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability)
- [GeeksforGeeks — Conditional Probability](https://www.geeksforgeeks.org/maths/conditional-probability/)
- [Math Is Fun — Conditional Probability](https://www.mathsisfun.com/data/probability-events-conditional.html)
- Kolmogorov, A. N. (1933). *Foundations of the Theory of Probability*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 1

---

<p align="center">
  *(Start of Chapter)* &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Next:</strong> <a href="Independent%20and%20dependent%20Events%20in%20Probability.md">Independent and Dependent Events in Probability</a> ➡️
</p>
