# Conditional Probability — Advanced Concepts

> [!NOTE]
> **Reference:** Based on [Wikipedia — Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability)

## Contents

1. [As the Probability of a Conditional Event](#1-as-the-probability-of-a-conditional-event)
2. [Conditioning on an Event of Probability Zero](#2-conditioning-on-an-event-of-probability-zero)
3. [Conditioning on a Discrete Random Variable](#3-conditioning-on-a-discrete-random-variable)
4. [Partial Conditional Probability](#4-partial-conditional-probability)
5. [Use in Inference](#5-use-in-inference)
6. [Information, Conditional Probability, and Statistical Independence](#6-information-conditional-probability-and-statistical-independence)
7. [Statistical Independence](#7-statistical-independence)
8. [Common Fallacies](#8-common-fallacies)
9. [Formal Derivation](#9-formal-derivation)
10. [References](#10-references)

---

## 1. As the Probability of a Conditional Event

- Conditional probability can be defined as the **probability of a conditional event** $A_B$. 
- The ***Goodman–Nguyen–Van Fraassen*** conditional event can be defined as:

$$A_B = \bigcup_{i \geq 1} \left( \bigcap_{j < i} \bar{B}_j,\ A_i \cap B_i \right)$$

- where $A_i$ and $B_i$ represent states or elements of $A$ or $B$.

It can be shown that:

$$\boxed{P(A_B) = \frac{P(A \cap B)}{P(B)}}$$

- which meets the **Kolmogorov definition** of conditional probability.

---

## 2. Conditioning on an Event of Probability Zero

- If $P(B) = 0$, then according to the definition, $P(A \mid B)$ is **undefined**.

- The case of greatest interest is that of a ***random variable*** $Y$, conditioned on a **continuous random variable** $X$ resulting in a particular outcome $x$. 
- The event $B = \{X = x\}$ has **probability zero** and, as such, ***cannot be conditioned on***.

Instead of conditioning on $X$ being exactly $x$, we can condition on it being ***closer than distance*** $\varepsilon$ away from $x$. The event:

$$B = \{x - \varepsilon < X < x + \varepsilon\}$$

- will generally have **nonzero probability** and can therefore be conditioned on. We then take the limit:

$$\lim_{\varepsilon \to 0} P\!\left(A \mid x - \varepsilon < X < x + \varepsilon\right) \tag{1}$$

- For example, if two continuous random variables $X$ and $Y$ have a **joint density** $f_{X,Y}(x, y)$, then by ***L'Hôpital's rule*** and the ***Leibniz integral rule***, upon differentiation with respect to $\varepsilon$:

$$\boxed{\lim_{\varepsilon \to 0} P\!\left(Y \in U \mid x_0 - \varepsilon < X < x_0 + \varepsilon\right) = \frac{\displaystyle\int_U f_{X,Y}(x_0,\, y)\, dy}{\displaystyle\int_\mathbb{R} f_{X,Y}(x_0,\, y)\, dy}}$$

- The resulting limit is the ***conditional probability distribution*** of $Y$ given $X$, and exists when the denominator — the **probability density** $f_X(x_0)$ — is ***strictly positive***.

> [!WARNING]
> It is tempting to define $P(A \mid X = x)$ using limit (1), but this ***cannot be done consistently***. It is possible to find random variables $X$ and $W$ and values $x$, $w$ such that the events $\{X = x\}$ and $\{W = w\}$ are *identical* but the resulting limits are ***not equal***:
>
> $$\lim_{\varepsilon \to 0} P(A \mid x - \varepsilon \leq X \leq x + \varepsilon) \neq \lim_{\varepsilon \to 0} P(A \mid w - \varepsilon \leq W \leq w + \varepsilon)$$
>
> The ***Borel–Kolmogorov paradox*** demonstrates this with a geometrical argument.

---

## 3. Conditioning on a Discrete Random Variable

> [!NOTE]
> *See also: Conditional probability distribution, Conditional expectation, Regular conditional probability*

Let $X$ be a **discrete random variable** with possible outcomes denoted $V$. For example, if $X$ represents the value of a rolled die, then $V = \{1, 2, 3, 4, 5, 6\}$.

- For a value $x \in V$ and an event $A$, the conditional probability is given by $P(A \mid X = x)$.
- Writing $c(x, A) = P(A \mid X = x)$, we see that it is a **function of two variables** — $x$ and $A$.
- For a fixed $A$, we can form the **random variable** $Y = c(X, A)$, which represents an outcome of $P(A \mid X = x)$ whenever a value $x$ of $X$ is observed.

The **conditional probability of $A$ given $X$** can thus be treated as a random variable $Y$ with outcomes in $[0, 1]$. From the ***law of total probability***, its **expected value** is equal to the ***unconditional probability*** of $A$.

---

## 4. Partial Conditional Probability

The **partial conditional probability** $P(A \mid B_1 \equiv b_1, \ldots, B_m \equiv b_m)$ is the probability of event $A$ given that each condition event $B_i$ has occurred to a ***degree*** $b_i$ (degree of belief, degree of experience) that might be **different from 100%**.

- ***Frequentistically***, partial conditional probability makes sense if the conditions are tested in experiment repetitions of appropriate length $n$.
- Such ***$n$-bounded partial conditional probability*** is defined as the conditionally expected average occurrence of event $A$ in testbeds of length $n$ that adhere to all probability specifications $B_i \equiv b_i$:

$$\boxed{P_n(A \mid B_1 \equiv b_1, \ldots, B_m \equiv b_m) = \mathbb{E}\!\left(\bar{A}_n \mid \bar{B}_{1n} = b_1, \ldots, \bar{B}_{mn} = b_m\right)}$$

Based on that, **partial conditional probability** can be defined as:

$$\boxed{P(A \mid B_1 \equiv b_1, \ldots, B_m \equiv b_m) = \lim_{n \to \infty} P_n(A \mid B_1 \equiv b_1, \ldots, B_m \equiv b_m)}, \quad \text{where } b_i \cdot n \in \mathbb{N}$$

***Jeffrey conditionalization*** is a **special case** of partial conditional probability, in which the condition events must form a ***partition***:

$$\boxed{P(A \mid B_1 \equiv b_1, \ldots, B_m \equiv b_m) = \sum_{i=1}^{m} b_i\, P(A \mid B_i)}$$

---

## 5. Use in Inference

In **statistical inference**, the conditional probability is an ***update of the probability*** of an event based on ***new information***. The new information can be incorporated as follows:

- Let $A$ — the **event of interest** — be in the sample space $(X, P)$.
- The occurrence of event $A$ ***knowing that event $B$ has occurred*** means the occurrence of $A$ restricted to $B$, i.e. $A \cap B$.
- Without knowledge of $B$, the information about $A$ would simply be $P(A)$.
- The probability of $A$ **knowing that** $B$ has occurred is the probability of $A \cap B$ *relative to* $P(B)$:

$$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}} \quad \text{whenever } P(B) > 0$$

This results in a probability measure that is ***consistent with the original*** probability measure and satisfies all **Kolmogorov axioms**.

> [!TIP]
> **Bayesian interpretation:** The term *"evidence"* or *"information"* is used in the **Bayesian framework**. The conditioning event is interpreted as ***evidence*** for the conditioned event. $P(A)$ is the probability of $A$ *before* accounting for evidence $E$, and $P(A \mid E)$ is the probability *after* updating on $E$.

### Example — Morse Code Transmission

- When **Morse code** is transmitted, there is a probability that the *"dot"* or *"dash"* received is ***erroneous*** due to interference. The key question: given that a dot was *received*, what is the probability that a dot was *sent*?

$$\boxed{P(\text{dot sent} \mid \text{dot received}) = \frac{P(\text{dot received} \mid \text{dot sent}) \cdot P(\text{dot sent})}{P(\text{dot received})}}$$

**Given information:**

- In Morse code, the ratio of dots to dashes at the point of sending is **3:4**:
  - $P(\text{dot sent}) = \dfrac{3}{7}$ and $P(\text{dash sent}) = \dfrac{4}{7}$
- Probability that a **dot** is transmitted as a **dash** = $\dfrac{1}{10}$
- Probability that a **dash** is transmitted as a **dot** = $\dfrac{1}{10}$

**Step 1 — Find $P(\text{dot received})$:**

$$P(\text{dot received}) = P(\text{dot received} \mid \text{dot sent}) \cdot P(\text{dot sent}) + P(\text{dot received} \mid \text{dash sent}) \cdot P(\text{dash sent})$$

$$= \frac{9}{10} \times \frac{3}{7} + \frac{1}{10} \times \frac{4}{7} = \frac{27}{70} + \frac{4}{70} = \frac{31}{70}$$

**Step 2 — Apply Bayes' Theorem:**

$$\boxed{P(\text{dot sent} \mid \text{dot received}) = \frac{\dfrac{9}{10} \times \dfrac{3}{7}}{\dfrac{31}{70}} = \frac{27}{31} \approx 0.871}$$

---

## 6. Information, Conditional Probability, and Statistical Independence

The concepts of **conditional probability** and ***statistical independence*** can be understood through the idea of how ***new information changes an individual's uncertainty*** about an event.

### Example 1 — Dependent Events (Six-Sided Die)

Consider rolling a fair **six-sided die** with sample space $S = \{1, 2, 3, 4, 5, 6\}$:

- **Event A** = outcome is even → $A = \{2, 4, 6\}$, $\quad P(A) = \dfrac{3}{6} = \dfrac{1}{2}$
- **Event B** = outcome is $\leq 3$ → $B = \{1, 2, 3\}$, $\quad P(B) = \dfrac{3}{6} = \dfrac{1}{2}$
- **Intersection:** $A \cap B = \{2\}$, $\quad P(A \cap B) = \dfrac{1}{6}$

**Check for independence:** $P(A) \cdot P(B) = \dfrac{1}{2} \times \dfrac{1}{2} = \dfrac{1}{4} \neq \dfrac{1}{6} = P(A \cap B)$

Since $P(A \cap B) \neq P(A) \cdot P(B)$, events **A** and **B** are ***statistically dependent***.

Now suppose we learn that event $A$ has occurred. The sample space ***shrinks*** from $S$ to $A = \{2, 4, 6\}$. Within this space, $B$ can only occur at outcome 2, so:

$$\boxed{P(B \mid A) = \frac{P(A \cap B)}{P(A)} = \frac{1/6}{1/2} = \frac{1}{3}}$$

The probability of $B$ ***changes*** from $\dfrac{1}{2}$ to $\dfrac{1}{3}$ — confirming **dependence**. Knowledge of $A$ ***reduces uncertainty*** and alters the probability of $B$.

### Example 2 — Independent Events (Eight-Sided Die)

Consider rolling a fair **eight-sided die** with sample space $S = \{1, 2, 3, 4, 5, 6, 7, 8\}$:

- **Event A** = outcome $< 5$ → $A = \{1, 2, 3, 4\}$, $\quad P(A) = \dfrac{4}{8} = \dfrac{1}{2}$
- **Event B** = outcome $\in \{3, 4, 5, 6\}$ → $P(B) = \dfrac{4}{8} = \dfrac{1}{2}$
- **Intersection:** $A \cap B = \{3, 4\}$, $\quad P(A \cap B) = \dfrac{2}{8} = \dfrac{1}{4}$

**Check for independence:** $P(A) \cdot P(B) = \dfrac{1}{2} \times \dfrac{1}{2} = \dfrac{1}{4} = P(A \cap B)$ ✓

Now suppose we learn event $A$ has occurred. The space shrinks to $\{1, 2, 3, 4\}$, within which $B$ occurs at $\{3, 4\}$:

$$\boxed{P(B \mid A) = \frac{P(A \cap B)}{P(A)} = \frac{1/4}{1/2} = \frac{1}{2} = P(B)}$$

The probability of $B$ ***does not change*** — $P(B \mid A) = P(B)$. Learning $A$ provides ***no information*** about $B$, demonstrating **statistical independence**.

### Interpretation

**Statistical independence** can be interpreted as the ***absence of informational value*** between events:

- If knowledge of one event ***changes*** the probability of another → events are **dependent**
- If the probability ***remains unchanged*** after obtaining new information → events are **independent**

Formally, two events $A$ and $B$ are ***independent if and only if***:

$$\boxed{P(B \mid A) = P(B)} \quad \Longleftrightarrow \quad \boxed{P(A \cap B) = P(A) \cdot P(B)}$$

---

## 7. Statistical Independence

> [!NOTE]
> **Main article:** [Independence (probability theory)](https://en.wikipedia.org/wiki/Independence_(probability_theory))

Events $A$ and $B$ are ***statistically independent*** if:

$$\boxed{P(A \cap B) = P(A) \cdot P(B)}$$

- If $P(B) \neq 0$, this is equivalent to: $P(A \mid B) = P(A)$
- If $P(A) \neq 0$, this is also equivalent to: $P(B \mid A) = P(B)$

> [!NOTE]
> The ***preferred definition*** uses $P(A \cap B) = P(A) \cdot P(B)$ because it is **symmetrical** in $A$ and $B$, and avoids undefined conditional probabilities when $P(A) = 0$ or $P(B) = 0$. **Independence is not the same as** disjoint events.

**Conditional Independence:** Given event pair $[A, B]$ and an event $C$, the pair is ***conditionally independent*** given $C$ if:

$$\boxed{P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)}$$

This is useful in applications where ***multiple independent events*** are observed simultaneously.

### Independent vs. Mutually Exclusive Events

The concepts of **mutually independent** and **mutually exclusive** events are ***separate and distinct***:

| Scenario | $P(A \mid B)$ | $P(B \mid A)$ | $P(A \cap B)$ |
|---|---|---|---|
| ***Statistically Independent*** | $P(A)$ | $P(B)$ | $P(A) \cdot P(B)$ |
| ***Mutually Exclusive*** | $0$ | $0$ | $0$ |

> [!IMPORTANT]
> Mutually exclusive events ***cannot*** be statistically independent (unless both are impossible), since knowing one occurs means the other ***certainly will not***.

### Independent vs. Positive vs. Negative Association

| Relation between $A$ and $B$ | Condition | Result |
|---|---|---|
| ***Independent*** | $P(A \cap B) = P(A) \cdot P(B)$ | $P(B \mid A) = P(B)$ |
| ***Positive Association*** | $P(A \cap B) > P(A) \cdot P(B)$ | $P(B \mid A) > P(B)$ |
| ***Negative Association*** | $P(A \cap B) < P(A) \cdot P(B)$ | $P(B \mid A) < P(B)$ |

---

## 8. Common Fallacies

> [!CAUTION]
> These fallacies should not be confused with Robert K. Shope's 1978 ***"conditional fallacy"***, which deals with counterfactual examples that beg the question.

### Fallacy 1 — Assuming $P(A \mid B) \approx P(B \mid A)$

> [!NOTE]
> **Main article:** [Confusion of the inverse](https://en.wikipedia.org/wiki/Confusion_of_the_inverse)

In general, it ***cannot*** be assumed that $P(A \mid B) \approx P(B \mid A)$. This is an ***insidious error***, even for those highly conversant with statistics.

The relationship is given by **Bayes' Theorem**:

$$\boxed{P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A)}} \qquad \Longleftrightarrow \qquad \frac{P(B \mid A)}{P(A \mid B)} = \frac{P(B)}{P(A)}$$

Therefore, $P(A \mid B) \approx P(B \mid A)$ ***only if*** $P(B) \approx P(A)$.

### Fallacy 2 — Assuming $P(A) \approx P(A \mid B)$

In general, it ***cannot*** be assumed that the **marginal probability** $P(A)$ and the **conditional probability** $P(A \mid B)$ are similar. These are linked through the ***law of total probability***:

$$\boxed{P(A) = \sum_n P(A \cap B_n) = \sum_n P(A \mid B_n) \cdot P(B_n)}$$

where the events $(B_n)$ form a **countable partition** of $\Omega$.

This fallacy may arise through ***selection bias***. A medical example:

- Let $S_C$ = event that a **sequela** (chronic disease) $S$ occurs due to circumstance $C$
- Let $H$ = event that the individual seeks **medical help**
- Suppose $C$ rarely causes $S$, so $P(S_C)$ is **low**
- But suppose medical attention is *only sought* when $S$ has occurred

A doctor seeing only patients who sought help may ***erroneously conclude*** $P(S_C)$ is high — when what they observe is actually $P(S_C \mid H)$, not $P(S_C)$.

### Fallacy 3 — Over- or Under-Weighting Priors

- ***Base rate neglect*** — not taking **prior probability** into account partially or completely
- ***Conservatism*** — **insufficient adjustment** from the prior probability

---

## 9. Formal Derivation

$P(A \mid B)$ is formally defined as the **probability of $A$ according to a new probability function** on the sample space, such that:

- Outcomes ***not in $B$*** have probability $0$
- It is ***consistent*** with all original probability measures

Let $\Omega$ be a **discrete sample space** with elementary events $\{\omega\}$, and $P$ the probability measure with respect to the $\sigma$-algebra of $\Omega$. Suppose event $B \subseteq \Omega$ has occurred.

A **new probability distribution** must satisfy — for some scale factor $\alpha$:

$$\omega \in B:\quad P(\omega \mid B) = \alpha\, P(\omega) \tag{1}$$

$$\omega \notin B:\quad P(\omega \mid B) = 0 \tag{2}$$

$$\sum_{\omega \in \Omega} P(\omega \mid B) = 1 \tag{3}$$

**Substituting (1) and (2) into (3) to solve for $\alpha$:**

$$1 = \alpha \sum_{\omega \in B} P(\omega) = \alpha \cdot P(B) \quad \Longrightarrow \quad \alpha = \frac{1}{P(B)}$$

**So the new probability distribution is:**

$$\boxed{\omega \in B:\quad P(\omega \mid B) = \frac{P(\omega)}{P(B)}, \qquad \omega \notin B:\quad P(\omega \mid B) = 0}$$

**For a general event $A$:**

$$P(A \mid B) = \sum_{\omega \in A \cap B} P(\omega \mid B) + \underbrace{\sum_{\omega \in A \cap B^c} P(\omega \mid B)}_{= \, 0} = \sum_{\omega \in A \cap B} \frac{P(\omega)}{P(B)} = \frac{P(A \cap B)}{P(B)}$$

$$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}}$$

This completes the **formal derivation** of the conditional probability formula from the ***Kolmogorov axioms***.

---

## 10. References

- [Wikipedia — Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability)
- [Back to Core Concept of Conditional Probability](README.md)
