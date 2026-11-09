- **[Multiclass ROC-AUC](#multiclass-roc-auc)**
  - [Can We Use ROC-AUC with Softmax Regression?](#can-we-use-roc-auc-with-softmax-regression)

---
## Multiclass ROC-AUC

**ROC-AUC** can also evaluate the performance of a Softmax Regression model.

> [!NOTE]
> **Multiclass ROC-AUC** requires a strategy such as **One-vs-Rest (OvR)** or
> **One-vs-One (OvO)** because a standard ROC curve compares *two classes*.

### Can We Use ROC-AUC with Softmax Regression?

**Yes.** For *multiclass classification*, we commonly use:

- **One-vs-Rest (OvR)**
- **One-vs-One (OvO)**

For example, with Cat, Dog, and Horse, OvR evaluates:

1. Cat vs. (Dog + Horse)
2. Dog vs. (Cat + Horse)
3. Horse vs. (Cat + Dog)

- Calculate an individual **ROC-AUC value** for each comparison.
- Aggregate the results using **macro averaging** or **weighted averaging**.