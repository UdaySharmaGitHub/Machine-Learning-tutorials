# Machine Learning Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-UdaySharmaGitHub-181717?logo=github&style=for-the-badge)](https://github.com/UdaySharmaGitHub)

A comprehensive, hands-on collection of Machine Learning tutorials built with Jupyter Notebooks. This repository covers core ML algorithms **from scratch** — starting from mathematical derivations and building up to full implementations using Python and scikit-learn.

Whether you're a beginner or looking to solidify your fundamentals, this repo walks you through each concept step by step with code, visualizations, and real-world datasets.

---

## Topics Covered

| # | Topic | Description |
|---|-------|-------------|
| 01 | **Introduction** | Introduction to Machine Learning concepts |
| 02 | **Complete Linear Regression** | Simple, standard, and multiple linear regression — theory, math, and implementation |
| 03 | **Gradient Descent** | End-to-end gradient descent from scratch, 3D visualizations, and all GD variants (Batch, Stochastic, Mini-Batch) |
| 04 | **Polynomial Regression** | Polynomial regression from scratch and comparison with linear models |
| 05 | **Regularization (Ridge / L2)** | Bias-variance tradeoff, Ridge regression theory, scratch implementations, and gradient descent approach |
| 06 | **Lasso Regression (L1)** | Lasso regression demo and key concepts |
| 07 | **Elastic Net Regression** | Combining L1 and L2 regularization |
| 08 | **Logistic Regression** | Logistic regression, perceptron trick, and sigmoid function with animations |

---

## Folder Structure & Notebook Links

### 📁 01 Intro

> Introduction to Machine Learning concepts.

---

### 📁 02 Complete Linear Regression

#### 📂 01 Simple Linear Regression

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Practical Simple Linear Regression](02%20Complete%20Linear%20Regression/01%20Simple%20Linear%20Regression/00%20Practical%20Simple%20Linear%20Regression.ipynb) | Practical walkthrough of simple linear regression |
| 01 | [Simple Linear Regression](02%20Complete%20Linear%20Regression/01%20Simple%20Linear%20Regression/01_Simple_Linear_Regression.ipynb) | Theory and implementation of SLR |
| 02 | [Custom Simple Linear Regression from Scratch](02%20Complete%20Linear%20Regression/01%20Simple%20Linear%20Regression/02_Custom_Simple_Linear_Regression_from_scratch.ipynb) | SLR built from scratch using NumPy |
| 03 | [Regression Metrics](02%20Complete%20Linear%20Regression/01%20Simple%20Linear%20Regression/03_Regression_metrices.ipynb) | MSE, RMSE, MAE, R² and other evaluation metrics |

> **Dataset:** [`placement.csv`](02%20Complete%20Linear%20Regression/01%20Simple%20Linear%20Regression/placement.csv)

#### 📂 02 Linear Regression

| # | Notebook | Description |
|---|----------|-------------|
| — | [Linear Regression ML Implementation](02%20Complete%20Linear%20Regression/02%20Linear%20Regression/Linear%20Regression%20ML%20Implementation.ipynb) | Linear regression using scikit-learn |
| — | [Linear Regression Practicals](02%20Complete%20Linear%20Regression/02%20Linear%20Regression/Linear%20Regression%20Practicals.ipynb) | Hands-on practice problems |

#### 📂 03 Multi Linear Regression (MLR)

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Complete Derivation — Multiple Linear Regression](02%20Complete%20Linear%20Regression/03%20Multi%20Linear%20Regression%20(MLR)/00_Complete_Derivation_Muliple_Linear_Regression.ipynb) | Full mathematical derivation of MLR |
| 01 | [MLR Using Random Data](02%20Complete%20Linear%20Regression/03%20Multi%20Linear%20Regression%20(MLR)/01_Mulitple_Linear_Regression_using_random_data.ipynb) | MLR on synthetic data |
| 02 | [MLR on Diabetes Dataset](02%20Complete%20Linear%20Regression/03%20Multi%20Linear%20Regression%20(MLR)/02_MLR_on_multiple_features_of_diabetes_dataset.ipynb) | MLR with multiple features from diabetes dataset |
| 04 | [MLR — Economic Index Dataset](02%20Complete%20Linear%20Regression/03%20Multi%20Linear%20Regression%20(MLR)/04_MLR_Economic_index_dataset.ipynb) | MLR on economic index data |

> **Datasets:** [`Student_Performance.csv`](02%20Complete%20Linear%20Regression/03%20Multi%20Linear%20Regression%20(MLR)/Student_Performance.csv) · [`economic_index.csv`](02%20Complete%20Linear%20Regression/03%20Multi%20Linear%20Regression%20(MLR)/economic_index.csv)

---

### 📁 03 Gradient Descent

#### 📂 01 Gradient Descent end-to-end From Scratch

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Gradient Descent Intro](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/00_Gradient_descent_intro.ipynb) | Introduction to gradient descent |
| 01 | [Gradient Descent Step by Step](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/01_Gradient_descent_step_by_step.ipynb) | Step-by-step walkthrough |
| 01.1 | [Gradient Descent Step by Step (Custom)](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/01.1_gradient_descent_step_by_step_custom.ipynb) | Custom implementation step-by-step |
| 02 | [Gradient Descent Code from Scratch](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/02_Gradient_descent_code_from_scratch.ipynb) | Full GD implementation from scratch |
| 03 | [GD Animation (only b)](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/03_gradient-descent-animation(onlyb).ipynb) | Animated GD convergence for intercept |
| 04 | [GD Animation (both m and b)](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/04_gradient-descent-animation(both-m-and-b).ipynb) | Animated GD convergence for slope & intercept |
| 05 | [Gradient Descent 3D](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/05_gradient-descent-3d.ipynb) | 3D cost surface visualization |

> **Interactive Plots:** [`cost_function.html`](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/cost_function.html) · [`cost_function2.html`](03%20Gradient%20Descent/01%20Gradient%20Descent%20end-to-end%20From%20Sractch/cost_function2.html)

#### 📂 02 Types of Gradient Descent

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Batch Gradient Descent](03%20Gradient%20Descent/02%20Types%20of%20Gradient_descent/01_Batch_Gradient_Descent.ipynb) | Batch GD implementation |
| 02 | [Stochastic Gradient Descent from Scratch](03%20Gradient%20Descent/02%20Types%20of%20Gradient_descent/02_Stochastic_Gradient_Descent_from_scratch.ipynb) | SGD built from scratch |
| 02.2 | [Stochastic Gradient Descent Animation](03%20Gradient%20Descent/02%20Types%20of%20Gradient_descent/02.2_Stochastic_Gradient_Descent_animation.ipynb) | SGD convergence animations |
| 03 | [Mini-Batch Gradient Descent from Scratch](03%20Gradient%20Descent/02%20Types%20of%20Gradient_descent/03_Mini_Batch_Gradient_Descent_from_srcatch.ipynb) | Mini-Batch GD implementation |

---

### 📁 04 Polynomial Regression

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Polynomial Regression from Scratch & Comparison](04%20Polynomial%20Regression/00_Polynomial_Regression_from_srcatch_and_comparision.ipynb) | Polynomial regression from scratch with linear comparison |
| 01 | [Polynomial Regression Implementation](04%20Polynomial%20Regression/01_Polynomial_Regression_Implementation.ipynb) | scikit-learn polynomial regression |

---

### 📁 05 Regularization in Machine Learning (Ridge / L2)

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Bias-Variance Tradeoff](05%20Regularization%20in%20Machine%20Learning/00_Bias_Variance_Trade_off.ipynb) | Understanding bias-variance tradeoff |
| 01 | [Ridge Regularization](05%20Regularization%20in%20Machine%20Learning/01_Ridge_Regularization.ipynb) | Ridge (L2) regularization theory & demo |
| 02 | [Ridge Regression from Scratch (m and b)](05%20Regularization%20in%20Machine%20Learning/02_Ridge_Regression_from_scratch_for_both_m_and_b.ipynb) | Ridge regression from scratch for slope & intercept |
| 03 | [Ridge Regression from Scratch](05%20Regularization%20in%20Machine%20Learning/03_Ridge_regression_from_Scratch.ipynb) | Ridge regression scratch implementation |
| 04 | [Ridge Regression — Gradient Descent](05%20Regularization%20in%20Machine%20Learning/04_Ridge_Regression-Gradient-Descent.ipynb) | Ridge regression using gradient descent |
| 05 | [Ridge Regression Key Understandings](05%20Regularization%20in%20Machine%20Learning/05_Ridge_Regression_Key_Understandings.ipynb) | Key takeaways and insights on Ridge |

---

### 📁 06 Lasso Regression (L1 Regularization)

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Lasso Regression Demo](06%20Lasso%20Regression%20or%20L1%20Regularization/01_lasso_regression_Demo.ipynb) | Lasso regression demonstration |
| 02 | [Lasso Regression Key Points](06%20Lasso%20Regression%20or%20L1%20Regularization/02_lasso_regression_key_points.ipynb) | Key concepts and feature selection behavior |

---

### 📁 07 Elastic Net Regression

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Elastic Net Regression](07%20Elasticnet%20Regression/01_Elastic_Regression.ipynb) | Combining L1 + L2 regularization |

---

### 📁 08 Logistic Regression

| # | Notebook | Description |
|---|----------|-------------|
| 00 | [Logistic Regression](08%20Logistic%20Regression/00_Logistic_Regression.ipynb) | Logistic regression theory & implementation |
| 01 | [Perceptron Logistic Regression](08%20Logistic%20Regression/01_Preceptron_Logistic_Regresssion.ipynb) | Perceptron approach to logistic regression |
| 02 | [Perceptron Trick Using Sigmoid](08%20Logistic%20Regression/02_preceptron_trick_using_Sigmoid.ipynb) | Sigmoid-based perceptron trick |

---

## Highlights

- **From-scratch implementations** — Most algorithms are coded from the ground up using NumPy before using scikit-learn, so you understand the math behind the magic.
- **Mathematical derivations** — Full derivations included (e.g., multiple linear regression closed-form solution, gradient descent update rules).
- **Visualizations & animations** — Gradient descent convergence animations, 3D cost surfaces, contour plots, and more.
- **Real-world datasets** — Practice with datasets like placement data, diabetes dataset, economic index, and student performance.
- **Regression metrics** — Dedicated notebook covering evaluation metrics (MSE, RMSE, MAE, R²).

---

## Tech Stack

- **Python 3**
- **Jupyter Notebook**
- **NumPy** — numerical computation
- **Pandas** — data manipulation
- **Matplotlib / Plotly** — visualization & interactive plots
- **scikit-learn** — ML model implementations

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Machine\ Learning
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib plotly scikit-learn jupyter
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

4. **Navigate to any topic folder** and open the notebooks in order (00, 01, 02, ...).

---

## How to Use This Repo

Each topic is organized in a numbered folder. Within each folder, notebooks are numbered sequentially — **start from `00` or `01`** and work your way through. The progression follows a logical learning path:

> Linear Regression → Gradient Descent → Polynomial Regression → Regularization (Ridge → Lasso → Elastic Net) → Logistic Regression

---

## Contributing

Contributions, suggestions, and improvements are welcome! Feel free to open an issue or submit a pull request.

---

## Author

**Uday Sharma** — [@UdaySharmaGitHub](https://github.com/UdaySharmaGitHub)

Feel free to check out my other repositories and follow for more!

---

## License

This project is open-source and available for educational purposes.
