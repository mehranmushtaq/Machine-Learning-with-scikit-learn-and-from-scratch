## Machine Learning Algorithms From Scratch

This directory contains Machine Learning algorithms implemented from scratch using NumPy only, without using high-level libraries like scikit-learn.

The goal of this project is to deeply understand the mathematics and inner workings of Machine Learning algorithms.

## Algorithms Implemented

## Linear Regression (Gradient Descent)

File: linear_reg.ipynb

Linear Regression models the relationship between input features **X** and target values **y**.

## Implementation includes:
	•	Gradient Descent Optimization
	•	Mean Squared Error Loss
	•	Weight and Bias Updates
	•	Multi-feature support

Model Equation:

```
y = wX + b
```

## Features:

✔ Gradient Descent optimization

✔ Bias and weights calculation

✔ Prediction function

✔ Vectorized NumPy operations

## Key Concepts:

	•	Loss minimization
	•	Parameter updates
	•	Convergence
  
## Ordinary Least Squares (OLS Method)

File: linear_reg.ipynb

OLS finds the best-fit line analytically by minimizing squared error.

## Implementation includes:
	Ordinary Least Squares
	Normal Equation solution

OLS Formula:

```
β = (XᵀX)^(-1)Xᵀy
```

## Features:

✔ Analytical solution

✔ No iterations required

✔ Fast computation

✔ Matrix operations

## Key Concepts:
	•	Exact solution
	•	Matrix algebra
	•	No iterations required

## Logistic Regression

File: logistic_reg.ipynb

## Implementation includes:
	•	Gradient Descent Optimization
	•	Sigmoid Function
	•	Binary Classification
	•	Probability Prediction

Sigmoid Function:
```
σ(z) = 1 / (1 + e^(-z))
```
## Key Concepts:
	•	Classification
	•	Decision Boundary
	•	Probability estimation

