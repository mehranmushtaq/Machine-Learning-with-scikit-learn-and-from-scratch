## Machine Learning

This repository contains implementations of fundamental machine learning algorithms using Python and Scikit-Learn, along with data preprocessing, feature engineering, visualization, pipelines, cross-validation, regularization, and model evaluation.

The goal of this repository is to build a strong practical foundation in machine learning by working with real datasets and complete ML workflows.

## Repository Structure
```
Machine-Learning-with-scikit-learn/
│
├── Datasets/
│   ├── insurance.csv
│   ├── Social_Network_Ads.csv
│   ├── house_price_practice.csv
│   └── Iris.csv
│
├── Linear Regression/
│   ├── linear_regression.ipynb
│   └── README.md
│
├── Logistic Regression/
│   ├── logistic_regression.ipynb
│   └── README.md
│
├── KNN/
│   ├── knn_iris_classification.ipynb
│   └── README.md
│
├── Regularization (Lasso, Ridge)/
│   ├── lasso_ridge.ipynb
│   └── README.md
│
└── README.md

```
## Machine Learning Workflow Implemented

This repository demonstrates a complete end-to-end ML pipeline:

	•	Data preprocessing
	•	Feature engineering
	•	Data visualization
	•	Pipeline creation
	•	Model training
	•	GridSearchCV
	•	Regularization (L1 / L2)
	•	Cross-validation
	•	Model evaluation

## Data Preprocessing

	•	Handling missing values
	•	Encoding categorical variables
	•	Feature scaling
	•	Train-test split

## Feature Engineering

Feature engineering techniques used in this repository include:

	•	One-hot encoding
	•	Creating interaction terms for combined feature effects
	•	Feature selection
	•	Removing irrelevant features

These steps help improve model performance and interpretability.

## Data Visualization

Visualization is performed before and after model training to better understand the dataset and model performance.

Libraries used:

	•	Matplotlib
	•	Seaborn

## Visualizations include:

	•	Correlation heatmaps
	•	Distribution plots
	•	Scatter plots
	•	Feature relationship plots
	•	Actual vs Predicted plots
	•	Residual analysis
	•	Model prediction visualization

Scatter plots and other charts are used to understand relationships between variables and to evaluate prediction performance.

## Machine Learning Models ##

## Linear Regression

Dataset used: House Price Practice

Used for regression problems such as prediction tasks.

Concepts covered:

	•	Model training using Scikit-Learn
	•	Predictions
	•	Residual analysis

Evaluation metrics:

	•	R² Score
	•	Adjusted R²
	•	Mean Squared Error (MSE)



## Logistic Regression

Dataset used: Social Network Ads

This section includes a complete classification workflow using Pipeline and GridSearchCV.

Concepts covered:

	•	Sigmoid function
	•	Decision boundary
	•	Binary classification
	•	Probability prediction
	•	Pipeline implementation
	•	Hyperparameter tuning using GridSearchCV

Evaluation metrics:

	•	Accuracy
	•	Confusion Matrix
	•	Precision
	•	Recall
	•	F1 Score

## Regularization (Lasso and Ridge)

Dataset used: insurance.csv

Regularization is implemented to reduce overfitting and improve generalization.

Concepts covered:

	•	Lasso Regression (L1 regularization)
	•	Ridge Regression (L2 regularization)
	•	Feature shrinkage
	•	Bias-variance tradeoff
	•	Model comparison
	•	Cross-validation

Evaluation metrics:

	•	Cross-validation score
	•	R² Score
	•	Mean Squared Error (MSE)

Predicted vs actual value visualizations are used to evaluate model performance.

## K-Nearest Neighbors (KNN)

Dataset used: Iris Dataset

This section demonstrates a classification workflow using the KNN algorithm with Scikit-learn pipelines.

Concepts covered:

	•	Feature scaling using StandardScaler
	•	KNeighborsClassifier
	•	Pipeline creation
	•	Hyperparameter tuning with GridSearchCV
	•	Model evaluation

Evaluation metrics:

	•	Accuracy
	•	Precision
	•	Recall
	•	Confusion Matrix

## Cross Validation and Pipelines

This repository uses:

	•	K-Fold Cross Validation
	•	Scikit-Learn Pipelines
	•	GridSearchCV for hyperparameter tuning

Pipelines are used to combine preprocessing and modeling steps into a single workflow, improving reproducibility and reducing data leakage.

## Tools and Libraries
	•	Python
	•	NumPy
	•	Pandas
	•	Matplotlib
	•	Seaborn
	•	Scikit-Learn
	•	Jupyter Notebook

## Learning Objective

This repository is part of my machine learning learning journey.

The focus is on:

	•	Understanding ML algorithms conceptually
	•	Implementing models using Scikit-Learn
	•	Practicing with real datasets
	•	Applying feature engineering techniques
	•	Using pipelines and GridSearchCV
	•	Building portfolio-ready ML projects

## Future Additions

	•	Naive Bayes
	•	Decision Trees
	•	Ensemble methods
	•	Model deployment
