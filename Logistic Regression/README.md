## Social Network Ads Purchase Prediction — Logistic Regression

This project demonstrates a machine learning classification problem using Logistic Regression to predict whether a user will purchase a product based on their age, estimated salary, and gender.
The goal of this project is to practice the complete machine learning workflow including data preprocessing, feature scaling, model training, evaluation, visualization, and hyperparameter tuning.

## Dataset

The project uses the Social Network Ads dataset containing user demographic information and purchase decisions.

## Input features:

Age

Estimated Salary

Gender

## Target variable:
Purchased (0 = No, 1 = Yes)

## Data Visualization

Exploratory visualizations are created before training the model to understand feature relationships.

Visualizations include:

	•	Age vs Salary scatter plots
	•	Purchase distribution plots
	•	Feature relationship charts

These help identify patterns influencing purchase decisions.

## Data Preprocessing

Steps performed:

	•	Encoding categorical variables
	•	Train-test split
	•	Feature scaling using StandardScaler

## Pipeline Implementation

A Scikit-Learn Pipeline is used to combine:

	•	Feature scaling
	•	Logistic Regression model

This ensures a clean and reproducible workflow while preventing data leakage.

## Hyperparameter Tuning

GridSearchCV is used to tune model parameters.

Parameters tuned include:

	•	Regularization strength (C)
	•	Penalty type
	•	Solver

Cross-validation improves model generalization.

## Model Evaluation

The model is evaluated using:

	•	Accuracy
	•	Confusion Matrix
	•	Precision
	•	Recall
	•	F1 Score

Classification performance is analyzed using these metrics.


## Key Learning Outcomes:

	•	Logistic Regression for classification
	•	Pipeline implementation
	•	GridSearchCV hyperparameter tuning
	•	Feature scaling importance
	•	Classification metrics interpretation
	•	Visualization for classification problems
