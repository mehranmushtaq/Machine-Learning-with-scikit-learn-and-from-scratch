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

## Project Steps

The following steps were performed in this project:

Data loading and exploration

Data preprocessing and feature selection

Gender encoding

Train–test split

Feature scaling using StandardScaler

Training a Logistic Regression model

Model evaluation using classification metrics

Hyperparameter tuning using GridSearchCV

Data visualization using Seaborn and Matplotlib

## Model Performance

The Logistic Regression model achieved approximately:
Accuracy: 88%
Precision: 93%

The confusion matrix and classification report were used to evaluate performance in detail.

## Visualizations
The project includes visualizations to better understand the dataset and model behavior, such as:

Age vs Estimated Salary purchase distribution

Estimated Salary vs Purchase decision comparison by gender

These plots help illustrate how user demographics relate to purchasing behavior.

## Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

## Learning Outcome
Through this project, the following concepts were practiced:
Logistic Regression for classification
Feature scaling importance
Model evaluation metrics (precision, recall, F1-score)
Hyperparameter tuning with cross-validation
Exploratory data visualization
