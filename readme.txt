Overview
This project implements Ridge Regression to analyze the relationship between various features and diabetes progression. It explores the impact of different regularization parameters (lambda values) on model performance, comparing mean squared error (MSE) for training and test data.

Features
Implementation of Ridge Regression from scratch

Analysis of MSE for different lambda values

Comparison of model performance with and without intercept

Visualization of weight distributions for different lambda values

Requirements
Python 3.x

NumPy

Matplotlib

Usage
Clone the repository

Run the main script: python ridge_regression_analysis.py

Results
The project generates plots showing:

MSE vs lambda for training and test data

Weight distributions for different lambda values

Key findings are printed to the console, including optimal lambda values and corresponding MSE.

Future Work
Implement cross-validation for more robust model evaluation

Explore other regularization techniques like Lasso and Elastic Net

Analyze feature importance based on ridge regression coefficients