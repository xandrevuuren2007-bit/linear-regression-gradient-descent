# Linear Regression From Scratch

Built to understand the mathematical and computational machinery behind machine learning.

## About

This project is an experiment in implementing linear regression from scratch using Python, NumPy, and Pandas.

The goal was not to build the most optimized regression model, but to understand what is actually happening underneath machine learning libraries.

I started with a simple linear regression model and manually implemented gradient descent before extending the experiment to multiple linear regression.

## Current Implementation

The current model uses multiple linear regression with three input features:

- Bill length
- Bill depth
- Flipper length

The target variable is:

- Body mass

The model is:

`ŷ = Xw + b`

Trained using gradient descent.

For the weights:

`dw = (2/n) Xᵀ(Xw - y)`

For the bias:

`db = 2 mean(Xw + b - y)`

The parameters are updated using:

`w = w - αdw`

`b = b - αdb`

## Technologies

- Python
- NumPy
- Pandas
- Git / GitHub

## Dataset

This experiment uses the Palmer Penguins dataset.

Rows containing missing values in these variables are removed before training.

Place `penguins.csv` in the project directory and run the Python file.

## Learning Progression

Experiment 1: Basic Linear Regression

Experiment 2: Implementing Gradient Descent

Experiment 3: Multiple Linear Regression

- `ŷ = wx + b` → `ŷ = Xw + b`

This introduced vectors, matrices, transposes, dot products, and vectorized gradient calculations.

## Purpose

The purpose of this project is educational.
