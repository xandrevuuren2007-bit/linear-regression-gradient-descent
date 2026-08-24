# Linear Regression From Scratch With Gradient Descent

A first-principles implementation of linear regression and gradient descent using Python and NumPy.

## What this project implements

- Linear regression
- Mean squared error
- Partial derivatives
- Gradient calculation
- Parameter updates using gradient descent
- Convergence detection
- MSE history logging

## Dataset

The project uses the Palmer Penguins dataset and predicts bill length from body mass.

## Notes

Feature standardization is intentionally not implemented in this version. Because body mass is relatively large, a small learning rate (1e-8) is used to keep the gradient descent updates stable.

The purpose of this project is to understand and implement the machinery of gradient descent rather than produce an optimized machine-learning library.

## Requirements

- Python
- NumPy
- Pandas

## Running

Place `penguins.csv` in the project directory and run the python file
