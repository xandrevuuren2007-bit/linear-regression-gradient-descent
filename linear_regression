# Multiple linear regression trained using vectorised gradient descent

import pandas as pd
import numpy as np

# Load dataset from the project directory
df = pd.read_csv("penguins.csv")
df = df[[
    "body_mass_g",
    "bill_length_mm", 
    "bill_depth_mm", 
    "flipper_length_mm"
    ]].dropna()

X = df[[
    "bill_length_mm" , 
    "bill_depth_mm", 
    "flipper_length_mm"
    ]].to_numpy()
print(X.shape)

y = df["body_mass_g"].to_numpy()

# I'm not going to stress coding standardization/regularizing for now 
# going to use learning rate "a = 1e-8" to temporarily ~compensate 
# I'm mainly focusing on the machinery of implementing gradient descent

# the model's initial params
w = np.zeros(3)
b = 0
# parameters = np.array([w, b])
print(f"Initial w: {w}")
print(f"Initial b: {b}")

# model: y-hat = Xw + b 
def prediction(X, w, b):
    return np.dot(X, w) + b

learning_rate = 0.1
tolerance = 1e-5
max_iterations = 1000
current_mse = None
mse_history = []

for i in range(max_iterations):

    predictions = prediction(X, w, b)

    previous_mse = current_mse

    # calculate error
    errors = predictions - y
    current_mse = np.mean(errors**2)
    mse_history.append(current_mse)
    
    print(f"\n===== LOOP NUMBER: {i + 1} =====")
    print(f"Current Average MSE: {current_mse:.4f}")
    print(f"Current Parameters  -> w: {w}, b: {b}")

    # calculate gradient 
    #dw = 2 * np.mean(errors * x)
    n = X.shape[0]   
    dw = (2 / n) * (X.T @ errors)

    db = 2 * np.mean(errors)

    # gradient descent step
    w = w - learning_rate * dw
    b = b - learning_rate * db

    print(f"w: {w}, b: {b}")
    print("===========================")

    # checks for convergence
    if previous_mse is not None:
        mse_change = abs(current_mse - previous_mse)

        if mse_change < tolerance:
            print(f"Optimization has effectively converged around iteration {i + 1}.")
            break

# for debugging
print("X:", X.shape)
print("w:", w.shape)
print("predictions:", predictions.shape)
print("errors:", errors.shape)
print("dw:", dw.shape)
