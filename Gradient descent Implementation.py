# Single-variable linear regression trained using Gradient descent

import pandas as pd
import numpy as np

# Load dataset from the project directory
df = pd.read_csv("penguins.csv")
df = df[["body_mass_g", "bill_length_mm"]].dropna()

x = np.array(df["body_mass_g"].tolist())
y = np.array(df["bill_length_mm"].tolist())


# scaled_x = (x - mean) / std
# ^^^ I'm not going to stress coding standardization for now 
# going to use learning rate "a = 1e-8" to compensate 
# I'm mainly focusing on the machinery of implementing gradient descent

# the model's initial parameters
w = 0
b = 0
parameters = np.array([w, b])
print(f"Initial w: {w:.0f}")
print(f"Initial b: {b:.0f}")

# our model: y-hat = m * x + b 
def prediction(w, x, b):
    return w * x + b

# small learning rate due to the large scale of body_mass_g
learning_rate = 1e-8
tolerance = 1e-5
max_iterations = 1000
current_mse = None
mse_history = []


for i in range(max_iterations):

    predictions = prediction(w, x, b)

    previous_mse = current_mse

    # calculate error
    errors = predictions - y
    current_mse = np.mean(errors**2)
    mse_history.append(current_mse)
    
    print(f"\n===== LOOP NUMBER: {i + 1} =====")
    print(f"Current Average MSE: {current_mse:.4f}")
    print(f"Current Parameters  -> w: {w:.6f}, b: {b:.6f}")

    # calculate gradient 
    dw = 2 * np.mean(errors * x)
    db = 2 * np.mean(errors)
    gradient = np.array([dw, db])

    # gradient descent step
    parameters = parameters - learning_rate * gradient
    w = parameters[0]
    b = parameters[1]

    print(f"w: {w:.4f}, b: {b:.4f}")
    print("===========================")

    # checks for convergence
    if previous_mse is not None:
        mse_change = abs(current_mse - previous_mse)

        if mse_change < tolerance:
            print(f"Optimization has effectively converged around iteration {i + 1}.")
            break
