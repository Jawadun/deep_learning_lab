import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("ultimate_anime_dataset.csv")

# Select numerical features
features = [
    "episodes",
    "popularity_anilist",
    "mal_members",
    "duration_per_ep"
]

target = "rating_anilist"

# Remove missing values
df = df[features + [target]].dropna()

# Input (X) and Target (T)
X = df[features].values
T = df[[target]].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias input column
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Initialize weights
np.random.seed(42)
W = np.random.randn(X.shape[1], 1) * 0.01

b = 0.0
learning_rate = 0.01
epochs = 2000

for epoch in range(epochs):

    # Forward pass
    Z = np.dot(X, W) + b
    Y = 1 / (1 + np.exp(-Z))

    # Scale target to [0,1]
    T_scaled = T / 10

    # Loss
    loss = 0.5 * np.mean((Y - T_scaled) ** 2)

    # Backpropagation
    dL_dY = Y - T_scaled
    dY_dZ = Y * (1 - Y)
    dL_dZ = dL_dY * dY_dZ

    dL_dW = np.dot(X.T, dL_dZ) / len(X)
    dL_dB = np.mean(dL_dZ)

    # Update
    W -= learning_rate * dL_dW
    b -= learning_rate * dL_dB

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss:.6f}"
        )

print("\nTraining Complete")
print("Final Loss:", loss)

# Predict first anime
sample = X[0:1]
prediction = 1 / (1 + np.exp(-(np.dot(sample, W) + b)))

print("Actual Rating:", T[0][0])
print("Predicted Rating:", prediction[0][0] * 10)