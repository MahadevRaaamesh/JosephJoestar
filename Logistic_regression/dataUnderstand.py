import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

n = 100

# Class 0 (low traffic)
x0 = np.random.randn(n, 2) + np.array([2, 2])
y0 = np.zeros((n, 1))

# Class 1 (high traffic)
x1 = np.random.randn(n, 2) + np.array([7, 7])
y1 = np.ones((n, 1))

# Combine
X = np.vstack((x0, x1))
y = np.vstack((y0, y1))

# Normalize the 2D features
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X_normalized = (X - mean_X) / std_X

# Reshape y to be 1D for consistent calculations
y = y.flatten()


# Plot the initial 2D data
# Plot
plt.scatter(x0[:,0], x0[:,1], color="blue", label="Low Traffic (0)")
plt.scatter(x1[:,0], x1[:,1], color="red", label="High Traffic (1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Binary Classification Data")
plt.show()

#---- Model settings and initialization ----#
epochs = 10000
alpha = 0.1

# Initialize weights (m) for two features and bias (c)
m = np.zeros(X_normalized.shape[1]) # m will be a vector [m1, m2]
c=0

losses=[]

#-- Sigmoid fn --#

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#---- binary cross loss ----#

def log_loss(y, p):
    # Add a small epsilon to prevent log(0)
    epsilon = 1e-9
    return -np.mean(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))

#---- training ----#

for _ in range(epochs):
    # Linear combination for 2D data: z = X @ m + c
    z = X_normalized @ m + c
    p = sigmoid(z)

    # Gradients for 2D data
    # dm is now a vector of gradients for each weight
    dm = (1 / n) * X_normalized.T @ (p - y)
    dc = (1 / n) * np.sum(p - y)

    m -= alpha * dm
    c -= alpha * dc

    current_loss = log_loss(y, p)
    losses.append(current_loss)
    if _ % 1000 == 0:
        print(f"Epoch {_}, Loss: {current_loss:.4f}")

#---- Plot Loss ----#
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training Loss (2D Logistic Regression)")
plt.show()

#---- Final Visualization with Decision Boundary ----#

# Convert weights back to original feature space for plotting the decision boundary
m_real = m / std_X
c_real = c - np.sum(m * mean_X / std_X)

# The decision boundary is where m1*x1 + m2*x2 + c = 0
# So, x2 = (-m1/m2)*x1 - (c/m2)
x_boundary = np.array([X[:, 0].min(), X[:, 0].max()])
y_boundary = (-m_real[0] / m_real[1]) * x_boundary - (c_real / m_real[1])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.6, label="Actual Data")
plt.plot(x_boundary, y_boundary, color="black", linewidth=2, label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("2D Logistic Regression (From Scratch)")
plt.show()