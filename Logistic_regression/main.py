

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Create classification data
# ---------------------------

np.random.seed(0)

n = 200
x = np.random.randn(n)

true_m = 3
true_c = -1

z = true_m * x + true_c
y = (z > 0).astype(int)   # class labels: 0 or 1

# ---------------------------
# 2. Normalize x
# ---------------------------

mean_x = np.mean(x)
std_x = np.std(x)
x = (x - mean_x) / std_x

# ---------------------------
# 3. Sigmoid
# ---------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------
# 4. Initialize model
# ---------------------------

m = 0.0
c = 0.0

alpha = 0.1
epochs = 1000
losses = []

# ---------------------------
# 5. Training loop
# ---------------------------

for epoch in range(epochs):
    z = m * x + c
    p = sigmoid(z)

    # Binary cross-entropy loss
    loss = -np.mean(y*np.log(p + 1e-9) + (1-y)*np.log(1-p + 1e-9))
    losses.append(loss)

    # Gradients
    dm = np.mean(x * (p - y))
    dc = np.mean(p - y)

    # Update
    m -= alpha * dm
    c -= alpha * dc

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---------------------------
# 6. Plot loss
# ---------------------------

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training Loss")
plt.show()

# ---------------------------
# 7. Decision boundary
# ---------------------------

# Convert weights back to raw-x space
m_real = m / std_x
c_real = c - (m * mean_x) / std_x

x_plot = np.linspace(min(x)*std_x + mean_x,
                     max(x)*std_x + mean_x, 200)

z_plot = m_real * x_plot + c_real
p_plot = sigmoid(z_plot)

# ---------------------------
# 8. Visualization
# ---------------------------

plt.scatter(x*std_x + mean_x, y, c=y, cmap="bwr", alpha=0.6)
plt.plot(x_plot, p_plot, color="black", linewidth=2)
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Logistic Regression (From Scratch)")
plt.show()
