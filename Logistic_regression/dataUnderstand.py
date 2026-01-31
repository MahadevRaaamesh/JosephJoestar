import numpy as np
import matplotlib.pyplot as plt

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

# Plot
plt.scatter(x0[:,0], x0[:,1], color="blue", label="Low Traffic (0)")
plt.scatter(x1[:,0], x1[:,1], color="red", label="High Traffic (1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Binary Classification Data")
plt.show()


#---- sigmoid and stuff ----#

x = np.random.randn(200)

true_m = 3
true_c = -1

z = true_m * x + true_c
y = (z > 0).astype(int)   


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

p = sigmoid(m * x + c)

y_pred = (p > 0.5).astype(int)

#---- binary cross loss ----#

def log_loss(y, p):
    return -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))


#---- training ----#

for _ in range(epochs):
    z = m*x + c
    p = sigmoid(z)

    dm = np.mean(x * (p - y))
    dc = np.mean(p - y)

    m -= alpha * dm
    c -= alpha * dc