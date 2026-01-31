import numpy as np
import matplotlib.pyplot as plt

#----initial values----#
n = int(input("No Of Points:"))
zero_to_r = int(input("Range From Zero:"))
true_m = int(input("Slope:"))
true_c = int(input("Y-Intercept:"))

#----generating randon x values----#

x = np.random.rand(n) * zero_to_r

#----creation of noise----#

noise = np.random.randn(n) * 2 

#---- normalization ----#
meanx=np.mean(x)
stdx=np.std(x)

x=(x-np.mean(x))/np.std(x)

#--- Y values ---#

true_y = true_m * x + true_c + noise

#----Plot the data----#

plt.scatter(x, true_y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Linear Data")
plt.show()


#----Baseline model----#

m = 0
c = np.mean(true_y)

#----Baseline Model prediction----#

y_pred = m * x + c

#----Baseline vs Actual data plot----#

plt.scatter(x, true_y, label="Actual Data")
plt.plot(x, y_pred, color="red", label="Model Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initial Random Model")
plt.legend()
plt.show()

#---- Loss FN ----#

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
 
#---- Regression ----#

alpha = 0.01
epochs = 100

losses = []

for epoch in range(epochs):
    
    y_pred = m * x + c
    
    #---- Loss ----#

    loss = mse_loss(true_y, y_pred)
    losses.append(loss)
    
    #---- Gradients ----#
    
    dm = (-2 / n) * np.sum(x * (true_y - y_pred))
    dc = (-2 / n) * np.sum(true_y - y_pred)
    
    #---- Descent (updation)----#
    
    m -= alpha * dm
    c -= alpha * dc

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

#---- Final Model Values----#

x=(x*stdx)+meanx
idx = np.argsort(x)
x_sorted = x[idx]

m_real = m / stdx
c_real = c - (m * meanx) / stdx

y_line = m_real * x_sorted + c_real

print("#---#")
print("Final m :", m)
print("Final c :", c)
print("True m :", true_m)
print("True c:", true_c)
print("#---#")

#---- Final Model Vs Actual Value Plot ----#

plt.scatter(x, true_y, label="Actual Data")
plt.plot(x_sorted, y_line, color="red", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.show()

#---- losses curve ----#

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss vs Epochs")
plt.show()