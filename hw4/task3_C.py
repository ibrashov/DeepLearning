import numpy as np
import matplotlib.pyplot as plt
x = np.array([0., 1., 2., 3., 4.])
y = np.array([1., 3., 5., 7., 9.])
def mse(w, b):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    return loss
def gradients(w, b):
    y_hat = w * x + b
    error = y_hat - y
    dw = (2 / len(x)) * np.sum(error * x)
    db = (2 / len(x)) * np.sum(error)
    return dw, db
def gd(lr, n_updates):
    w = 0.0
    b = 0.0
    path = [[w, b]]
    history = [mse(w, b)]
    for step in range(n_updates):
        dw, db = gradients(w, b)
        w = w - lr * dw
        b = b - lr * db
        path.append([w, b])
        history.append(mse(w, b))
    return np.array(path), np.array(history)
path, history = gd(0.05, 1)
print("Starting loss:", history[0])
print("Path:",path)
print("History:",history)
print("w =", path[1, 0])
print("b =", path[1, 1])
print("loss =", history[1])
w_values = np.linspace(-0.5, 4.5, 200)
b_values = np.linspace(-0.5, 2.5, 200)
W, B = np.meshgrid(w_values, b_values)
L = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        L[i, j] = mse(W[i, j], B[i, j])
learning_rates = [0.01, 0.05, 0.14]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, lr in zip(axes, learning_rates):
    path, history = gd(lr, 100)
    ax.contour(W,B,L,levels=30)
    ax.plot(path[:, 0],path[:, 1],marker="o",markersize=3,label=f"lr = {lr}")
    ax.scatter(0,0,marker="s",s=80,label="Start (0, 0)")
    ax.scatter(2,1,marker="*",s=200,label="Minimum (2, 1)")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.legend()
loss_learning_rates = [0.01, 0.05, 0.14, 0.16]
plt.figure(figsize=(10, 6))
for lr in loss_learning_rates:
    path, history = gd(lr, 100)
    plt.plot(history,label=f"lr = {lr}")
plt.yscale("log")
plt.xlabel("Update")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()