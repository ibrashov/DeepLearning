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
    print("Starting loss:", history[0])
    for step in range(n_updates):
        dw, db = gradients(w, b)
        w = w - lr * dw
        b = b - lr * db
        path.append([w, b])
        history.append(mse(w, b))
    return np.array(path), np.array(history)
path, history = gd(0.05, 1)
print("Path:",path)
print("History:",history)
print("w =", path[1, 0])
print("b =", path[1, 1])
print("loss =", history[1])
path, history = gd(0.05, 100)
updates = [0, 1, 2, 5, 20, 100]
x_line = np.linspace(0, 4, 100)
plt.figure(figsize=(10, 7))
plt.scatter(x,y,s=80,label="Data points")
for step in updates:
    w = path[step, 0]
    b = path[step, 1]
    y_line = w * x_line + b
    plt.plot(x_line,y_line,label=f"Update {step}: w={w:.3f}, b={b:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent Fitting a Line")
plt.legend()
plt.grid()
plt.show()