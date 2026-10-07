import numpy as np
import matplotlib.pyplot as plt
w_0 = np.array([2.0, 1.0])
a = np.array([0.01, 0.05, 0.09, 0.11])
def loss(w_0):
    L = w_0[0] ** 2 + 10 * (w_0[1] ** 2)
    return L
def grad(w_0):
    g1 = 2 * w_0[0]
    g2 = 20 * w_0[1]
    return g1, g2
def update(w_0, a):
    g1, g2 = grad(w_0)
    w1 = w_0[0] - a * g1
    w2 = w_0[1] - a * g2
    return np.array([w1, w2])
paths = []
losses = []
for eta in a:
    w = w_0.copy()
    path = [w.copy()]
    loss_path = [loss(w)]
    for step in range(30):
        w = update(w, eta)
        path.append(w.copy())
        loss_path.append(loss(w))
    path = np.array(path)
    loss_path = np.array(loss_path)
    paths.append(path)
    losses.append(loss_path)
for i in range(len(a)):
    print("eta =", a[i])
    print("final w =", paths[i][-1])
    print("final loss =", losses[i][-1])
    print("final gradient =", grad(paths[i][-1]))
    print()
w1, w2 = np.meshgrid(np.linspace(-2.5, 2.5, 200),np.linspace(-1.5, 1.5, 200))
z = w1**2 + 10 * w2**2
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
for i in range(len(a)):
    run_path = paths[i]
    ax[i].contour(w1, w2, z, colors="lightgray")
    ax[i].plot(run_path[:, 0], run_path[:, 1], marker="o", markersize=3)
    ax[i].plot(w_0[0], w_0[1], "ko", label="start")
    ax[i].plot(0, 0, "kx", label="min")
    ax[i].set_title("eta = " + str(a[i]))
    ax[i].set_xlabel("w1")
    ax[i].set_ylabel("w2")
    ax[i].set_xlim(-2.5, 2.5)
    ax[i].set_ylim(-1.5, 1.5)
    ax[i].grid()
    ax[i].legend()
plt.tight_layout()
plt.show()
for i in range(len(a)):
    plt.plot(losses[i], label="eta = " + str(a[i]))
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()