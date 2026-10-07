import numpy as np
import matplotlib.pyplot as plt
a = np.array([0.1, 0.1, 0.1, 0.2, 0.24])
w = np.array([0.5, -0.5, 1.5, 1.5, 1.5])
def loss(w):
    L= ((w ** 2)-1)**2
    return L
def grad(w):
    g = 4 * (w**3) - 4 * w
    return g
def update(w, a):
    g = grad(w)
    w = w - a*g 
    return w
w_start = w.copy()
points = [w.copy()]
for step in range(30):
    w = update(w,a)
    points.append(w.copy())
points = np.array(points)
print("Shape:", points.shape)
print("Final w:", w)
print("Final loss:", loss(w))
w_grid = np.linspace(-1.8, 1.8, 200)
fig, ax = plt.subplots(1,5,figsize=(20, 4))
for i in range(5):
    run_points = points[:, i]
    ax[i].plot(w_grid,loss(w_grid),color="lightgray")
    ax[i].plot(run_points,loss(run_points),"o-")
    ax[i].set_ylim(-0.2, 2.5)
    ax[i].set_xlabel("w")
    ax[i].set_ylabel("L(w)")
    ax[i].set_title(f"w0={points[0, i]}, eta={a[i]}")
plt.tight_layout()
plt.show()
    
