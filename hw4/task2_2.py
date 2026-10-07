import numpy as np
import matplotlib.pyplot as plt
w = np.array([2.0, 1.0])
a = 0.05
def loss(w):
    L = w[0] ** 2 + 10 * (w[1]**2)
    return L
def grad(w):
    g1 = 2 * w[0]
    g2 = 20 * w[1]
    return g1, g2
def update(w,a):
    g1, g2 = grad(w)
    w1 = w[0] - a * g1
    w2 = w[1]- a*g2
    return w1,w2
w1_new , w2_new = update(w,a)
path = [w1_new, w2_new]
L_new = loss(path)
g = np.array(grad(w))
gd = -g
zero = np.array([0.0, 0.0])
direct = zero - w
print(loss(w))
print(L_new)
print(grad(w))
print(update(w,a))
w1, w2 = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-1.5, 1.5, 200))
plt.contour(w1, w2, w1**2 + 10 * w2**2, levels=15, colors="lightgray")
plt.plot(w[0], w[1], "ko", label= "[2,1] start")
plt.plot(0, 0, "kx", label= "[0,0] min")
plt.arrow(w[0], w[1], gd[0], gd[1], width = 0.02, label = "-g")
plt.arrow(w[0],w[1],direct[0],direct[1], width= 0.02, label ="direction" )
plt.xlabel("w1")
plt.ylabel("w2")
plt.title("GD direction vs direction to minimum")
plt.xlim(-2.5, 2.5)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.legend()
plt.show()
