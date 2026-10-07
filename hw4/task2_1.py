import numpy as np
w1 = 2
w2 = 1
a = np.array([0.05, 0.09])
def loss(w1, w2):
    L = w1 ** 2 + 10 * (w2**2)
    return L
def grad(w1 , w2):
    g1 = 2 * w1
    g2 = 20 * w2
    return g1, g2
def update(w1, w2, a):
    g1, g2 = grad(w1, w2)
    w1 = w1 - a * g1
    w2 = w2 - a*g2
    return w1, w2
w1_new , w2_new = update(w1,w2,a)
L_new = loss(w1_new, w2_new)
print("Old Loss:",loss(w1,w2))
print("New Loss:",L_new)
print("Gradient:",grad(w1,w2))
print("Update:",update(w1,w2,a))
