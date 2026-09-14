import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
def f(w):
    return w[0] **2 + 3*w[0] *w[1]+2*w[1] **2 
def grad_analytic(w):
    grad_w0 = 2*w[0] + 3*w[1] 
    grad_w1 =  3*w[0] + 4*w[1]
    return np.array([grad_w0,grad_w1])
def number_gradient (f,w,h = 1e-5):
    g= np.zeros_like(w, dtype = float)
    for i in range (len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += h
        w_minus[i] -=h
        g[i] = (f(w_plus) - f(w_minus))/(2*h)
    return g
w = np.array([1.0, 2.0])
ga = grad_analytic(w)
gn = number_gradient(f, w, h = 1e-5)
print("f(w)  = ", f(w))
print("analytic grad = ", ga)
print("number gradient =", gn)
print("max abs diff =", np.max(np.abs(ga - gn)))
eps = 1e-3
d_grad = ga / np.linalg.norm(ga)
gain_grad = f(w + eps * d_grad)-f(w)
gains_random = []
for _ in range(20):
    d= rng.normal(size = 2)
    d /= np.linalg.norm(d)
    gains_random.append(f(w + eps *d)-f(w))
print("gain along gradient :", gain_grad)
print("best random gain    :", max(gains_random))
assert np.allclose(ga, [8.0, 11.0])
assert np.max(np.abs(ga - gn)) < 1e-6
assert gain_grad >= max(gains_random) - 1e-9

print("Task 2 OK")
