import numpy as np
a = 0.1
w =-0.5
def loss(w):
    L= ((w ** 2)-1)**2
    return L
def grad(w):
    g = 4 * (w**3) - 4 * w
    return g
def update(w, a):
    g = grad(w)
    w_new = w - a*g 
    return w_new
L_new = loss(w)
g = grad(w)
up = update(w,a)
print("L_new:", L_new)
print("Grad:", g)
print("Upgrade:", up)