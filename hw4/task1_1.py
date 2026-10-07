import numpy as np
a = 0.1
w =0.5
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
for step in range(2):
    g = grad(w)
    w_new = update(w,a)
    L = loss(w)
    L_new = loss(w_new)
    print("step:", step+1)
    print("grad:", g)
    print("update:", w_new)
    print("loss_old:", L)
    print("loss_new:", L_new)
    print()
    print("grad w=-1:", grad(-1))
    print("grad w=0:", grad(0))
    print("grad w=1:", grad(1))
    print()
    print("update w=-1:", update(-1,a))
    print("update w=0:", update(0,a))
    print("update w=1:", update(1,a))
    print()
    print("loss w=-1:", loss(-1))
    print("loss w=0:", loss(0))
    print("loss w=1:", loss(1))
    w = w_new
