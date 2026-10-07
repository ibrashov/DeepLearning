import numpy as np
steps = [0.1, 0.2, 0.24]
w =1.5
stepp = 0
for step in steps:
    stepp +=1
    def loss(w):
        L= ((w ** 2)-1)**2
        return L
    def grad(w):
        g = 4 * (w**3) - 4 * w
        return g
    def update(w, step):
        g = grad(w)
        w_new = w - step*g 
        return w_new
    
    L_new = loss(w)
    g = grad(w)
    up = update(w,step)
    print("step ", stepp)
    print("L_new:", L_new)
    print("Grad:", g)
    print("Upgrade:", up)