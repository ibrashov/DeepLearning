import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def forward(w,b,x,y):
    z = w*x + b
    a = sigmoid(z)
    L = (a - y)**2
    return z ,a ,L
def grad_analytic(w,b,x,y):
    z,a,L =forward(w,b,x,y)
    dL_da = 2*(a-y)
    da_dz = sigmoid(z) * (1-sigmoid(z))
    dz_db = 1
    dz_dw = x
    dL_dw = dL_da * da_dz * dz_dw
    dL_db = dL_da * da_dz * dz_db
    return dL_dw, dL_db
def numeric_gradient_wb(w,b,x,y,h= 1e-5):
    w_minus = w-h
    w_plus = w+h
    b_minus = b-h
    b_plus = b+h
    _, _, L_w_minus = forward(w_minus, b, x,y)
    _, _, L_w_plus = forward(w_plus, b, x,y)
    _, _, L_b_minus = forward(w,b_minus, x,y)
    _, _, L_b_plus = forward(w,b_plus, x,y)
    grad_w = (L_w_plus - L_w_minus)/(2*h)
    grad_b = (L_b_plus - L_b_minus)/(2*h)
    return grad_w, grad_b
x,y = 2.0, 1.0
w,b = 0.3, -0.1
z, a ,L = forward(w,b,x,y)
print("z,a,L =", z,a,L)
dL_dw, dL_db = grad_analytic(w, b, x, y)
print("analytic dL/dw, dL/db =", dL_dw, dL_db)
ndw, ndb = numeric_gradient_wb(w, b, x, y)
print("numeric  dL/dw, dL/db =", ndw, ndb)
assert abs(dL_dw - ndw) < 1e-6
assert abs(dL_db - ndb) < 1e-6
print("Task 3 OK")


