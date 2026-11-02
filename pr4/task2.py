import numpy as np
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)
w = 0
def forward(x,y,w):
    yhat = w * x
    return yhat
def mse(yhat, y):
    yhat,y = forward(yhat, y)
    L = np.mean((yhat - y)**2)
    return L
def grad(x, y, yhat, w):
    B = len(x)
    dl_dw = (1/B) * (np.sum(2*(w*x-y)*x))
    return dl_dw
yhat = forward(x,y,w)
print("Mse:", mse(yhat, y))
print("Grad:", grad(x,y,yhat,w))
print(f"Grad last 2: {grad:-2}")

