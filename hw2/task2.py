import numpy as np
x = np.array([0.5, 1.0, 1.5, 2.0])
y = np.array([2.0, 3.0, 4.0, 5.0])
params0 = ([0.5, 0.5, 1.0, 0.0])
def forward( params, x):
    w1,b1,w2,b2 = params
    z1 = w1 * x + b1
    h = np.maximum(0, z1)
    yhat = w2 * h +b2
    return z1, h, yhat
def batch_loss(params, x,y):
    z1,h,yhat= forward(params,x)
    assert yhat.shape == y.shape
    summ =0
    summ = np.sum((yhat - y)**2)
    L = (1/len(y))*summ
    return L
def batch_gradients(params, x,y):
    w1, b1,w2,b2 = params
    z1,h,yhat = forward(params, x)
    N = len(x)
    g_yhat = 2/N * (yhat - y)
    g_w2 = np.sum(g_yhat * h)
    g_b2 = np.sum(g_yhat * 1)
    g_h = g_yhat * w2
    g_z1 = g_h * (z1> 0).astype(float)
    g_w1 = np.sum(g_z1 * x)
    g_b1 = np.sum(g_z1 * 1)
    return np.array([g_w1, g_b1, g_w2, g_b2])
z1, h, yhat = forward(params0, x)
print("z1:", z1)
print("h:", h)
print("yhat:", yhat)
print("loss:", batch_loss(params0, x, y))
print("gradients:", batch_gradients(params0, x, y))