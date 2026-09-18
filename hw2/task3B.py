import numpy as np
import matplotlib.pyplot as plt
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
def num_grad(fn, params, eps =1e-5):
    gradient = np.zeros_like(params, dtype = float)
    for j in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[j] += eps
        minus[j] -= eps
        gradient[j] = (fn(plus) - fn(minus))/(2*eps)
    return gradient
z1, h, yhat = forward(params0, x)
old_params = params0.copy()
eta = 0.01

old_loss = batch_loss(old_params, x,y)
gradients = batch_gradients(old_params, x, y)
new_params = old_params - eta*gradients
new_loss = batch_loss(new_params,x,y)
print("old parameters:", old_params)
print("gradients:", gradients)
print("new parameters:", new_params)
print("old loss:", old_loss)
print("new loss:", new_loss)

params = params0.copy()
loss_history = [batch_loss(params,x, y)]
for step in range(200):
    params = params - eta * gradients
    gradients = batch_gradients(params, x, y)
    current_loss = batch_loss(params,x,y)
    loss_history.append(batch_loss(params, x,y))
z1, h, final_prediction = forward(params, x)


print("final params:", params)
print("final prediction:", final_prediction)
print("final_loss:", loss_history[-1])
assert len(loss_history) == 201
assert loss_history[-1] < loss_history[0]

print("Task 3B OK")
plt.plot(range(201), loss_history)
plt.xlabel("Updated number")
plt.ylabel("Loss")
plt.title("Loss during gradient descent")
plt.show()