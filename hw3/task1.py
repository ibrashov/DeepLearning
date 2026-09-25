import numpy as np
x = np.array([[1. , 0.], [0., 1.], [1. , 1.]])
y = np.array([[1.], [2.], [1.]])
w1 = np.array([[1. , -1.], [-1. , 1.]])
b1 = np.array([0.5 , 0.5])
w2 = np.array([[1.], [2.]])
b2 = np.array([0.])
def forward(x,w1,b1,w2,b2):
    z1 = x@w1 + b1
    h = np.maximum(0, z1)
    yhat = h@ w2 + b2
    return z1, h , yhat
def mse(y, yhat):
    return np.square(np.subtract(y, yhat)).mean()
def forward_bad(x,w1,b1,w2,b2):
    z1 = x@w1 + b1
    yhat_bad = z1@ w2 + b2
    return z1, yhat_bad
z1,h, yhat = forward(x,w1,b1,w2,b2)
yhat_bad = forward_bad(x,w1,b1,w2,b2)
loss = mse(y,yhat)
print("x shape:", x.shape)
print("y shape:", y.shape)
print("w1 shape:", w1.shape)
print("b1 shape:", b1.shape)
print("w2 shape:", w2.shape)
print("b2 shape:", b2.shape)
print("z1 shape:", z1.shape)
print("h shape:", h.shape)
print("yhat shape:", yhat.shape)
print("Forward z1:", z1 )
print("Forward yhat:", yhat)
print("Forward h:", h)
print("Forward_bad yhat:", yhat_bad)
print("Loss:", loss)
assert y.shape == yhat.shape

