import numpy as np
x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
w = np.zeros(2, dtype=float)
b = 0.0
eta = 0.5
examples = [
    (np.array([1.0, 1.0]), 1),
    (np.array([0.0, 0.0]), 0),
]

for x_i, target in examples:
    score = w @ x_i + b
    pred = (score > 0).astype(float)
    error = target - pred

    w_new = w + eta * error*x_i
    b_new = b + eta * error

    print("score, prediction, error:", score, pred, error)
    print("new parameters:", w_new, b_new)

    w, b = w_new, b_new

def perceptron_predict(x, w, b):
    z = np.dot(x,w )+b
    prediction = (z>0).astype(int)
    return z, prediction
z, prediction = perceptron_predict(x, w, b)
print("all AND predictions:", perceptron_predict(x, w, b)[1])