import numpy as np
x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = np.array([0, 0, 0, 1])

w = np.array([1.0, 1.0])
b = -1.5



def predict(x,w,b):
    return x @ w +b
yhat = predict(x,w,b)

def perceptron_predict(x, w, b):
    z = yhat
    prediction = (z>0).astype(int)
    return z, prediction
z, prediction = perceptron_predict(x, w, b)

z_column = x @ w.reshape(2, 1) + b

print("column scores:", z_column.shape)
print("comparison shape:", ((z_column > 0) == y).shape)

print("scores:", z)
print("prediction:", prediction)
print("accuracy:", np.mean(prediction == y))

assert z.shape == (4,)
assert prediction.shape == y.shape
assert np.array_equal(prediction, y)

print("prediction: ", yhat)
print("shape of y:", y.shape)
print("shape of w: ",w.shape )
print("shape of x:", x.shape)