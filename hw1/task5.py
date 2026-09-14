import numpy as np
x = np.array([
    [1.0, 2.0],
    [2.0, 0.5],
    [3.0, 1.0],
    [4.0, 3.0],
])
y = np.array([1.0, 3.0, 5.0, -2.0])

w = np.array([2.0, -1.0])
b = 0.5
def predict(x,w,b):
    return x @ w + b
yaht = predict(x,w,b)
def mse_loop(yaht, y):
    sum = 0
    for i in range(len(y)):
        sum +=(yaht[i]-y[i])**2
    return sum/len(yaht)

def mse(yaht, y):
    return np.square(np.subtract(y,yaht)).mean()
def mse_mean_bad(yaht_bad, y):
    return np.square(np.subtract(y,yaht_bad)).mean()

print(" yaht shape:", yaht.shape)
print("yaht:", yaht)
w_bad = w.reshape(2,1)
yaht_bad = predict(x,w_bad,b)
print("yaht_bad: ", yaht_bad.shape)
print("yaht_bad:    ", yaht_bad)
mse_for = mse_loop(yaht, y)
mse_mean = mse(yaht, y)
print("mse_loop= ", mse_for)
print("mse_mean= ", mse_mean)
mse_bad = mse_mean_bad(yaht_bad, y)
print("mse_mean_bad= ", mse_bad)
assert yaht.shape == (4,)
assert yaht_bad.shape == (4,1)
print("Task 4 OK")