import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y_and = np.array([0,0,0,1])
y_xor = np.array([0,1,1,0])
def predict(X,w,b):
    z = X @ w +b
    prediction = (z>0).astype(int)
    return prediction
def train_perceptron(X,y, eta=0.5, max_epochs = 20):
    w = np.zeros(2)
    b = 0.0 
    mistake_per_epoch = []
    for epoch in range(max_epochs):
        mistakes = 0
        for i in range(len(X)):
            z = w @ X[i]+b
            y_pred = 1 if z>0 else 0
            err = y[i] - y_pred
            w += eta*err*X[i]
            b += eta*err
            mistakes +=int( err != 0)
        mistake_per_epoch.append(mistakes)
        if (mistakes == 0):
            break 
    return w,b,mistake_per_epoch
w_and, b_and, hist_and = train_perceptron(X,y_and)
w_xor, b_xor, hist_xor = train_perceptron(X,y_xor)
print("AND: epochs =", len(hist_and), "w =", w_and, "b =", b_and)
print("AND: mistakes per epoch =", hist_and)
print("AND: final predictions  =", predict(X, w_and, b_and))
print("XOR: epochs =", len(hist_xor), "w =", w_xor, "b =", b_xor)
print("XOR: mistakes per epoch =", hist_xor)
print("XOR: final predictions  =", predict(X, w_xor, b_xor))
and_err = predict(X,w_and, b_and)
and_final = np.sum(and_err != y_and)
print("AND final predict:", and_final)
xor_err = predict(X,w_xor, b_xor)
xor_final = np.sum(xor_err != y_xor)
print("XOR final predict:", xor_final)
plt.plot(range(1,len(hist_and)+1), hist_and, "o", label= 'AND')
plt.plot(range(1,len(hist_xor)+1), hist_xor, "-m", label= 'XOR')
plt.xlabel("Epoch")
plt.ylabel("Mistake")
plt.legend()
plt.title("Mistakes per epoch")
plt.tight_layout()
plt.show()
assert np.all(predict(X, w_and, b_and) == y_and)
print("Task 1 OK")