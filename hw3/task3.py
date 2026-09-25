import numpy as np
import matplotlib.pyplot as plt
x = np.array([[-2.], [-1.], [-0.5], [0.5], [1.], [2.]])
y = np.abs(x)
initial = {
    "W1": np.array([[0.5, -0.5]]),
    "b1": np.array([0. , 0.]),
    "W2": np.array([[0.5], [0.5]]),
    "b2": np.array([0.]),
}
def forward( params, x):
    z1 = x @ params["W1"] + params["b1"]
    h = np.maximum(0, z1)
    yhat = h @ params["W2"] + params["b2"]
    return z1, h , yhat
def mse(params, x, y):
    z1, h ,yhat = forward(params, x)
    L = np.mean((yhat - y)**2)
    return L
z1, h, yhat = forward( initial, x)
print("Initial prediction:", yhat)
print("Initial MSE:", mse(initial, x, y))
def grad(params, x, y):
    z1, h ,yhat = forward(params, x)
    n = x.shape[0]
    G = (2/n)*(yhat - y)
    dW2 = h.T @ G
    db2 = np.sum(G, axis = 0)
    D = (G @ params["W2"].T) * (z1 > 0)
    dW1 = x.T @ D
    db1 = np.sum(D, axis = 0)
    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }
    return grads
params = {name: value.copy() for name, value in initial.items()}
gradients = grad(params, x ,y)
for name in gradients:
    print("Gradients:", gradients[name])
    assert params[name].shape == gradients[name].shape
print("Params old:", params)
mse_old = mse(params, x,y)
params_new = {name: params[name] - 0.05 * gradients[name] for name in params}
mse_new = mse(params_new, x, y)
print("Params new:", params_new)
print("Old MSE:", mse_old)
print("New MSE:", mse_new)
params = {name: value.copy() for name, value in initial.items()}
history = [mse(params, x,y)]
for step in range(300):
    grades = grad(params, x, y)
    params = {name: params[name] - 0.05 * grades[name] for name in params}
    history.append(mse(params, x, y))
print("First param:", history[0])
print("Last param:",history[-1])
z1, h, yhat_final = forward( params, x)
mse_final = mse(params, x, y)
print("Final prediction:", yhat_final)
print("Final params:", params)
plt.figure()
plt.plot(range(301), history)
plt.xlabel("Update number")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid()
plt.show()

x_check = np.array([[-1.5],[-0.75], [0.75],[1.5]])
y_check = np.abs(x_check)
z1_check, h_check, predictions_check = forward( params,x_check)
check_mse = mse(params,x_check,y_check)
print("X_check:", x_check)
print("Targets:", y_check)
print("Predictions:", predictions_check)
print("Check MSE:",check_mse)
x_plot = np.linspace(-2,2,101).reshape(-1, 1)
y_plot = np.abs(x_plot)
z_plot, h_plot, predictions_plot = forward(params,x_plot)
plt.figure()
plt.plot( x_plot[:, 0], y_plot[:, 0], label="Target" )
plt.plot(x_plot[:, 0], predictions_plot[:, 0], label="MLP prediction" )
plt.scatter( x[:, 0], y[:, 0], label="Training points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Target function vs MLP")
plt.legend()
plt.grid()
plt.show()