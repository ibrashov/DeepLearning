import numpy as np
target = np.array([1,2,4])
prediction = np.array([0,2,10])
error = prediction - target
def mse(target, prediction):
    return np.square(np.subtract(target, prediction)).mean()
def mae(target, prediction):
    return np.absolute(np.subtract(target, prediction)).mean()
loss_mse = mse(target, prediction)
loss_mae = mae(target, prediction)
print("error:", error)
print("MSE:", loss_mse)
print("MAE:", loss_mae)