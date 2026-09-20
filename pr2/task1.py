import numpy as np
y = np.array([0.0, 0.0, 0.0, 1.0])
y_pred = np.array([0.1, 0.2, 0.6, 0.1])
def cef(y, y_pred):
    L = -(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
    return L
error = cef(y, y_pred)
sortt = np.sort(error)
meann = np.mean(error)
print("Error", error)
print("Sorted", sortt)
print("Meann:", meann)