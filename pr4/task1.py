# Your code here
import numpy as np
def grad(w):
    df = 2 * (w-3)
    return df
a = np.array([0.1, 0.5, 0.9, 1.1])
for i in a:
    w = 0
    for step in range(1,11):
        df = grad(w)
        w = w - i*df
        print(f"Step{step} w: {w:.3f}")