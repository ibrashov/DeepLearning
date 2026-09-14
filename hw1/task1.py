import numpy as np
import matplotlib.pyplot as plt

def f(x):
        return np.sin(x)
def df_true(x):
        return np.cos(x)
def derive_ford(f,x,h):
        return ((f(x+h)-f(x))/h)
def derive_cent(f,x,h):
        return ((f(x+h)-f(x-h))/(2*h))
x0=1.0
print("true:", df_true(x0) )
print("forward h = 1e-5: ", derive_ford(f,x0, 1e-5))
print("central h = 1e-5: ", derive_cent(f, x0, 1e-5) )
hs = np.logspace(-1, -12, 45)
err_ford = np.array([abs(derive_ford(f,x0,h)- df_true(x0))
    for h in hs
])
err_cent = np.array([abs(derive_cent(f,x0,h)-df_true(x0))
    for h in hs
])
plt.figure(figsize =(6,4))
plt.loglog(hs, err_ford, "o-", label = "forward")
plt.loglog(hs, err_cent, "s-", label = "central")
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("absolute error")
plt.legend()
plt.title("Finite-Difference Error vs Step Size")
plt.tight_layout()
plt.show()
print("best, h, forward: ", hs[np.argmin(err_ford)])
print("best, h, center: ", hs[np.argmin(err_cent)])
assert abs(derive_cent(f,x0, 1e-5)- df_true(x0)) < 1e-7
print("Task 1 OK")
