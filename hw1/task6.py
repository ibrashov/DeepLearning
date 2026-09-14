import numpy as np
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])
w = 0.0
b = 0
a = 0.1
def predict(x,w,b):
    return x * w + b
    
def mse(yhat, y):
    sum = 0
    for i in range(len(y)):
        sum +=(yhat[i]-y[i])**2
    return sum/len(yhat)

def gradient_dw(yhat, x,y):
    sum = 0
    for i in range(len(y)):
        sum +=(yhat[i]-y[i])*x[i]
    return 2*sum/len(yhat)

def gradient_db(yhat, x,y):
    sum = 0
    for i in range(len(y)):
        sum +=(yhat[i]-y[i])
    return 2*sum/len(yhat)
def num_grad(w,x,y,b,h=1e-5):
    w_plus = predict(x,w+h,b)
    w_minus = predict(x,w-h,b)
    b_plus = predict(x,w,b+h)
    b_minus = predict(x,w,b-h)
    grad_num_dw = (mse(w_plus,y)-mse(w_minus,y))/(2*h)
    grad_num_db = (mse(b_plus,y)-mse(b_minus,y))/(2*h)
    return grad_num_dw, grad_num_db

yhat = predict(x,w,b)
loss_before = mse(yhat, y)

dl_dw = gradient_dw(yhat,x,y)
dl_db = gradient_db(yhat,x,y)
grad_num_dw, grad_num_db = num_grad(w,x,y,b,h=1e-5)
w_new = w- a*dl_dw
b_new = b-a*dl_db

yhat_new = predict(x, w_new, b_new)
loss_after = mse(yhat_new, y)
print("loss before:", loss_before)
print("loss after:", loss_after)
print("analytical gradient dl_dw:", dl_dw)
print("analytical gradient dl_db:", dl_db)
print("numerical gradient dw:", grad_num_dw)
print("numerical gradient db:", grad_num_db)
print("compare after and before loss:", loss_before > loss_after)
