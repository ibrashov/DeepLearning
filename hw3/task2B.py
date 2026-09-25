import numpy as np
target = np.array([1,0])
predict_a = np.array([0.8 , 0.2])
predict_b = np.array([0.6 , 0.4])
def bce_a (target, predict_a):
    L_a = -(target * np.log(predict_a) + (1-target) * np.log(1-predict_a))
    return L_a
def bce_b(target, predict_b):
    L_b = -(target * np.log(predict_b) + (1- target) *np.log(1-predict_b))
    return L_b
predict_aa = predict_a.copy()
predict_bb = predict_b.copy()
h_a = (predict_a > 0.5).astype(float)
h_b = (predict_b > 0.5).astype(float)
L_a = bce_a(target, predict_a)
L_b = bce_b(target, predict_b)
print("BCE prediction A:", L_a)
print("BCE prediction B:", L_b)
print("Mean BCE prediciton A:", np.mean(L_a))
print("Mean BCE prediction B:", np.mean(L_b))
print(h_a)
print(h_b)
