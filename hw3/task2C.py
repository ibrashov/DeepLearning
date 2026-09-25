import numpy as np
logist = np.array([0, np.log(3), 0])
corect = 1
def multiclass(logist):
    summ = 0
    for i in range(len(logist)):
        summ += np.exp(logist[i])
    p = (np.exp(logist))/summ
    l = -(np.log(p[corect]))
    return p ,l 
p, l = multiclass(logist )
print("Probability:", p)
print("Loss:", l)
print("Sum:", np.sum(p))
