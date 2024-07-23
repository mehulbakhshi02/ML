import numpy as np

def sigmoid(x):
    sig=1/(1+np.exp(-x))
    s=sig*(1-sig)
    return s

x = np.array([1, 2, 3])

def image2vector(image):
    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v

def normalizeRows(x):
    norm=np.linalg.norm(x, axis=1, keepdims=True)
    x_norm=x/norm
    return x_norm

def softmax(x):
    x_exp=np.exp(x)
    sum = np.sum(x_exp, axis=1, keepdims=True)
    s=x_exp/sum
    return s

def L1(yhat, y):
    loss=np.sum(abs(y-yhat))
    return loss

def L2(yhat, y):
    loss=np.dot(abs(y-yhat), abs(y-yhat))
    return loss