import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Caclulating dimensions and shapes
m_train=train_set_x_orig.shape[0] #number of training images
m_test=test_set_x_orig.shape[0] #number of test images
num_px_x=train_set_x_orig.shape[1] #number of pixels (same on x and y direction)
num_px_y=train_set_x_orig.shape[2]
channel=train_set_x_orig.shape[3]

#Flattening the train and test
train_set_x_flatten=train_set_x_orig.reshape(num_px_x*num_px_y*channel,m_train)
test_set_x_flatten=test_set_x_orig.reshape(num_px_x*num_px_y*channel,m_test)

#Data standardize
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

#Helper function
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

#Initialize w & b, dim is num_px*num_px*3
def initialize_with_zeros(dim):
    w=np.zeros((dim, 1))
    b=0
    return w, b

def propagate(w, b, X, Y):
    #Forward propogation (X to cost)
    A=sigmoid(np.dot(w.T,X)+b)
    m=X.shape[1]
    J=-(np.dot(np.log(A),Y.T)+np.dot(np.log(1-A),(1-Y).T))/m

    #Back propogation (to find grad)
    dw=(np.dot(X,(A-Y).T))/m
    db=(np.sum(A-Y))/m
    return dw, db, J

def optimize(w, b, X, Y, learning_rate, num_iterations):
    costs=[]
    for i in range(num_iterations):
        dw, db, cost = propagate(w, b, X, Y)
        w=w-learning_rate*dw
        b=b-learning_rate*db

        if i%100==0:
            costs.append(cost)
            print("Cost after iteration %i: %f" %(i, cost))
    return w, b, dw, db, costs

def predict(w, b, X):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    Y_prediction = []
    for i in range(m):
        if A[0, i] >=0.5:
            Y_prediction.append(1)
        else:
            Y_prediction.append(0)
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    #Initialization and training
    w, b = initialize_with_zeros(X_train.shape[0])
    w, b, dw, db, costs = optimize(w, b, X_train, Y_train, learning_rate, num_iterations)

    #Testing
    Y_prediction_test=predict(w, b, X_test)
    Y_prediction_train=predict(w, b, X_train)

    #Error calculation
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    return None

model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005)