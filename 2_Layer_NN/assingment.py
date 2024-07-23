import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

#Calculating dimensions, shapes and input layer structure
def layer_sizes(X, Y):
    m=X.shape[1]
    n_x=X.shape[0]
    n_y=Y.shape[0]
    n_h=4
    return (n_x, n_y, m, n_h)

def initialize_parameters(n_h, n_x, n_y):
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1=np.dot(W1,X) + b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1) + b2
    A2=sigmoid(Z2)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

def compute_cost(parameters, m, A2, Y, regularization):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    if regularization == "no":
        J=-(np.dot(np.log(A2),Y.T)+np.dot(np.log(1-A2),(1-Y).T))/m
    elif regularization == "L2":
        lambd = 0.5
        J=-(np.dot(np.log(A2),Y.T)+np.dot(np.log(1-A2),(1-Y).T))/m+lambd/(2*m)*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    return (J)

def back_propagation(parameters, cache, X, Y, regularization):
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    dZ2 = A2 - Y
    if regularization == "no":
        dW2 = (np.dot(dZ2, A1.T))/m
        db2 = (np.sum(dZ2, axis=1, keepdims=True))/m
        g1_d = 1-np.power(A1, 2)
        dZ1 = np.dot(W2.T, dZ2) * g1_d
        dW1 = (np.dot(dZ1, X.T))/m
        db1 = (np.sum(dZ1, axis=1, keepdims=True))/m

    elif regularization == "L2":
        lambd = 0.5
        dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m)*W2
        db2 = (np.sum(dZ2, axis=1, keepdims=True))/m
        g1_d = 1-np.power(A1, 2)
        dZ1 = np.dot(W2.T, dZ2) * g1_d
        dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m)*W1
        db1 = (np.sum(dZ1, axis=1, keepdims=True))/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def optimize(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def model(X, Y, learning_rate, iterations, regularization):
    n_x, n_y, m, n_h = layer_sizes(X, Y)
    parameters = initialize_parameters(n_h, n_x, n_y)
    for i in range(0, iterations):
        A2, cache = forward_propagation(X, parameters)
        J = compute_cost(parameters, m, A2, Y, regularization)
        grads = back_propagation(parameters, cache, X, Y, regularization)
        parameters = optimize(parameters, grads, learning_rate)
        if i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, J))
    return parameters

def prediction(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions

parameters = model(X, Y, 1.2, 10000, "L2")

predictions = prediction(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

# Plot the decision boundary
plot_decision_boundary(lambda x: prediction(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()