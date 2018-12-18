
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')

print(data)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)

y_onehot = encoder.fit_transform(data['y'])
for a in y_onehot:
    print(a)
X = np.matrix(data['X'])
# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
print(params.shape)
print(theta1.shape[0],theta1.shape[1])
print(theta2.shape[0],theta2.shape[1])
print(X.shape[0],X.shape[1])


# In[2]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    #Write codes here
    a1 = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1 ) # the first column is all one, used for bias
    z2 = a1.dot(theta1.T)
    a2 = np.concatenate((np.ones((z2.shape[0],1)),sigmoid(z2)), axis = 1 ) # the first column is all one, used for bias
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h
    
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:]))))
    
    return J
    


# In[3]:


# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    J = 0
    
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    
    grad2 = np.zeros(theta1.shape)
    grad3 = np.zeros(theta2.shape)
    ''' # No Vectorized
    for i in range(m):
        d3 = -(y[i,:] - h[i,:]) # 1 x 10
        d2 = np.multiply(d3.dot(theta2[:,1:]) , sigmoid_gradient(z2[i,:]) )
        
        grad3 = grad3 + d3.T * a2[i,:] # 10 x 1 dot 1 x 401 = 10 x 401 
        grad2 = grad2 + d2.T * a1[i,:] # 10 x 1 dor 1 x 11 = 10 x 11
    '''
    # Vectorized method
    d3 = -(y - h) # 1 x 10
    d2 = np.multiply(d3.dot(theta2[:,1:]) , sigmoid_gradient(z2) )
    grad3 = (d3.T * a2)/m # 10 x m dot m x 401 = 10 x 401 
    grad2 = (d2.T * a1)/m # 10 x m dor m x 11 = 10 x 11


    # add the gradient regularization term
    grad2[:,1:] = grad2[:,1:] + (theta1[:,1:] * learning_rate) / m
    grad3[:,1:] = grad3[:,1:] + (theta2[:,1:] * learning_rate) / m
    
    grad = np.concatenate((np.ravel(grad2), np.ravel(grad3)))
    return J,grad
    
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250})
      
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))

