
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("./Concrete_Data.csv")


# In[3]:


x = df.iloc[:,0:8].values # get all variable
y = df.iloc[:,8].values # get target
ones = np.ones((x.shape[0],1)) # constant
# normalization
# x = (x - min) / (max - min)
x = (x - np.amin(x, axis=0))/(np.amax(x, axis=0)-np.amin(x, axis=0)) 
y = (y - np.amin(y, axis=0))/(np.amax(y, axis=0)-np.amin(y, axis=0))
# change y's dimension : 1D -> 2D
y = y[:,np.newaxis]
# get the square of variable
x_square = x**2
# get the multiplication of variable 
x_multi = [np.multiply(x[:,i],x[:,j]) for i in range(0,x.shape[1]-1) for j in range(i+1,x.shape[1])]
# concatenate all 
x_process = np.concatenate((ones,x,x_square,np.transpose(x_multi)), axis = 1 )
# seperate training and testing set
x_train, x_test, y_train, y_test = train_test_split( x_process, y, test_size = 0.2)


# In[4]:


"""
MSE = sum of [(y - x*w)**2] / length
"""
def MSE(X,W,Y):
    return np.sum(np.dot((np.dot(X,W)-Y).T,
                            (np.dot(X,W)-Y)) /(2*X.shape[0]))

"""
error: MSE
shape: the number of data
R2 = 1 - SSE/SSTO
   = 1 - MSE*shape/ sum of [(y-y.mean)**2]
"""
def R2(error,shape,y):
    SSTO = np.sum((y - np.mean(y))**2)
    return 1 - error*shape/SSTO


# In[5]:


lr = 5 # learning rate 
iterations = 1000 # max loop size
iteration = 0 # loop number now
epison = 0.005 # minimum of error
w = np.random.randn(x_train.shape[1],1) # initial weight

# Training
while iteration < iterations:
    iteration += 1
    for var in range(x_train.shape[1]): # do GD to each variable
        gradient = np.dot((y_train - np.dot( x_train, w )).T,
               x_train[:,var])/x_train.shape[0]
        w[var,0] = w[var,0] + lr*gradient
    error = MSE( x_train, w, y_train)
    R_squared = R2(error, x_train.shape[0], y_train)
    print("Traing %d iteration, error:%.5f , R2: %.5f" % (iteration,error,R_squared))
    if error < epison: # the error is small enough, break
        break

# Testing
error = MSE( x_test, w, y_test)
R_squared = R2(error, x_test.shape[0], y_test)
print("Testing\nerror:%.5f , R2: %.5f" % (error,R_squared))

