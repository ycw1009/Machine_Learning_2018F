import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Concrete_Data.csv")

x = df.iloc[:,0].values # get x0 variable
y = df.iloc[:,8].values # get target
ones = np.ones((x.shape[0],1)) # constant
# normalization
# x = (x - min) / (max - min)
x = (x - np.amin(x, axis=0))/(np.amax(x, axis=0)-np.amin(x, axis=0)) 
y = (y - np.amin(y, axis=0))/(np.amax(y, axis=0)-np.amin(y, axis=0))
# change dimension : 1D -> 2D
x = x[:,np.newaxis]
y = y[:,np.newaxis]
x_process = np.concatenate((ones,x), axis = 1 )
# seperate training and testing set
x_train, x_test, y_train, y_test = train_test_split( x_process, y, test_size = 0.2)

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

lr = 0.5 # learning rate 
iterations = 1000 # max loop size
iteration = 0 # loop number now
epsilon = 0.005 # minimum of error
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
    print("Training %d iteration, error:%.5f , R2: %.5f" % (iteration,error,R_squared))
    if error < epsilon: # the error is small enough, break
        break

# Testing
error = MSE( x_test, w, y_test)
R_squared = R2(error, x_test.shape[0], y_test)
print("Testing\nerror:%.5f , R2: %.5f" % (error,R_squared))
print("weight: ",w[0][0],",\nbias: ",w[1][0])

plt.plot(x_train[:,1].flatten(),y_train,'.',markersize=15,alpha=0.3)
plt.plot(x_train[:,1].flatten(),x_train[:,1].flatten()*w[0][0]+w[1][0],linewidth=5)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20,rotation=0)
plt.title('Training')
plt.savefig('GD')

