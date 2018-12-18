import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Concrete_Data.csv")

"""
MSE = sum of [(y - x*w)**2] / length
"""
def MSE(X,coef,intercept,Y):
    return np.sum((Y-(X*coef+intercept))**2)/X.shape[0]

"""
error: MSE
shape: the number of data
R2 = 1 - SSE/SSTO
   = 1 - MSE*shape/ sum of [(y-y.mean)**2]
"""
def R2(error,shape,y):
    SSTO = np.sum((y - np.mean(y))**2)
    return 1 - error*shape/SSTO

y = df.iloc[:,8].values
column_name = list(df.columns.values)

for i in range(8):
    x = df.iloc[:,i].values 
    plt.scatter(x, y,marker='.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(column_name[i])
    plt.savefig(column_name[i])
    plt.show()
    
# use your eye to select x0 
x = df.iloc[:,0].values
y = df.iloc[:,8].values # get target
x = (x - np.amin(x, axis=0))/(np.amax(x, axis=0)-np.amin(x, axis=0)) 
y = (y - np.amin(y, axis=0))/(np.amax(y, axis=0)-np.amin(y, axis=0))

x_train, x_test, y_train, y_test = train_test_split( x , y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train[:,np.newaxis],y_train)
error = MSE(x_train,model.coef_,model.intercept_,y_train)
R_squared = R2(error, x_train.shape[0], y_train)
print("Training\nerror:%.5f , R2: %.5f" % (error,R_squared))
error = MSE(x_test,model.coef_,model.intercept_,y_test)
R_squared = R2(error, x_test.shape[0], y_test)
print("Testing\nerror:%.5f , R2: %.5f" % (error,R_squared))

plt.plot(x_train,y_train,'.',markersize=15,alpha=0.3)
plt.plot(x_train,model.intercept_+model.coef_*x_train,linewidth=5)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20,rotation=0)
plt.savefig('scikit_learn')

