import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#data preprocessing (so che data)
ufo = pd.read_csv("studentscores.csv")
X = ufo.iloc[ : , : 1 ].values
Y = ufo.iloc[ : , 1 ].values
print(len(X))
print("----------")
print(len(Y))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
#Fitting Simple Linear Regression Model to the training set
from sklearn import linear_model
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
#accuracy
acc = linear.score(x_test,y_test)
print(acc)
# predict result
print(x_test)
result = linear.predict(x_test)
print("---------------")
for x in range(len(result)):
    print(result[x],x_test[x],y_test[x])
# visualization
#visualising the result train
plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,linear.predict(x_train) ,color = "blue")
plt.show()
#visualising the result test
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,result,color = "blue")
plt.show()
print("haha")