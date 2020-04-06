import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv",sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]] # cut ra nhung column co ten nhu nay
predict = "G3"
X = np.array(data.drop([predict],1))
Y = np.array(data[predict])
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)

print(acc)

print("coefficient : \n",linear.coef_)
print("intercept : \n", linear.intercept_)
print("ket qua predict :")
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

