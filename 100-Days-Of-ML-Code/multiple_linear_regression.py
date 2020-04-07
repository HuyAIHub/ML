import numpy as np
import pandas as pd

ufo = pd.read_csv("50_Startups.csv")
X = ufo.iloc[ :, :-1].values # luon luon nho la 1 hang cua X se tuong ung vs 1 kq ben y => len(x)==len(y)
Y = ufo.iloc[ :, 4  ].values
print(ufo.head())
print(len(X))
print("----------")
print(len(Y))
# encoding categorical data(ma hoa va phan loai data)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
leberencoder = LabelEncoder()
X[:,3] = leberencoder.fit_transform(X[:,3])
onehotencoder = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
#X = X[:,1:]
#Splitting the dataset into the Training set and Test set(tach about 80% training , 20% test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#fitting Muntiple Linear Regression to train test
from sklearn import linear_model
linear = linear_model.LinearRegression()
result = linear.fit(x_train,y_train)
# accuracy
acc = linear.score(x_test,y_test)
# predict result
result_predict = linear.predict(x_test)
for x in range(len(result_predict)):
    print(result_predict[x],":",x_test[x],":",y_test)



