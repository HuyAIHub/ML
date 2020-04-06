import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer                      #1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #2
from sklearn.compose import ColumnTransformer                 #3
from sklearn.model_selection import train_test_split          #4
from sklearn.preprocessing import StandardScaler              #5
#import dataset
dataset = pd.read_csv('/home/quanghuy/Downloads/Data.csv')
X = dataset.iloc[ : , :-1].values  #data.iloc[<row selection>, <column selection>]
Y = dataset.iloc[ : , 3].values
#handling missing data (1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
#encoding categorical data (ma hoa data phan loai )(2)(3)
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#create a dummy variable (tao bien hinh nom)(2)(3)
onehotencoder = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
#splitting(tach) the datasets into traning sets and test sets
# cai nay dung de chia dataset chung ta ra 75% dung de training , 25% con lai de test

X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0) # lido viet tren 1 dong la do cai train_test_split tra ve dang danh sach tra ve 4 tham so
                                                                            #random_state = 0 thi no se lay ngau nhien trong bo datasets ra 75% de training(moi lan lay no ra giong nhau)
                                                                            #             = 1 moi lan chay lai no ra bo training khac
#feature scaling (tinh nang mo rong)(5)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)