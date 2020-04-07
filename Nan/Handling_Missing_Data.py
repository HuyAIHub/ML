# xu li data bi thieu (Nan) not a number
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

ufo = pd.read_csv("test_missing_data.csv", header=None) # đây là dữ liệu ở dạng dataframe
print(ufo)
print("----------------------")
X = ufo.values # convert dataframe to array chi can ( .values ) la dk
data = Imputer(missing_values=np.nan,strategy='mean') # muốn tìm hiểu kỹ cần đưa vào cái j ấn ctrl click chuột vô
            # missing_values=np.nan :de ns cai missing value la cai j : va mk bao no la no la gia tri (not a number trong np )
            #strategy='mean' : tuc la lay trung binh cua cot chua no dien vao cho NaN do
            # muốn điền vào chỗ Nan tức chỗ dữ liệu thiếu hụt đó bằng dữ liệu có tần suất nhiều nhất thì thay đổi "mean"="most_frequent"

data.fit(X) # cai h àmnày cần dữ liệu truyền vào là 1 mảng (ta phai quay lên convert lại ufp )
            # cai ham fit nay chung ms chi cho data vao thoi de chuyen doi ta can
result = data.transform(X)
print(result)