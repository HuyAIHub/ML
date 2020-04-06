#step 1: thu thập dữ liệu
#step 2: xử lí dữ liệu
#step 3: xây dựng model
#step 4: dự đoán kết quả
#step 5: danh gia model co hieu qua k

from sklearn import tree
# khoi tao cay quyet dinh
my_tree = tree.DecisionTreeClassifier()
#step 1 encoding categorical

dactrung = [[1,3,3,7],
            [5,2,4,6],
            [1,2,4,6],
            [5,4,4,3],
            [1,4,4,7],
            [3,2,3,7],
            [3,3,3,6],
            [5,2,2,7]]
kqtuonguong = [0,1,1,0,0,0,0,1]

#training
result = my_tree.fit(dactrung,kqtuonguong)
#predict
kq = result.predict([[1,4,3,6]])
print(kq)
