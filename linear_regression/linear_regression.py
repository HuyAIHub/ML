#là 1 thuật toán học có giám sát , trong đó đầu ra dự đoán là liên tục và có độ dốc không đổi
# học có giám sát giống việc học có thầy vậy (cái đó có thể nhìn thông qua bộ dataset thường thì bộ datasets se dk gán nhãn )
# note : học có giám sát là dataset sẽ dk gán nhãn
# trong đó đầu ra dự đoán là liên tục (vd : doanh số , giá cả )(ít khi dùng trong những bài toán phân loại )

# va nó có 2 loại chính :
# 1. simple regression (đơn giản như 1 pt đt thôi)
# 2. Multiveriable regression (nhiều tham số phức tạp hơn )

#dạng simple , đề bài : y = wx + b (w : weight , b : bias dùng để bù đắp lại những sai số)
# biến độc lập là biến k phụ thuộc vào cái j hết (x)
# biến phụ thuộc là y (phụ thuộc x , b , w)



# cost function :
# chúng ta cần 1 hàm chi phí để tối ưu weight
# hàm lỗi MSE (measures squared error ) : là hàm trung bình bình phương của tất cả các cái lỗi
# đo bằng các tính trung bình bình phương của giá trị sai khác và giá trị thực tế

import numpy as np
