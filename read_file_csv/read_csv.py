import numpy as np
import pandas as pd

ufo = pd.read_csv("UNSW_NB15_testing-set.csv" , header=None) # header la cai dong dau ma co ten cac thu ys
                                                            # neu header = i no se lay dong so i lm header
print(ufo[0])
# muon luu 1 cot hay nhieu cot hay j do sang 1 file khac thi ta lm nhu sau
ufo = ufo[0]
ufo.to_csv("colunm1.csv")
