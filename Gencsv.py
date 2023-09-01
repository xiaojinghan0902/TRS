import sys
import pandas as pd

dataP = pd.read_csv("RM数据.csv",usecols=["prompt"],encoding="utf8")
dfp = pd.DataFrame(dataP)

dfp.to_csv('Chuizhi.txt', index=False)


