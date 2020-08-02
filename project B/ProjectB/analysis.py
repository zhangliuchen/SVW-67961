# Project B：产品关联分析
# PLPA-1 张刘晨 67961
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori

# 读入原始数据
data = pd.read_csv("订单表.csv", encoding="gbk")
# 清洗数据，获得每个客户的订单
products = data.groupby(data["客户ID"])["产品名称"].value_counts().unstack()
products[products > 1] = 1
products[np.isnan(products)] = 0
print(products.shape)

# 将数据存放到transactions中
transactions = []
for i in range(0, products.shape[0]):
    temp = []
    for j in range(0, 17):
        if str(products.values[i, j]) != 'nan':
           temp.append(str(products.values[i, j]))
    transactions.append(temp)
#print(transactions)
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.4)
print("频繁项集：", itemsets)
print("关联规则：", rules)
