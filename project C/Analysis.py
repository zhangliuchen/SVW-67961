# Project C：汽车产品聚类分析
# PLPA-1 张刘晨 67961
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
data = pd.read_csv('CarPrice_Assignment.csv')
# 创建训练数据集,去除无用数据
train_x = data.drop(['car_ID','CarName'],axis=1)
# 使用LabelEncoder将文本类型特征转化为数字
from sklearn.preprocessing import LabelEncoder
cols = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
le = LabelEncoder()
for col in cols:
    train_x[col] = le.fit_transform(train_x[col])
# 将数据规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x.astype(float))
pd.DataFrame(train_x).to_csv('temp.csv', index=False)
import matplotlib.pyplot as plt
sse = []
for k in range(1, 50):
	# kmeans算法
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(train_x)
	# 计算inertia簇内误差平方和
	sse.append(kmeans.inertia_)
x = range(1, 50)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()

#由手肘法，取K=7~10，使用KMeans聚类
kmeans = KMeans(n_clusters=7)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
print(result)
result.to_csv('result.csv',index=False)