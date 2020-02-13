import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("./data.csv")

pd.set_option('display.max_columns', None)
print(data.columns)
print(data.head(5))
print(data.describe())

features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst =list(data.columns[22:32])

data.drop("id",axis=1,inplace=True) # 数据清洗

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

sns.countplot(data['diagnosis'],label="Count")# 热力图显示
plt.show()

corr = data[features_mean].corr() # 表示features_mean字段之间的相关性
plt.figure(figsize=(14,14))

sns.heatmap(corr, annot=True) # annot=True显示每个方格的数据
plt.show()

# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 

train, test = train_test_split(data, test_size = 0.2)

train_X = train[features_remain]
train_y=train['diagnosis']
test_X= test[features_remain]
test_y =test['diagnosis']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
transformer = StandardScaler()
train_X = transformer.fit_transform(train_X)
test_X = transformer.transform(test_X)

mode = 1  #different mode you can change

if mode == 1:
    model = svm.SVC()
else:
    model = svm.LinearSVC()

model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction,test_y))

