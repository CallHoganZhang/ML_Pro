import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import data_load

def change_date_type(features):
    years = features['year']
    months = features['month']
    days = features['day']

    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    return dates

dates = change_date_type(data_load.features)
# print(dates)

def encoding_features(features):
    onehot_coding_features = pd.get_dummies(features) #one hot coding
    onehot_coding_labels = onehot_coding_features['actual']
    dropped_features = onehot_coding_features.drop('actual', axis = 1)
    dropped_features_list = list(dropped_features.columns)

    array_features = np.array(dropped_features)
    array_labels = np.array(onehot_coding_labels)
    return array_features, array_labels, dropped_features_list

array_features, array_labels, dropped_features_list = encoding_features(data_load.features)
train_features, test_features, train_labels, test_labels = train_test_split(array_features, array_labels,test_size = 0.1, random_state = 0)
#
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

def encoding_origin_features(features, feature_list):
    original_features = pd.get_dummies(features)
    original_feature_indices = [feature_list.index(feature) for feature in
                                          feature_list if feature not in
                                          ['ws_1', 'prcp_1', 'snwd_1']]

    array_original_labels = np.array(original_features['actual'])
    original_features= original_features.drop('actual', axis = 1)

    original_feature_list = list(features.columns)
    array_original_features = np.array(original_features)
    return array_original_features, array_original_labels, original_feature_list

array_original_features, array_original_labels, original_feature_list = encoding_origin_features(data_load.original_features, dropped_features_list)
original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(array_original_features, array_original_labels, test_size = 0.1, random_state = 42)
#
print('original_train_features Shape:', original_train_features.shape)
print('original_train_features:', original_train_features.shape)
print('original_train_features:', original_train_features.shape)
print('original_train_features:', original_train_features.shape)
#
# rf = RandomForestRegressor(n_estimators= 100, random_state=0)
#
# rf.fit(original_train_features, original_train_labels);
#
# predictions = rf.predict(test_features[:,original_feature_indices])
#
# errors = abs(predictions - test_labels)
#
# print('平均温度误差:', round(np.mean(errors), 2), 'degrees.')
#
# mape = 100 * (errors / test_labels)
#
# # 这里的Accuracy为了方便观察，我们就用100减去误差了，希望这个值能够越大越好
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
#
#
#
# #===============new data==============
# original_train_features = train_features[:,original_feature_indices]
# original_test_features = test_features[:, original_feature_indices]
#
# rf = RandomForestRegressor(n_estimators= 100 ,random_state=0)
# rf.fit(original_train_features, train_labels);
#
# baseline_predictions = rf.predict(original_test_features)
# baseline_errors = abs(baseline_predictions - test_labels)
# print('平均温度误差:', round(np.mean(baseline_errors), 2), 'degrees.')
#
# baseline_mape = 100 * np.mean((baseline_errors / test_labels))
# baseline_accuracy = 100 - baseline_mape
# print('Accuracy:', round(baseline_accuracy, 2), '%.')
#
#
#
# # new data features
# rf_exp = RandomForestRegressor(n_estimators= 100, random_state=0)
# rf_exp.fit(train_features, train_labels)
# predictions = rf_exp.predict(test_features)
#
# errors = abs(predictions - test_labels)
# print('平均温度误差:', round(np.mean(errors), 2), 'degrees.')
#
# mape = np.mean(100 * (errors / test_labels))
#
# improvement_baseline = 100 * abs(mape - baseline_mape) / baseline_mape
# print('特征增多后模型效果提升:', round(improvement_baseline, 2), '%.')
#
# accuracy = 100 - mape
# print('Accuracy:', round(accuracy, 2), '%.')
#
#
# importances = list(rf_exp.feature_importances_)
#
# # 名字，数值组合在一起
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#
# # 排序
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#
# # 打印出来
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#
# plt.style.use('fivethirtyeight')
#
# # 指定位置
# x_values = list(range(len(importances)))
#
# # 绘图
# plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
#
# # x轴名字得竖着写
# plt.xticks(x_values, feature_list, rotation='vertical')
#
# # 图名
# plt.ylabel('Importance')
# plt.xlabel('Variable')
# plt.title('Variable Importances')