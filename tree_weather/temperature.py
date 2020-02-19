import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import data_load

def change_date_type(features):
    years = features['year']
    months = features['month']
    days = features['day']

    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    return dates

def encoding_features(features):
    onehot_coding_features = pd.get_dummies(features) #one hot coding
    onehot_coding_labels = onehot_coding_features['actual']
    dropped_features = onehot_coding_features.drop('actual', axis = 1)
    dropped_features_list = list(dropped_features.columns)

    array_features = np.array(dropped_features)
    array_labels = np.array(onehot_coding_labels)
    return array_features, array_labels, dropped_features_list

def get_feature_indice(feature_list):
    original_feature_indices = [feature_list.index(feature) for feature in
                                          feature_list if feature not in
                                          ['ws_1', 'prcp_1', 'snwd_1']]
    return original_feature_indices

def randomForest_analyse(train_features, train_labels,test_features, test_labels):
    rf = RandomForestRegressor(n_estimators= 100 ,random_state=0)
    rf.fit(train_features, train_labels)
    baseline_predictions = rf.predict(test_features)
    baseline_errors = abs(baseline_predictions - test_labels)
    print('平均温度误差:', round(np.mean(baseline_errors), 2), 'degrees.')
    baseline_mape = 100 * np.mean((baseline_errors / test_labels))
    baseline_accuracy = 100 - baseline_mape
    print('Accuracy:', round(baseline_accuracy, 2), '%.')

def new_data_feature(train_features, test_features, feature_indices):
    original_train_features = train_features[:,feature_indices]
    original_test_features = test_features[:, feature_indices]
    return original_train_features, original_test_features

dates = change_date_type(data_load.features)
# print(dates)

array_features, array_labels, dropped_features_list = encoding_features(data_load.features)
train_features, test_features, train_labels, test_labels = train_test_split(array_features, array_labels, test_size = 0.1, random_state = 0)
# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

array_original_features, array_original_labels, original_dropped_features_list = encoding_features(data_load.original_features)
original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(array_original_features, array_original_labels, test_size = 0.1, random_state = 42)
# print('original_train_features Shape:', original_train_features.shape)
# print('original_train_labels:', original_train_labels.shape)
# print('original_test_features:', original_test_features.shape)
# print('original_label_features:', original_test_labels.shape)

original_feature_indices = get_feature_indice(dropped_features_list)

randomForest_analyse(original_train_features, original_train_labels, test_features[:,original_feature_indices], test_labels)

new_train_features, new_test_features = new_data_feature(train_features, test_features, original_feature_indices)
randomForest_analyse(new_train_features, train_labels,new_test_features, test_labels)

randomForest_analyse(train_features, train_labels,test_features, test_labels)