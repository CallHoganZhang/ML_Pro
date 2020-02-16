import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import numpy as np

features = pd.read_csv('data/temps_extended.csv')
original_features = pd.read_csv('data/temps.csv')
print(features.head(5))
print('data shape',features.shape)
print(round(features.describe(), 2))


def change_date_type(features):
    years = features['year']
    months = features['month']
    days = features['day']

    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    return dates

dates = change_date_type(features)
# print(dates)

features = pd.get_dummies(features) #one hot coding

labels = features['actual']
features = features.drop('actual', axis = 1)

feature_list = list(features.columns)

features = np.array(features)
labels = np.array(labels)



train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size = 0.2, random_state = 0)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

original_features = pd.get_dummies(original_features)
original_feature_indices = [feature_list.index(feature) for feature in
                                      feature_list if feature not in
                                      ['ws_1', 'prcp_1', 'snwd_1']]

original_labels = np.array(original_features['actual'])
original_features= original_features.drop('actual', axis = 1)

original_feature_list = list(original_features.columns)

original_features = np.array(original_features)

original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(original_features, original_labels, test_size = 0.2, random_state = 42)

print('original_train_features Shape:', original_train_features.shape)
print('original_train_features:', original_train_features.shape)
print('original_train_features:', original_train_features.shape)
print('original_train_features:', original_train_features.shape)