import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import data_load

def encoding_features(features):
    onehot_coding_features = pd.get_dummies(features) #one hot coding
    onehot_coding_labels = onehot_coding_features['actual']
    dropped_features = onehot_coding_features.drop('actual', axis = 1)
    dropped_features_list = list(dropped_features.columns)

    array_features = np.array(dropped_features)
    array_labels = np.array(onehot_coding_labels)
    return array_features, array_labels, dropped_features_list

def get_important_list(feature_list):
    important_feature_names = ['temp_1', 'average', 'ws_1', 'temp_2', 'friend', 'year']
    important_indices = [feature_list.index(feature) for feature in important_feature_names]
    return important_indices

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    print('平均气温误差.',np.mean(errors))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

array_features, array_labels, dropped_features_list = encoding_features(data_load.features)
train_features, test_features, train_labels, test_labels = train_test_split(array_features, array_labels, test_size = 0.1, random_state = 0)

'''
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

print('{:0.1f} years of data in the training set'.format(train_features.shape[0] / 365.))
print('{:0.1f} years of data in the test set'.format(test_features.shape[0] / 365.))
'''

important_indices = get_important_list(dropped_features_list)
important_train_features = train_features[:, important_indices]
important_test_features = test_features[:,important_indices ]

print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)

rf = RandomForestRegressor(random_state = 42)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# rf_random.fit(train_features, train_labels)
# print('best_params_', rf_random.best_params_)

# best_random = rf_random.best_estimator_
# evaluate(best_random, test_features, test_labels)


def rfFit(randomForest, model, param):
    rf_random = model(estimator=randomForest, param_distributions=param,
                              n_iter = 10, scoring='neg_mean_absolute_error',
                              cv = 3, verbose=2, random_state=42, n_jobs=-1)
    return rf_random

randomForestResult = rfFit(RandomForestRegressor(), RandomizedSearchCV, random_grid)

param_grid = {
    'bootstrap': [True],
    'max_depth': [8,10,12],
    'max_features': ['auto'],
    'min_samples_leaf': [2,3, 4, 5,6],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [800, 900, 1000, 1200]
}

rf = RandomForestRegressor()

# 网络搜索
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           scoring = 'neg_mean_absolute_error', cv = 3,
                           n_jobs = -1, verbose = 2)

grid_search.fit(train_features, train_labels)
print('best_params_', grid_search.best_params_)

best_grid = grid_search.best_estimator_
evaluate(best_grid, test_features, test_labels)