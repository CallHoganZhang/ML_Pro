import pandas as pd

features = pd.read_csv('data/temps_extended.csv')
original_features = pd.read_csv('data/temps.csv')
print(features.head(5))
print('data shape',features.shape)
print(round(features.describe(), 2))