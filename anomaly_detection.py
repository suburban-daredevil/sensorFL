import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE

from DBSCAN_anomaly_detection import GridSearchHelper
from helper_functions.print_head_df import print_head_df

# url = 'https://raw.githubusercontent.com/anik801/BDL_data_1/refs/heads/main/data/plus/rpi_20_plus.csv'

url1 = 'https://raw.githubusercontent.com/anik801/BDL_data_1/refs/heads/main/data/plus/rpi_21_plus.csv'

df = pd.read_csv(url1)

df['date_time'] = pd.to_datetime(df['date_time'])
print_head_df(df)

print_head_df(df)

df.drop(columns=['date_time', 'id', 'rpi_id'], axis = 1, inplace = True)
print_head_df(df)

# Isolation Forest based anomaly detection technique
outliers_fraction = float(0.1)

print('\nStandardizing the dataframe\n')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns = df.columns)
df = df_scaled.copy()

print('\nStandardized Dataframe\n')
print_head_df(df)

# train isolation forest
model =  IsolationForest(contamination = outliers_fraction)
model.fit(df)
df['anomaly'] = model.predict(df)

print_head_df(df)

count_of_anomalies = 0
for i in df['anomaly']:
    if (i == -1):
        count_of_anomalies += 1

print('Number of anomalies:', count_of_anomalies)

X = df.drop(columns=['anomaly'])
y = df['anomaly']

counter = Counter(y)
print('\nBefore SMOTE transformation\n')
for key in counter.keys():
    print('Class:', key, '=', counter[key])

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

counter = Counter(y)
print('\nAfter SMOTE transformation\n')
for key in counter.keys():
    print('Class:', key, '=', counter[key])

df.to_csv('rpi_21_plus_with_anomalies.csv', index=False)
print('Dataframe saved to file successfully!')

# visualization
# fig, ax = plt.subplots(figsize=(10,6))
# a = df.loc[df['anomaly'] == -1, ['Total']] #anomaly
# ax.plot(df.index, df['Total'], color='black', label = 'Normal')
# ax.scatter(a.index,a['Total'], color='red', label = 'Anomaly')
# plt.legend()
# plt.show()

# DBSCAN GridSearch Helper Function
# best_parameter = GridSearchHelper(df)
# print('\nBest Parameters: \n', best_parameter)


