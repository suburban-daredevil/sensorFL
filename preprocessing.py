import pandas as pd
from sklearn.model_selection import train_test_split
from helper_functions.print_head_df import print_head_df
from helper_functions.missing_values import missing_values

# df = pd.read_csv('rpi_20_plus_with_anomalies.csv')
# df = pd.read_csv('rpi_21_plus/rpi_21_plus_with_anomalies.csv')

df = pd.read_csv('rpi_20_plus/rpi_20_plus_with_anomalies.csv')
print_head_df(df)

'''
dropping the first column - unwanted
'''
df = df.iloc[:,1:]

'''
Checking for NULL Values
'''

missing_values(df)

'''
Dropping the missing values
'''
df.dropna(inplace=True)

missing_values(df)


'''
Train and test split
'''
X = df.drop(columns = 'anomaly', axis = 1)
y = df['anomaly']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    stratify = y,
                                                    shuffle = True,
                                                    random_state = 6704)

X_train.to_csv('rpi_20_plus/x_train_rpi_20_plus.csv', index = False)
X_test.to_csv('rpi_20_plus/x_test_rpi_20_plus.csv', index = False)
y_train.to_csv('rpi_20_plus/y_train_rpi_20_plus.csv', index = False)
y_test.to_csv('rpi_20_plus/y_test_rpi_20_plus.csv', index = False)

# X_train.to_csv('x_train_rpi_21_plus.csv', index = False)
# X_test.to_csv('x_test_rpi_21_plus.csv', index = False)
# y_train.to_csv('y_train_rpi_21_plus.csv', index = False)
# y_test.to_csv('y_test_rpi_21_plus.csv', index = False)