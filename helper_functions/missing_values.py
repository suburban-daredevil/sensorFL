def missing_values(df):
    print('The missing values are:', df.isnull().sum())