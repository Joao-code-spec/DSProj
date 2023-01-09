from pandas import DataFrame, read_csv, unique


df=read_csv('entrega6/data/glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
df.sort_values('Date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
df=df.dropna()
#df.to_csv("data/glucoseMRows.csv")

day_df = df.copy().groupby(df.index.date).mean()

day_df.to_csv(f'entrega6/data/glucose_final.csv', index=True, index_label='Date')


df=read_csv('entrega6/data/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
df.sort_values('date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
df=df.dropna()
#df.to_csv("data/glucoseMRows.csv")

#day_df = df.copy().groupby(df.index.date).mean()
day_df=df

day_df.to_csv(f'entrega6/data/drought_final.csv', index=True, index_label='date')