
from pandas import read_csv, DataFrame, Series

file_tag = 'Drought_with_smothing'
filename = 'Drought forecasting'
index_col='date'
target='QV2M'
data = read_csv('entrega6/data/drought.forecasting_dataset_DROP.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True,dayfirst=True, infer_datetime_format=True)


def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(data, trn_pct=0.75)

test.to_csv(f'{filename}_test.csv')
train.to_csv(f'{filename}_train.csv')
