from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig, show
from ts_functions import HEIGHT, split_dataframe


def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df



file_tag = 'glucose/forecastingRedone/H_Win150_2Diff'
nameOfData='glucose_H_Win150_2Diff'
index_col='Date'
target='Glucose'
data = read_csv('data/glucoseDateTarget.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

data.sort_values('Date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
data = data.diff().diff()
data = data.dropna()
#data = aggregate_by(data, 'Date', 'D')
train, test = split_dataframe(data, trn_pct=0.75)
WIN_SIZE = 150
rolling = train.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
print(data.head())

def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test


#simple average

print("\n DOING simple average \n")
#train, test = split_dataframe(data, trn_pct=0.75)


measure = 'R2'
flag_pct = False
eval_results = {}


class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'{nameOfData}_simpleAvg_eval')
#savefig( f'images/{file_tag}_simpleAvg_eval.png')
#show()
plot_forecasting_series(train, test, prd_trn, prd_tst, f'{nameOfData}_simpleAvg_plots', x_label=index_col, y_label=target)
#savefig( f'images/{file_tag}_simpleAvg_plots.png')
#show()

#Persistence Model
print("\n DOING Persistence Model \n")



class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd

fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)
plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'{nameOfData}persisEval')
savefig( f'images/{file_tag}_persistence_eval.png')
show()
plot_forecasting_series(train, test, prd_trn, prd_tst, f'{nameOfData}_persistence_plots', x_label=index_col, y_label=target)
savefig( f'images/{file_tag}_persistence_plots.png')
show()