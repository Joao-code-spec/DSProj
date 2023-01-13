from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots, savefig
from ts_functions import HEIGHT, split_dataframe

file_tag = 'diabetic_comparison'
index_col='Date'
target='glucose'
data = read_csv('entrega6/data/glucoseDateTarget.csv', index_col=[0], sep=',', decimal='.', parse_dates=True,dayfirst=True, infer_datetime_format=True)

print(data.head())

def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(data, trn_pct=0.75)

def plot_forecasting_series(trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
    _, ax = subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(figname)
    ax.plot(trn.index, trn, label='train', color='b')
    ax.plot(trn.index, prd_trn, '--y', label='train prediction')
    ax.plot(tst.index, tst, label='test', color='g')
    ax.plot(tst.index, prd_tst, '--r', label='test prediction')
    ax.legend(prop={'size': 5})


measure = 'R2'
flag_pct = False
eval_results = {}

from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'entrega6/catarina/{file_tag}_simpleAvg_eval.png')
savefig( f'entrega6/catarina/{file_tag}_simpleavg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'entrega6/catarina/{file_tag}_simpleAvg_plots.png', x_label=index_col, y_label=target)
savefig( f'entrega6/catarina/{file_tag}_simpleavg_plot.png')

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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/{file_tag}_persistence_eval.png')
savefig( f'entrega6/catarina/{file_tag}_persistence_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/{file_tag}_persistence_plots.png', x_label=index_col, y_label=target)
savefig( f'entrega6/catarina/{file_tag}_persistence_plot.png')

class RollingMeanRegressor3 (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

fr_mod = RollingMeanRegressor3()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean3'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/{file_tag}_rollingMean3_eval.png')
savefig( f'entrega6/catarina/{file_tag}_rollingmean3_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/{file_tag}_rollingMean3_plots.png', x_label=index_col, y_label=target)
savefig( f'entrega6/catarina/{file_tag}_rollingmean3_plot.png')


class RollingMeanRegressor5 (RegressorMixin):
    def __init__(self, win: int = 5):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

fr_mod = RollingMeanRegressor5()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean5'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/{file_tag}_rollingMean5_eval.png')
savefig( f'entrega6/catarina/{file_tag}_rollingmean5_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/{file_tag}_rollingMean5_plots.png', x_label=index_col, y_label=target)
savefig( f'entrega6/catarina/{file_tag}_rollingmean5_plot.png')

class RollingMeanRegressor2 (RegressorMixin):
    def __init__(self, win: int = 2):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

fr_mod = RollingMeanRegressor2()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean2'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/{file_tag}_rollingMean2_eval.png')
savefig( f'entrega6/catarina/{file_tag}_rollingmean2_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/{file_tag}_rollingMean2_plots.png', x_label=index_col, y_label=target)
savefig( f'entrega6/catarina/{file_tag}_rollingmean2_plot.png')

from ds_charts import bar_chart

figure()
bar_chart(list(eval_results.keys()), list(eval_results.values()), title = 'Basic Regressors Comparison', xlabel= 'Regressor', ylabel=measure, percentage=flag_pct, rotation = False)
savefig( f'entrega6/catarina/{file_tag}_plot.png')
