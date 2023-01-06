#NOME

nameOfData='Drought_Daily'

#Training e defenições

from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe
from pandas import read_csv, DataFrame, Series

file_tag = 'Drought'
index_col='date'
target='QV2M'
data = read_csv('entrega6/data/drought.forecasting_dataset_DROP.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True,dayfirst=True, infer_datetime_format=True)

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

data = aggregate_by(data, 'date', 'D')

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

#Persistence Model

from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from matplotlib.pyplot import figure, xticks, show
from matplotlib.pyplot import figure, savefig

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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'entrega6/images/Drought/forecasting/{file_tag}_persistence_eval.png')
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_persistence_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'entrega6/images/Drought/forecasting/{file_tag}_persistence_plots.png', x_label=index_col, y_label=target)
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_persistence_plots.png')

#Smoothing

from ts_functions import plot_series, HEIGHT

WIN_SIZE = 1
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_smoothing' + str(WIN_SIZE) + '.png')

WIN_SIZE = 10
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_smoothing' + str(WIN_SIZE) + '.png')

WIN_SIZE = 50
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_smoothing' + str(WIN_SIZE) + '.png')

WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_smoothing' + str(WIN_SIZE) + '.png')

WIN_SIZE = 1600
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'entrega6/images/Drought/forecasting/{nameOfData}_smoothing' + str(WIN_SIZE) + '.png')

show()