
from pandas import read_csv, DataFrame
from torch import manual_seed, Tensor
from torch.autograd import Variable
from ts_functions import split_dataframe, sliding_window
from sklearn.preprocessing import MinMaxScaler

index='Date'
dataset = 'diabetic'

if dataset == 'diabetic':
    file_tag = 'diabetic'
    data = read_csv('entrega6/data/glucoseDateTarget.csv', index_col=[0], sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
    target = 'glucose'
else:
    file_tag = 'drought'
    data = read_csv('entrega6/data/drought.forecasting_dataset_DROP.csv', index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
    target = 'QV2M'

data.sort_values('Date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
nr_features = 1
sc = MinMaxScaler()
data = DataFrame(sc.fit_transform(data), index=data.index, columns=data.columns)
manual_seed(1)
train, test = split_dataframe(data, trn_pct=.70)

from dsLSTM import DS_LSTM

seq_length = 4
num_epochs = 2000

trnX, trnY = sliding_window(train, seq_length = seq_length)
trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
tstX, tstY = sliding_window(test, seq_length = seq_length)
tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))

my_lstm = DS_LSTM(input_size=1, hidden_size=8, learning_rate=0.001)

for epoch in range(num_epochs+1):
    loss = my_lstm.fit(trnX, trnY)
    if epoch % 500 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss))

from sklearn.metrics import r2_score

prd_trn = my_lstm(trnX)
prd_tst = my_lstm(tstX)

print('TRAIN R2=', r2_score(trnY.data.numpy(), prd_trn.data.numpy()))
print('TEST R2=', r2_score(tstY.data.numpy(), prd_tst.data.numpy()))

from torch import manual_seed, Tensor
from torch.autograd import Variable
from dsLSTM import DS_LSTM
from ds_charts import HEIGHT, multiple_line_chart
from matplotlib.pyplot import subplots, show, savefig
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, sliding_window

best = ('',  0, 0.0)
last_best = -100
best_model = None

measure = 'R2'
flag_pct = False

learning_rate = 0.001
sequence_size = [4, 20, 60, 100]
nr_hidden_units = [1, 2, 3, 4, 8, 10]
max_iter = [500, 500, 1500, 2500]
episode_values = [max_iter[0]]
for el in max_iter[1:]:
    episode_values.append(episode_values[-1]+el)

nCols = len(sequence_size)
_, axs = subplots(1, nCols, figsize=(nCols*HEIGHT, HEIGHT), squeeze=False)
values = {}
for s in range(len(sequence_size)):
    length = sequence_size[s]
    trnX, trnY = sliding_window(train, seq_length = length)
    trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
    tstX, tstY = sliding_window(test, seq_length = length)
    tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))

    for k in range(len(nr_hidden_units)):
        hidden_units = nr_hidden_units[k]
        yvalues = []
        model = DS_LSTM(input_size=nr_features, hidden_size=hidden_units, learning_rate=learning_rate)
        next_episode_i = 0
        for n in range(1, episode_values[-1]+1):
            model.fit(trnX, trnY)
            if n == episode_values[next_episode_i]:
                next_episode_i += 1
                prd_tst = model.predict(tstX)
                yvalues.append((PREDICTION_MEASURES[measure])(tstY, prd_tst))
                print((f'LSTM - seq length={length} hidden_units={hidden_units} and nr_episodes={n}->{yvalues[-1]:.2f}'))
                if yvalues[-1] > last_best:
                    best = (length, hidden_units, n)
                    last_best = yvalues[-1]
                    best_model = model
        values[hidden_units] = yvalues

    multiple_line_chart(
        episode_values, values, ax=axs[0, s], title=f'LSTM seq length={length}', xlabel='nr episodes', ylabel=measure, percentage=flag_pct)
print(f'Best results with seq length={best[0]} hidden={best[1]} episodes={best[2]} ==> measure={last_best:.2f}')
savefig(f'entrega6/catarina/{file_tag}_lstm_study.png')

trnX, trnY = sliding_window(train, seq_length = best[0])
trainY = DataFrame(trnY)
trainY.index = train.index[best[0]+1:]
trainY.columns = [target]
trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
prd_trn = best_model.predict(trnX)
prd_trn = DataFrame(prd_trn)
prd_trn.index=train.index[best[0]+1:]
prd_trn.columns = [target]

tstX, tstY = sliding_window(test, seq_length = best[0])
testY = DataFrame(tstY)
testY.index = test.index[best[0]+1:]
testY.columns = [target]
tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))
prd_tst = best_model.predict(tstX)
prd_tst = DataFrame(prd_tst)
prd_tst.index=test.index[best[0]+1:]
prd_tst.columns = [target]

plot_evaluation_results(trnY.data.numpy(), prd_trn, tstY.data.numpy(), prd_tst, f'entrega6/catarina/{file_tag}_lstm_eval.png')
savefig(f'entrega6/catarina/{file_tag}_lstm_eval.png')
plot_forecasting_series(trainY, testY, prd_trn.values, prd_tst.values, f'entrega6/catarina/{file_tag}_lstm_plots.png', x_label=index, y_label=target)
savefig(f'entrega6/catarina/{file_tag}_lstm_plots.png')