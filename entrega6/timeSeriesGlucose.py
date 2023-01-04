from pandas import read_csv
from matplotlib.pyplot import figure, xticks, savefig, show
from ts_functions import plot_series, HEIGHT

data = read_csv('data/glucoseDateTarget.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

data.sort_values('Date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='Date', y_label='Glucose', title='Glucose')
xticks(rotation = 45)
savefig('images/glucose/GlucoseHour.png')
show()

day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Daily consumptions', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig('images/glucose/glucoseDay.png')
show()