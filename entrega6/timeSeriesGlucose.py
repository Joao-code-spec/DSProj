from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, savefig, show, subplots
from ts_functions import plot_series, HEIGHT

#help function
def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df


data = read_csv('data/glucoseDateTarget.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
data.sort_values('Date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)

#hour

print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='Date', y_label='Glucose', title='Hour Glucose')
xticks(rotation = 45)
savefig('images/glucose/GlucoseHour.png')
show()

#day

day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Daily Glucose', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
savefig('images/glucose/glucoseDay.png')
show()


#week
index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['Date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('Date', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Weekly Glucose', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
savefig('images/glucose/glucoseWeek.png')
show()

#month
index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['Date'] = index.drop_duplicates().to_timestamp()
month_df.set_index('Date', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Monthly Glucose', x_label='Date', y_label='Glucose')
savefig('images/glucose/glucoseMonth.png')
show()

#quarterly probably not needed

#data distribution
#five number summery and box plot


index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['Date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('Date', drop=True, inplace=True)
_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))
savefig('images/glucose/glucose5Num.png')
show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0])
week_df.boxplot(ax=axs[1])
savefig('images/glucose/glucoseBox.png')
show()

#histogram

#data = aggregate_by(data, 'Date', 'D')
bins = (15, 30, 60)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for glucoseHour reading %d bins'%bins[j])
    axs[j].set_xlabel('Glucose')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j])
savefig('images/glucose/glucoseHistogramHour.png')
show()

#Data Stationarity
from numpy import ones
from pandas import Series

dt_series = Series(data['Glucose'])

mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
series = {'glucose': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='Date', y_label='Glucose', title='Stationary study', show_std=True)
savefig('images/glucose/glucoseStattionarity1.png')
show()

BINS = 60
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'glucose': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='time', y_label='Glucose', title='Stationary study', show_std=True)
savefig('images/glucose/glucoseStattionarity60.png')
show()
