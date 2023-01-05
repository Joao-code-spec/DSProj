from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from ds_charts import bar_chart
from matplotlib.pyplot import figure, xticks, savefig, show, subplots

#ASHRAE

data = read_csv('entrega6/data/drought.forecasting_dataset_DROP.csv', index_col='date', sep=',', decimal='.', parse_dates=True,dayfirst=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data, x_label='Date', y_label='Specific Humidity at 2 Meters (g/kg)', title='ASHRAE original')
xticks(rotation = 45)
savefig('entrega6/images/Drought/ASHRAEdrought.png')

#aggregate_by

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

#DAILY

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'Date', 'D')
plot_series(agg_df, title='Daily', x_label='Date', y_label='Specific Humidity at 2 Meters (g/kg)')
xticks(rotation = 45)
savefig('entrega6/images/Drought/Dailydrought.png')


#Weekly

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'timestamp', 'W')
plot_series(agg_df, title='Weekly', x_label='timestamp', y_label='Specific Humidity at 2 Meters (g/kg)')
xticks(rotation = 45)
savefig('entrega6/images/Drought/Weeklydrought.png')


#Monthly

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'timestamp', 'M')
plot_series(agg_df, title='Monthly', x_label='timestamp', y_label='Specific Humidity at 2 Meters (g/kg)')
xticks(rotation = 45)
savefig('entrega6/images/Drought/Monthlydrought.png')


#
#box plot
#

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
QV2M = data['QV2M']
ax.boxplot(QV2M)
ax.set_title('Boxplot for QV2M')
savefig('entrega6/images/Drought/DroughtBox.png')

#
#histogram
#

#DAILY

histograma = aggregate_by(data, 'Date', 'D')
bins = (15, 30, 60)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram (Daily) for Drought reading %d bins'%bins[j])
    axs[j].set_xlabel('Specific Humidity at 2 Meters (g/kg)')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(histograma.values, bins=bins[j])
savefig('entrega6/images/Drought/DroughtHistogramDaily.png')

#Weekly

histograma = aggregate_by(data, 'Date', 'W')
bins = (15, 30, 60)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram (Weekly) for Drought reading %d bins'%bins[j])
    axs[j].set_xlabel('Specific Humidity at 2 Meters (g/kg)')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(histograma.values, bins=bins[j])
savefig('entrega6/images/Drought/DroughtHistogramWeekly.png')

#Monthly

histograma = aggregate_by(data, 'Date', 'M')
bins = (15, 30, 60)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram (Monthly) for Drought reading %d bins'%bins[j])
    axs[j].set_xlabel('Specific Humidity at 2 Meters (g/kg)')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(histograma.values, bins=bins[j])
savefig('entrega6/images/Drought/DroughtHistogramMonthly.png')