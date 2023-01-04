from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show
from ts_functions import plot_series, HEIGHT
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from ds_charts import bar_chart

data = read_csv('entrega6/data/drought.forecasting_dataset_DROP.csv', index_col='date', sep=',', decimal='.', parse_dates=True,dayfirst=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data, x_label='Date', y_label='Specific Humidity at 2 Meters (g/kg)', title='ASHRAE original')
xticks(rotation = 45)
savefig('entrega6/images/Drought/ASHRAEdrought.png')
show()

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'Date', 'D')
plot_series(agg_df, title='Daily', x_label='Date', y_label='Specific Humidity at 2 Meters (g/kg)')
xticks(rotation = 45)
savefig('entrega6/images/Drought/Dailydrought.png')
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'timestamp', 'W')
plot_series(agg_df, title='Weekly', x_label='timestamp', y_label='Specific Humidity at 2 Meters (g/kg)')
xticks(rotation = 45)
savefig('entrega6/images/Drought/Weeklydrought.png')
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'timestamp', 'M')
plot_series(agg_df, title='Monthly', x_label='timestamp', y_label='Specific Humidity at 2 Meters (g/kg)')
xticks(rotation = 45)
savefig('entrega6/images/Drought/Monthlydrought.png')
show()