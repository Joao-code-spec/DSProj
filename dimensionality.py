from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# filename = 'data/algae.csv'
# data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = 'diabetic_data.csv'
data = read_csv(filename, na_values='?')

data.shape

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('images/records_variables.png')
show()