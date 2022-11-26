#pacotes
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT


register_matplotlib_converters() 
drought = 'data/drought.csv'

data_drought = read_csv(drought, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True) #converte de csv para dataFrame
summary5 = data_drought.describe()

numeric_vars = get_variable_types(data_drought)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

data_drought.boxplot(rot=45)
savefig('images/distribution/global_boxplot.png')
show()
