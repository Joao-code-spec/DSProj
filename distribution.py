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

rows, cols = choose_grid(len(numeric_vars))

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0

for n in range(len(numeric_vars)):
    print(n)
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data_drought[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/distribution/single_boxplots.png')
show()