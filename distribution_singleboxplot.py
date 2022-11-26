#pacotes
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT


register_matplotlib_converters() 
drought = 'data/drought.csv'

data_drought = read_csv(drought, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True) #converte de csv para dataFrame
numeric_vars = get_variable_types(data_drought)['Numeric']

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

