import matplotlib
import pandas
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show, figure, title
from ds_charts import get_variable_types, HEIGHT
from seaborn import heatmap

matplotlib.use('Agg')
register_matplotlib_converters()
filename = 'data/drought.csv'
data = read_csv(filename, index_col='date',parse_dates=True, infer_datetime_format=True)

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1

corr_mtx = abs(data.corr())

fig = figure(figsize=(cols*HEIGHT, rows*HEIGHT))
heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'images/correlation_analysis_drought.png')
show()


fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'images/sparsity_study_drought_numeric.png')
show()


filename = 'data/diabetic_data.csv'
data_2 = read_csv(filename)
numeric_vars = get_variable_types(data_2)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1

corr_mtx_2 = abs(data_2.corr())

fig = figure(figsize=(cols*HEIGHT, rows*HEIGHT))
heatmap(abs(corr_mtx_2), xticklabels=corr_mtx_2.columns, yticklabels=corr_mtx_2.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'images/correlation_analysis_diabetic.png')
show()


fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data_2[var1], data_2[var2])
savefig(f'images/sparsity_study_diabetic_numeric.png')
show()