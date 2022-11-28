from pandas import read_csv
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

filename = 'data/drought - Copy.csv'
data= read_csv(filename)
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')


rows, cols = 1, 3
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

i, j = 0, 0
axs[i, j].set_title('Histogram for %s'%variables[1])
axs[i, j].set_xlabel(variables[1])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(data[variables[1]])

#Meses

i, j = 0, 1
axs[i, j].set_title('Histogram for %s'%variables[2])
axs[i, j].set_xlabel(variables[2])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(data[variables[2]])
              
#Dias

i, j = 0, 2
axs[i, j].set_title('Histogram for %s'%variables[3])
axs[i, j].set_xlabel(variables[3])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(data[variables[3]])

savefig('images/granularity_drought_symbolic_study.png')
show()