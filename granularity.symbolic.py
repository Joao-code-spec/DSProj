from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types

register_matplotlib_converters()

filename = 'data/diabetic_data.csv'
data= read_csv(filename)
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    axs[i, j].set_title('Histogram for %s'%symbolic_vars[n])
    axs[i, j].set_xlabel(symbolic_vars[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].scatter(symbolic_vars[n], data.shape[0])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/granularity_symbolic.png')
show()
