from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types

register_matplotlib_converters()

filename = 'data/diabetic_data.csv'
data= read_csv(filename)
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = len(symbolic_vars), 3
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    for j in range(3):
        axs[i, j].set_title("Histogram")
        axs[i, j].set_xlabel(symbolic_vars[i])
        axs[i, j].set_ylabel('nr records')
        axs[i, j].hist(data[symbolic_vars[i]], data.shape[0])
savefig(f'images/sparsity_study_symbolic.png')
show()