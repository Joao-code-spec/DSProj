from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# filename = 'data/algae.csv'
# data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = 'data/glucoseDateTarget.csv'
data = read_csv(filename, na_values='?',parse_dates=['Date'])

data.shape

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
show()


def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'datetime64[ns]':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'int64':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float64':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)

    return variable_types

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types

variable_types = get_variable_types(data)
print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
show()

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr
if len(mv)<1:
    mv["NA"] = 0
figure(figsize=(4,4))
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
show()