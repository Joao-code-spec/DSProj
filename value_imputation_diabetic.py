from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart
from sklearn.preprocessing import OneHotEncoder
from numpy import number,nan
from matplotlib.pyplot import figure, savefig
from sklearn.impute import SimpleImputer

register_matplotlib_converters()
file = 'diabetic_mean'
filename = 'data/my_diabetic_data.csv'
data = read_csv(filename, index_col=['encounter_id', 'patient_nbr'])

mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
savefig(f'images/value_imputation/diabetic_missing_values.png')

data.fillna(data.mean(),inplace=True)
data.to_csv(f'data/{file}_filling_missing_values.csv', index=False)

file = 'diabetic_mode'
filename = 'data/my_diabetic_data.csv'
data = read_csv(filename)

data.fillna(data.mode(),inplace=True)
data.to_csv(f'data/{file}_filling_missing_values.csv', index=False)