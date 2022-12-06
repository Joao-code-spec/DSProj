from pandas import read_csv, DataFrame
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart
from sklearn.preprocessing import OneHotEncoder
from numpy import number,nan
from matplotlib.pyplot import figure, savefig
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

register_matplotlib_converters()
file = 'diabetic_IterativeImputer'
filename = 'data/my_diabetic_data.csv'
data = read_csv(filename, index_col=0)

mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
savefig(f'images/value_imputation/diabetic_missing_values.png')

exEstimator = DecisionTreeRegressor(max_features='sqrt', random_state=42)
exStyle = 'descending'
exImputer = IterativeImputer(estimator=exEstimator, imputation_order=exStyle, random_state=42)
exImputer.fit(data)
data = DataFrame(exImputer.transform(data), columns = data.columns)

data.to_csv(f'data/MVI/{file}_filling_missing_values.csv', index=False)

file = 'diabetic_mean'
filename = 'data/my_diabetic_data.csv'
data = read_csv(filename, index_col=0)

data.fillna(data.mean(),inplace=True)
data.to_csv(f'data/MVI/{file}_filling_missing_values.csv', index=False)

