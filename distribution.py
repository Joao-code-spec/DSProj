#pacotes
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT


register_matplotlib_converters() 
drought = 'data/drought.csv'
diabetic = 'data/diabetic_data.csv'

data_drought = read_csv(drought, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True) #converte de csv para dataFrame
data_diabetic= read_csv(diabetic, index_col=['encounter_id', 'patient_nbr'], na_values='?') #converte de csv para dataFrame

def distribution(datadF):
    if datadF.equals(data_drought):
        data_name="Drought"
    elif datadF.equals(data_diabetic):
        data_name="Diabetes"

    summary5 = datadF.describe()

    numeric_vars = get_variable_types(datadF)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    datadF.boxplot(rot=45)
    savefig('images/distribution/global_boxplot_'+data_name+'.png')

    rows, cols = choose_grid(len(numeric_vars))

    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0

    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(datadF[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig('images/distribution/single_boxplots_'+data_name+'.png')
    show()

    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(datadF[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig('images/distribution/single_histograms_numeric_'+data_name+'.png')
    show()

distribution(data_drought)


