#pacotes
from pandas import read_csv, Series
from numpy import log
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots, figure, Axes, plot
from ds_charts import get_variable_types, choose_grid, HEIGHT,  multiple_bar_chart, multiple_line_chart, bar_chart
from scipy.stats import norm, expon, lognorm


register_matplotlib_converters() 
drought = 'data/drought.csv'
diabetic = 'data/diabetic_data.csv'

#data_drought = read_csv(drought, index_col=['fips', 'date'],na_values='', parse_dates=True, infer_datetime_format=True) #converte de csv para dataFrame
#data_diabetic= read_csv(diabetic, index_col=['encounter_id','patient_nbr'], na_values='?') #converte de csv para dataFrame

data_drought = read_csv(drought,na_values='', parse_dates=True, infer_datetime_format=True).drop(['fips', 'date'], axis=1) #converte de csv para dataFrame
data_diabetic= read_csv(diabetic, na_values='?').drop(['encounter_id','patient_nbr'], axis=1) #converte de csv para dataFrame


def distribution(datadF):
    if datadF.equals(data_drought):
        data_name="Drought"
    elif datadF.equals(data_diabetic):
        data_name="Diabetes"
    summary5 = datadF.describe()

    #NUMERIC VARIABLES

    numeric_vars = get_variable_types(datadF)['Numeric'] 

    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    datadF.boxplot(rot=45)
    savefig('images/distribution/global_boxplot_'+data_name+'.png')

    rows, cols = choose_grid(len(numeric_vars))

    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0

    fig, axs2 = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    k, m =0,0

    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(datadF[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)


        axs2[k, m].set_title('Histogram for %s'%numeric_vars[n])
        axs2[k, m].set_xlabel(numeric_vars[n])
        axs2[k, m].set_ylabel("nr records")
        axs2[k, m].hist(datadF[numeric_vars[n]].dropna().values, 'auto')
        k, m= (k + 1, 0) if (n+1) % cols == 0 else (k, m + 1)


    savefig('images/distribution/single_boxplots_'+data_name+'.png')
    savefig('images/distribution/single_histograms_numeric_'+data_name+'.png')

    NR_STDEV: int = 3

    outliers_iqr = []
    outliers_stdev = []
    summary5 = datadF.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            datadF[datadF[var] > summary5[var]['75%']  + iqr].count()[var] +
            datadF[datadF[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            datadF[datadF[var] > summary5[var]['mean'] + std].count()[var] +
            datadF[datadF[var] < summary5[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    savefig('images/distribution/outliers_'+data_name+'.png')
    

    symbolic_vars = get_variable_types(datadF)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')


    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = datadF[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig('images/distribution/histograms_symbolic_'+data_name+'.png')



distribution(data_diabetic)



