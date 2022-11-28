#pacotes
from pandas import read_csv, Series
from numpy import log
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots, figure, Axes
from ds_charts import get_variable_types, choose_grid, HEIGHT,  multiple_bar_chart, multiple_line_chart
from scipy.stats import norm, expon, lognorm

drought = 'data/drought.csv'
diabetic ='data/diabetic_data.csv'


data_drought = read_csv(drought, index_col=['fips', 'date'],na_values='', parse_dates=True, infer_datetime_format=True) #converte de csv para dataFrame
data_diabetic= read_csv(diabetic, index_col=['encounter_id','patient_nbr'], na_values='?') #converte de csv para dataFrame


def distribution_charts(datadF):
    if datadF.equals(data_drought):
        data_name="Drought"
    elif datadF.equals(data_diabetic):
        data_name="Diabetes"
    summary5 = datadF.describe()

    def compute_known_distributions(x_values: list) -> dict:
        distributions = dict()
        # Gaussian
        mean, sigma = norm.fit(x_values)
        distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
        # Exponential
        loc, scale = expon.fit(x_values)
        distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
        # LogNorm
        sigma, loc, scale = lognorm.fit(x_values)
        distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
        return distributions

    def histogram_with_distributions(ax: Axes, series: Series, var: str):
        values = series.sort_values().values
        ax.hist(values, 20, density=True)
        distributions = compute_known_distributions(values)
        multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

    numeric_vars = get_variable_types(datadF)['Numeric'] 

    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))

    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        print(n)
        histogram_with_distributions(axs[i, j], datadF[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    savefig('images/distribution/dist_charts'+data_name+'.png')

show()

distribution_charts(data_drought)
