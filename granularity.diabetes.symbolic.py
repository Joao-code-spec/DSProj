from pandas import read_csv
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types

register_matplotlib_converters()

filename = 'data/diabetic_data.csv'
data= read_csv(filename)
df=pd.read_csv(filename)
values = {'nr records': df.shape[0], 'nr variables': df.shape[1]}

symbolic_vars = get_variable_types(df)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = 3, 2
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

df['gender'] = df['gender'].replace(['Female', 'Male'], ['Known', 'Known'])
i, j = 0, 0
axs[i, j].set_title('Histogram for %s'%symbolic_vars[1])
axs[i, j].set_xlabel(symbolic_vars[1])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(df[symbolic_vars[1]])


#Nova abertura do Excel para poder ler oa valores sem serem modificados

df=pd.read_csv(filename)
symbolic_vars = get_variable_types(df)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')
i, j = 0, 1
axs[i, j].set_title('Histogram for %s'%symbolic_vars[1])
axs[i, j].set_xlabel(symbolic_vars[1])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(df[symbolic_vars[1]])
              
#Agora para a Idade

df=pd.read_csv(filename)

symbolic_vars = get_variable_types(df)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

df['age'] = df['age'].replace(['[0-10)', '[10-20)', '[20-30)', '[40-50)', '[30-40)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], ['<20', '20 a 60', '20 a 60', '20 a 60', '20 a 60', '20 a 60','mais de 60','mais de 60','mais de 60','mais de 60'])
i, j = 1, 0
axs[i, j].set_title('Histogram for %s'%symbolic_vars[2])
axs[i, j].set_xlabel(symbolic_vars[2])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(df[symbolic_vars[2]])


#Nova abertura do Excel para poder ler oa valores sem serem modificados

df=pd.read_csv(filename)
symbolic_vars = get_variable_types(df)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

i, j = 1, 1
axs[i, j].set_title('Histogram for %s'%symbolic_vars[2])
axs[i, j].set_xlabel(symbolic_vars[2])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(df[symbolic_vars[2]])

#Agora para a Raça

df=pd.read_csv(filename)

symbolic_vars = get_variable_types(df)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

df['race'] = df['race'].replace(['AfricanAmerican', 'Hispanic', 'Asian', 'Caucasian'], ['Known', 'Known', 'Known', 'Known'])
i, j = 2, 0
axs[i, j].set_title('Histogram for %s'%symbolic_vars[0])
axs[i, j].set_xlabel(symbolic_vars[0])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(df[symbolic_vars[0]])


#Nova abertura do Excel para poder ler oa valores sem serem modificados

df=pd.read_csv(filename)
symbolic_vars = get_variable_types(df)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

i, j = 2, 1
axs[i, j].set_title('Histogram for %s'%symbolic_vars[0])
axs[i, j].set_xlabel(symbolic_vars[0])
axs[i, j].set_ylabel('nr records')
axs[i, j].hist(df[symbolic_vars[0]])

savefig('images/granularity_diabetes_symbolic_study.png')
show()