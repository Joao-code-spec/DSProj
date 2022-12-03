#----------------------------FOR DIABETIC_DATA----------------------------

from pandas import read_csv
from pandas import DataFrame
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()
# filename = 'data/algae.csv'
# data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = 'data/diabetic_data.csv'
data = read_csv(filename, na_values=['?',"Unknown/Invalid"])


#race
data=data.replace("Caucasian",1)
data=data.replace("AfricanAmerican",2)
data=data.replace("Hispanic",3)
data=data.replace("Other",4)
data=data.replace("Asian",5)

#gender
data=data.replace("Male",0)
data=data.replace("Female",1)
#data=data.replace("Unknown/Invalid",3)

#weight
uniques = data["weight"].dropna(inplace=False).unique()
i=0
for u in uniques:
    data=data.replace(u,i)
    i+=1

#age
data=data.replace("[0-10)",5)
data=data.replace("[10-20)",15)
data=data.replace("[20-30)",25)
data=data.replace("[30-40)",35)
data=data.replace("[40-50)",45)
data=data.replace("[50-60)",55)
data=data.replace("[60-70)",65)
data=data.replace("[70-80)",75)
data=data.replace("[80-90)",85)
data=data.replace("[90-100)",95)

#payer_code
uniques = data["payer_code"].dropna(inplace=False).unique()
i=0
for u in uniques:
    data["payer_code"]=data["payer_code"].replace(u,i)
    i+=1

#print(data["payer_code"].dropna(inplace=False).unique())

#medical_specialty
uniques = data["medical_specialty"].dropna(inplace=False).unique()
i=0
for u in uniques:
    data=data.replace(u,i)
    i+=1


#data=data.replace("Pediatrics-Endocrinology",1)
#data=data.replace("InternalMedicine",2)
#data=data.replace("Family/GeneralPractice",3)
#data=data.replace("Cardiology",4)
#data=data.replace("Surgery-General",5)
#data=data.replace("Orthopedics",6)
#data=data.replace("Gastroenterology",7)
#data=data.replace("Surgery-Cardiovascular/Thoracic",8)
#data=data.replace("Nephrology",9)

#print(data["medical_specialty"].dropna(inplace=False).unique())

#diag_1,2,3
#data=data.replace("V42",42)
#data=data.replace("V57",57)
#data=data.replace("E818",818)

uniques = pd.unique(data[["diag_1","diag_2","diag_3"]].dropna(inplace=False).values.ravel())
#uniques = data["diag_1"].dropna(inplace=False).unique()
i=0
for u in uniques:
    data["diag_1"]=data["diag_1"].replace(u,i)
    data["diag_2"]=data["diag_2"].replace(u,i)
    data["diag_3"]=data["diag_3"].replace(u,i)
    i+=1
#print(pd.unique(data[["diag_1","diag_2","diag_3"]].dropna(inplace=False).values.ravel()))

#max_glu_serum, A1Cresult
#Both
data=data.replace("None",1)
data=data.replace("Norm",2)
#max_glu_serum
data=data.replace(">200",3)
data=data.replace(">300",4)
#A1Cresult
data=data.replace(">7",5)
data=data.replace(">8",6)
#results
#print(data["max_glu_serum"].dropna(inplace=False).unique())
#print(data["A1Cresult"].dropna(inplace=False).unique())


#change, diabetesMed
data=data.replace("No",0)
data=data.replace("Ch",1)
data=data.replace("Yes",1)
#print(data["change"].dropna(inplace=False).unique())
#print(data["diabetesMed"].dropna(inplace=False).unique())

#readmitted
data=data.replace("NO",0)
data=data.replace("<30",1)
data=data.replace(">30",2)
#print(data["readmitted"].dropna(inplace=False).unique())

#medicines
data=data.replace("Down",1)
data=data.replace("Steady",2)
data=data.replace("Up",3)


#for c in data.columns:
    #print(c)
    #print(data[c].value_counts())

data.to_csv("data/my_diabetic_data.csv")



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
from ds_charts import bar_chart

variable_types = get_variable_types(data)
print(variable_types)
#counts = {}
#for tp in variable_types.keys():
#    counts[tp] = len(variable_types[tp])
#figure(figsize=(4,2))
#bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
#show()

