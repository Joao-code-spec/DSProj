from pandas import DataFrame, read_csv, unique

df=read_csv('data/glucose.csv')
df=df.dropna()
df.to_csv("data/glucoseMRows.csv")