import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, show
import ds_charts as ds
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score,precision_score
from sklearn.neighbors import KNeighborsClassifier



file_tag = 'diabetic_IterativeImputer'
data: DataFrame = read_csv('data/MVI/diabetic_IterativeImputer_filling_missing_values.csv')

#para dados diabeticos
#target = 'readmitted'
#verypositive = 2
#positive = 1
#negative = 0
#values = {'Original': [len(data[data[target] == verypositive]), len(data[data[target] == positive]), len(data[data[target] == negative])]}



y: np.ndarray = data.pop(target).values
X: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv(f'data/MVI/out/{file_tag}_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'data/MVI/out/{file_tag}_test.csv', index=False)

values['Train'] = [len(np.delete(trnY, np.argwhere(trnY!=verypositive))), len(np.delete(trnY, np.argwhere(trnY!=positive))),len(np.delete(trnY, np.argwhere(trnY!=negative)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY!=verypositive))), len(np.delete(tstY, np.argwhere(tstY!=positive))),len(np.delete(tstY, np.argwhere(tstY!=negative)))]

plt.figure(figsize=(12,4))
ds.multiple_bar_chart([verypositive, positive, negative], values, title='Data distribution per dataset')
plt.savefig('images/classification/diabetic_IterativeImputer_readmitted_distribution.png')