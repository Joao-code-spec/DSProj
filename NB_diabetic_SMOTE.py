import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, show
import ds_charts as ds
from ds_charts import multiple_bar_chart, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import ndarray
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import itertools


file_tag = 'diabetic_SMOTE'
data = read_csv('data/Balancing/unbalanced.csv', index_col=0)
target = 'readmitted'
data[target]=2*data[target]

verypositive = 2
positive = 1
negative = 0
values = {'Original': [len(data[data[target] == verypositive]), len(data[data[target] == positive]), len(data[data[target] == negative])]}


######
from pandas import Series
from imblearn.over_sampling import SMOTE
RANDOM_STATE = 42

smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
y = data.pop(target).values
X = data.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(data.columns) + [target]
df_smote.to_csv(f'data/Balancing/{file_tag}_train.csv', index=False)

train=df_smote

trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values

######
labels: np.ndarray = unique(y)
labels.sort()
tnX, tstX, tnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)



test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'data/Balancing/{file_tag}_test.csv', index=False)

values['Train'] = [len(np.delete(trnY, np.argwhere(trnY!=verypositive))), len(np.delete(trnY, np.argwhere(trnY!=positive))),len(np.delete(trnY, np.argwhere(trnY!=negative)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY!=verypositive))), len(np.delete(tstY, np.argwhere(tstY!=positive))),len(np.delete(tstY, np.argwhere(tstY!=negative)))]

plt.figure(figsize=(12,4))
ds.multiple_bar_chart([verypositive, positive, negative], values, title='Data distribution per dataset')
plt.savefig(f'images/Balancing/{file_tag}_distribution.png')


CMAP = plt.cm.Blues

clf = MultinomialNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)

labels: ndarray = unique(y)
labels.sort()
prdY: ndarray = clf.predict(tstX)
cnf_mtx_tst: ndarray = confusion_matrix(tstY, prdY, labels=labels)

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(confusion_matrix(tstY, prdY, labels=labels), labels, ax=axs[0,0], )
plot_confusion_matrix(confusion_matrix(tstY, prdY, labels=labels), labels, ax=axs[0,1], normalize=True)
plt.tight_layout()
plt.savefig(f'images/balancing/{file_tag}_MultinomialNB_confusion_matrix.png')


estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))

plt.figure()
ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
plt.savefig(f'images/balancing/{file_tag}_accuracy.png')

####CATARINA

clf = MultinomialNB() #Defining the NB classifier
clf.fit(trnX, trnY) #Training the classifier
prdY = clf.predict(tstX) #predicted values for the testing set
prdY_train =clf.predict(trnX) #predicted va??ues for the training set

recall_test=recall_score(prdY, tstY,average="macro")
accuracy_test=accuracy_score(prdY, tstY)
precision_test= precision_score(prdY, tstY,average="macro")

recall_train=recall_score(prdY_train, trnY,average="macro")
accuracy_train=accuracy_score(prdY_train, trnY)
precision_train= precision_score(prdY_train, trnY,average="macro")

evaluation = {
        'Accuracy': [accuracy_train, accuracy_test],
        'Recall': [recall_train, recall_test],
        'Precision': [precision_train, precision_test]}
plt.figure()
multiple_bar_chart(['Train', 'Test'], evaluation, title="Model's performance over Train and Test sets", percentage=True)
savefig(f'images/balancing/{file_tag}_MultinomialNB_study.png')