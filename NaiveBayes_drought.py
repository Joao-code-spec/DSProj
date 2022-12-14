import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split

file_tag = 'drought'
data: DataFrame = read_csv('data\drought.csv', index_col='date',parse_dates=True, infer_datetime_format=True)
target = 'class'
positive = 1
negative = 0
values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

y: np.ndarray = data.pop(target).values
X: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv(f'data/{file_tag}_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'data/{file_tag}_test.csv', index=False)

values['Train'] = [len(np.delete(trnY, np.argwhere(trnY!=positive))),len(np.delete(trnY, np.argwhere(trnY!=negative)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY!=positive))),len(np.delete(tstY, np.argwhere(tstY!=negative)))]

plt.figure(figsize=(12,4))
ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
plt.savefig('images/classification/drought_distribution.png')

########
def plot_confusion_matrix(cnf_matrix: np.ndarray, classes_names: np.ndarray, ax: plt.Axes = None,
                          normalize: bool = False):
    if ax is None:
        ax = plt.gca()
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix'
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=CMAP)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), color='y', horizontalalignment="center")
########

from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import itertools
import matplotlib.pyplot as plt
CMAP = plt.cm.Blues

clf = BernoulliNB()
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
plt.savefig('images/classification/drought_confusion_matrix.png')

from sklearn.metrics import accuracy_score

estimators = {'GaussianNB': GaussianNB(),
              
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB, 'MultinomialNB': MultinomialNB()
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
plt.savefig(f'images/classification/drought_accuracy.png')