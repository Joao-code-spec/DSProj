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
target = 'readmitted'
verypositive = 2
positive = 1
negative = 0
values = {'Original': [len(data[data[target] == verypositive]), len(data[data[target] == positive]), len(data[data[target] == negative])]}

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
plt.savefig('images/classification/diabetic_IterativeImputer_confusion_matrix.png')
show() 

from sklearn.metrics import accuracy_score

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
plt.savefig(f'images/classification/diabetic_IterativeImputer_accuracy.png')
show()

####CATARINA

clf = BernoulliNB() #Defining the NB classifier
clf.fit(trnX, trnY) #Training the classifier
prdY = clf.predict(tstX) #predicted values for the testing set
prdY_train =clf.predict(trnX) #predicted vaçues for the training set

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

multiple_bar_chart(['Train', 'Test'], evaluation, title="Model's performance over Train and Test sets", percentage=True)
savefig('images/value_imputation/Intimputer_BernoulliNB_study.png')
show()
#### 