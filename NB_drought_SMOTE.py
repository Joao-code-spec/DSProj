import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, show
import ds_charts as ds
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split
from numpy import ndarray
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import itertools




file_tag = 'drought_SMOTE'
data = read_csv('data/Balancing/unbalanced_drought.csv', index_col=0)
target = 'class'

positive = 1
negative = 0
values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}




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
df_smote.to_csv(f'data/{file_tag}_smote.csv', index=False)

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

values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))),len(np.delete(trnY, np.argwhere(trnY==positive)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))),len(np.delete(tstY, np.argwhere(tstY==positive)))]

plt.figure(figsize=(12,4))
ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
plt.savefig(f'images/Balancing/{file_tag}_distribution.png')

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
plt.savefig(f'images/balancing/{file_tag}_BernoulliNB_confusion_matrix.png')


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
    yvalues.append(recall_score(tstY, prdY, average="macro"))

plt.figure()
ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='recall', percentage=True)
plt.savefig(f'images/balancing/{file_tag}_recall.png')

####CATARINA

clf = BernoulliNB() #Defining the NB classifier
clf.fit(trnX, trnY) #Training the classifier
prdY = clf.predict(tstX) #predicted values for the testing set
prdY_train =clf.predict(trnX) #predicted values for the training set

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
savefig(f'images/balancing/{file_tag}_BernoulliNB_study.png')