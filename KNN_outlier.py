from numpy import ndarray
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, savefig, show, subplots, title
from sklearn.neighbors import KNeighborsClassifier
import ds_charts as ds
import matplotlib as plt
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study, multiple_bar_chart
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
CMAP = plt.cm.Blues
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


##catarina ################################
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score,precision_score
from sklearn.neighbors import KNeighborsClassifier

######################################


x=1








file_tag = 'diabetic_mean'
filename = 'data/MVI/out/diabetic_mean'
target = 'readmitted'

train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

eval_metric = accuracy_score
nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    print(d)
    y_tst_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        if y_tst_values[-1] > last_best:
            best = (n, d)   
            last_best = y_tst_values[-1]
    values[d] = y_tst_values
figure()

multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
savefig('images/'+file_tag+'_knn_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))  

####CATARINA
if x==1:
    data = read_csv('data/MVI/diabetic_IterativeImputer_filling_missing_values.csv')
else:
    data = read_csv('data/MVI/diabetic_mean_filling_missing_values.csv')

y = data.pop('readmitted').values
X = data.values
labels = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1]) #Defining the KNN classifier
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
savefig('images/value_imputation/'+file_tag+'_knn_study.png')
show()

#### CATARINA



############### confusion matrix

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

############### chamar a funcao da confusion matrix ############# alterar data AQUI O NOME COMLETO NAO COMO LA EM CIMA

clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1]) #escrever o classificador
clf.fit(trnX, trnY) # treinar classificador como treining set trn
prd_trn = clf.predict(trnX) # preverresultados do treino com base no treino
prd_tst = clf.predict(tstX) # previsao do testing set sendo dado o testing set 

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(confusion_matrix(tstY, prd_tst, labels=labels), labels, ax=axs[0,0], )
plot_confusion_matrix(confusion_matrix(tstY, prd_tst, labels=labels), labels, ax=axs[0,1], normalize=True)
plt.tight_layout()
plt.show()
savefig('images/'+file_tag+'matrix.png')

############### plot_overfitting

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig('images/overfitting_'+file_tag+'.png')

d = 'euclidean'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in nvalues:
    print(n)
    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(trnX, trnY)
    prd_tst_Y = knn.predict(tstX)
    prd_trn_Y = knn.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))