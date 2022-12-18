from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.tree import DecisionTreeClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.metrics import accuracy_score
from ds_charts import HEIGHT,subplots, confusion_matrix, multiple_line_chart, plot_overfitting_study, multiple_bar_chart, plot_confusion_matrix
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score

#

#Preencher com os fiheiros necessários
file_tag = 'diabtic'
filename = 'data/Balancing/diabetic_undersample'
target = 'readmitted'

#Prencher para o Overfitting
f = 'entropy'
eval_metric = accuracy_score

#

train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values

trnY=trnY.astype('float')

labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
max_depths = [2, 5, 10, 15, 20, 25]
criteria = ['entropy', 'gini']
best = ('',  0, 0.0)
last_best = 0
best_model = None

figure()
fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for imp in min_impurity_decrease:
            tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (f, d, imp)
                last_best = yvalues[-1]
                best_model = tree

        values[d] = yvalues
    multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                           xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
savefig(f'images/DT/{file_tag}_dt_study.png')
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

#

labels_2=labels

from sklearn import tree

labels = [str(value) for value in labels]
tree.plot_tree(best_model, feature_names=train.columns, class_names=labels)
savefig(f'images/DT/{file_tag}_dt_best_tree.png')

#

if target == 'class':
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)

    plot_evaluation_results(labels_2, trnY, prd_trn, tstY, prd_tst)
    savefig(f'images/DT/{file_tag}_dt_best.png')
if target == 'readmitted':
    def plot_evaluation_results(labels: ndarray, trn_y, prd_trn, tst_y, prd_tst, pos_value: int = 1, average_param: str = 'binary'):

        def compute_eval(real, prediction):
                evaluation = {
                'acc': accuracy_score(real, prediction),
                'recall': recall_score(real, prediction, pos_label=pos_value, average=average_param),
                'precision': precision_score(real, prediction, pos_label=pos_value, average=average_param),
                'f1': f1_score(real, prediction, pos_label=pos_value, average=average_param)
                }
                return evaluation

        eval_trn = compute_eval(trn_y, prd_trn)
        eval_tst = compute_eval(tst_y, prd_tst)
        evaluation = {}
        for key in eval_trn.keys():
            evaluation[key] = [eval_trn[key], eval_tst[key]]

        _, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
        multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets", percentage=True)

        cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
        plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1], title='Test')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param="macro")
    savefig(f'images/{file_tag}_knn_best.png')

#

from numpy import argsort, arange
from ds_charts import horizontal_bar_chart
from matplotlib.pyplot import Axes

variables = train.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
elems = []
imp_values = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
savefig(f'images/DT/{file_tag}_dt_ranking.png')

#

from ds_charts import plot_overfitting_study

imp = 0.0001
f = 'entropy'
#eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for d in max_depths:
    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
    tree.fit(trnX, trnY)
    prdY = tree.predict(tstX)
    prd_tst_Y = tree.predict(tstX)
    prd_trn_Y = tree.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
figure()
plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str(eval_metric))
savefig(f'images/DT/{file_tag}_overfitting.png')