from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_confusion_matrix, multiple_line_chart, horizontal_bar_chart, multiple_bar_chart, confusion_matrix, HEIGHT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_tag = 'diabetic_recall'
filename = 'data/Balancing/diabetic_undersample'
target = 'readmitted'

train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
max_depths = [5, 10, 17, 25]
max_features = [.3, .5, .7, 1]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(max_depths)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    print(d)
    for f in max_features:
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(recall_score(tstY, prdY, average="macro"))
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_model = rf

        values[f] = yvalues
    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                           xlabel='nr estimators', ylabel='recall', percentage=True)
savefig(f'images/RandomForest/diabetic/{file_tag}_rf_study.png')
print('Best depth=%d features %1.2f estimators %d, with recall=%1.2f'%(best[0], best[1], best[2], last_best))


##################################################################################################
#best model confusion matrix and bar chart
##################################################################################################


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
savefig(f'images/RandomForest/diabetic/{file_tag}_rf_best.png')


############################################################################
#feature importance
############################################################################

from numpy import std, argsort

variables = train.columns
importances = best_model.feature_importances_
stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
indices = argsort(importances)[::-1]
elems = []
#JOHN: there where no masks in teachers code
mask_importances=[]
mask_stdevs=[]
for f in range(len(variables)):
    #JOHN: prevents "useless" from spamming chart
    if(importances[indices[f]]>=0.001):
        elems += [variables[indices[f]]]
        mask_importances += [importances[indices[f]]]
        mask_stdevs += [stdevs[indices[f]]]
    #print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')
figure()
horizontal_bar_chart(elems, mask_importances, mask_stdevs , title='Random Forest Features importance', xlabel='importance', ylabel='variables')
savefig(f'images/RandomForest/diabetic/{file_tag}_rf_ranking.png')


##############################################################################
# overfit study
##############################################################################


f = 0.7
max_depth = 25
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
    rf.fit(trnX, trnY)
    prd_tst_Y = rf.predict(tstX)
    prd_trn_Y = rf.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'images/RandomForest/diabetic/overfitting_{name}.png')
plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric))
