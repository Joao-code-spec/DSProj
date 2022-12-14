from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import HEIGHT,subplots, confusion_matrix, multiple_line_chart, plot_overfitting_study, multiple_bar_chart, plot_confusion_matrix
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score

file_tag = 'diabetes_recall'
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

eval_metric = recall_score
nvalues = nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 25, 27, 30, 32, 35, 36, 37, 39, 40]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}

best = (0, '')
last_best = 0
for d in dist:
    y_tst_values = []
    for n in nvalues:
        print(n)
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y,average="macro"))
        if y_tst_values[-1] > last_best:
            best = (n, d)
            last_best = y_tst_values[-1]
    values[d] = y_tst_values

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(recall_score), percentage=True)
savefig('images/balancing/KNN/diabetic/undersample/' + file_tag + '_knn_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))



#eval results
#best= (37 , 'manhattan')
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

clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, average_param="macro")
savefig(f'images/balancing/KNN/diabetic/undersample/{file_tag}_knn_best.png')
show()

