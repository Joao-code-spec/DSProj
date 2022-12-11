from numpy import ndarray
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, savefig, show, subplots, title
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import ds_charts as ds
import matplotlib as plt
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study, multiple_bar_chart, plot_confusion_matrix
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
target = 'readmitted'

def outliers_diabetic(model, outlier_treatment):

    if outlier_treatment=="drop":
        file_tag = 'OutlierDrop_diabetic_mean'
        filename = 'data/outliers/out/OutlierDrop_diabetic_mean'
        data = read_csv('data/outliers/my_diabetic_data_drop_outliers.csv')
    else:
        file_tag = 'OutlierTruncate_diabetic_mean'
        filename = 'data/outliers/out/OutlierTruncate_diabetic_mean'
        data = read_csv('data/outliers/my_diabetic_data_truncate_outliers.csv')

    train: DataFrame = read_csv(f'{filename}_train.csv')
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filename}_test.csv')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    y = data.pop('readmitted').values
    X = data.values
    labels = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)



    #####################################

    if model == "KNN":
        eval_metric = accuracy_score
        nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20 , 25, 100]
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
        savefig('images/outliers/KNN variants/'+file_tag+'_KNN_study.png')
        #show()
        print('Best results with %d neighbors and %s'%(best[0], best[1]))  

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
        savefig('images/outliers/ModelPerformance/'+file_tag+'_KNN_study.png')
        #show()

        ############## confusion matrix
        clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1]) #escrever o classificador
        clf.fit(trnX, trnY) # treinar classificador como treining set trn
        prd_trn = clf.predict(trnX) # preverresultados do treino com base no treino
        prd_tst = clf.predict(tstX) # previsao do testing set sendo dado o testing set 

    else:
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
        plt.savefig(f'images/outliers/GaussianNB_Multi_Berno copy/naive_bayes_models_diabetic.png')
        #show()

        clf = BernoulliNB()
        clf.fit(trnX, trnY)
        prd_tst = clf.predict(tstX)

        labels: ndarray = unique(y)
        labels.sort()
        prdY: ndarray = clf.predict(tstX)
        cnf_mtx_tst: ndarray = confusion_matrix(tstY, prd_tst, labels=labels)

        clf = BernoulliNB() #Defining the KNN classifier
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
        
        plt.figure()
        multiple_bar_chart(['Train', 'Test'], evaluation, title="Model's performance over Train and Test sets", percentage=True)
        savefig('images/outliers/ModelPerformance/'+file_tag+model+'_study.png')
        #show()

##abaixo fora do else
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(confusion_matrix(tstY, prd_tst, labels=labels), labels, ax=axs[0,0], )
    plot_confusion_matrix(confusion_matrix(tstY, prd_tst, labels=labels), labels, ax=axs[0,1], normalize=True)
    plt.tight_layout()
    savefig('images/outliers/Matrix/'+file_tag+model+'matrix.png')
    #plt.show()

    ############### plot_overfitting
    if model == "KNN":
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
        savefig('images/outliers/overfitting/'+file_tag+model+'KNN.png')
outliers_diabetic("KNN", "truncate")