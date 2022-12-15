from numpy import ndarray
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, savefig, show, subplots, title
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import ds_charts as ds
import matplotlib as plt
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study, multiple_bar_chart, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
import itertools
CMAP = plt.cm.Blues
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#target = 'readmitted'

#Parte do codigo a mudar sempre que queremos fazer outros testes
x=1

image_Folder="Scaling"

if x==1:
    data = read_csv('data/Scaling/my_diabetic_data_scaled_minmax.csv')
    operation_type = "scalled_minmax"
else:
    data = read_csv('data/Scaling/my_diabetic_data_scaled_zscore.csv')
    operation_type = "scalled_zscore"

#making training and testing sets
y = data.pop('readmitted').values
X = data.values

y=y.astype('float')
X=X.astype('float')



trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    

##
def Tests_diabetic(model):

    labels = unique(y)
    labels.sort()

    if model=="KNN":
        file_tag = operation_type + '_diabetic'
    elif model=="NB":
        file_tag = operation_type + '_diabetic'

    if model == "KNN":
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
        savefig('images/' +image_Folder+ '/KNN variants/'+file_tag+'_KNN_study.png')
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
        savefig('images/'+image_Folder+'/ModelPerformance/'+file_tag+'_KNN_study.png')
        #show()

        # confusion matrix
        clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1]) #escrever o classificador
        clf.fit(trnX, trnY) # treinar classificador como treining set trn
        prd_trn = clf.predict(trnX) # preverresultados do treino com base no treino
        prd_tst = clf.predict(tstX) # previsao do testing set sendo dado o testing set 

    else:
        estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
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
        plt.savefig(f'images/'+image_Folder+'/GaussianNB_Multi_Berno copy/naive_bayes_models_diabetic.png')
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
        savefig('images/'+image_Folder+'/ModelPerformance/'+file_tag+model+'_study.png')
        #show()

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(confusion_matrix(tstY, prd_tst, labels=labels), labels, ax=axs[0,0], )
    plot_confusion_matrix(confusion_matrix(tstY, prd_tst, labels=labels), labels, ax=axs[0,1], normalize=True)
    plt.tight_layout()
    savefig('images/'+image_Folder+'/Matrix/'+file_tag+model+'matrix.png')
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
        savefig('images/'+image_Folder+'/overfitting/'+file_tag+model+'KNN.png')



Tests_diabetic("KNN")