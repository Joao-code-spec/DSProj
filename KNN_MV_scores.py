from sklearn.metrics import recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from KNN_MV import best, trnX, tstX, trnY, tstY

print(best)
print(trnX)
clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1]) #Defining the KNN classifier
clf.fit(trnX, trnY) #Training the classifier
prdY = clf.predict(tstX) #predicted values


#cnf_mtx_trn = confusion_matrix(trn_y, prd_trn, labels=labels)
#tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
#cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
#tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()