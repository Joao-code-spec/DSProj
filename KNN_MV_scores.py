from sklearn.metrics import recall_score, confusion_matrix



cnf_mtx_trn = confusion_matrix(trn_y, prd_trn, labels=labels)
tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()