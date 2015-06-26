# A number of tools for doing MVA analysis.
# Includes:
# Cross validation 
# Grid search
# Learning curves
# Probability distributions
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
import os, sys, time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
        

#import cv_fold
def persist_cv_splits(X, y, w, n_cv_iter=5, name='data', prefix='persist/',\
                      suffix="_cv_%03d.pkl", test_size=0.25, random_state=None, scale=True, overwrite=True):
    """Materialize randomized train test splits of a dataset."""
    import os.path
    from root_numpy import array2root
    import numpy.lib.recfunctions as nf
    cv = StratifiedKFold(y,n_cv_iter)
    #cv = StratifiedShuffleSplit(y, n_cv_iter)#KFold(y,n_cv_iter)
    #cv = ShuffleSplit(X.shape[0], n_iter=n_cv_iter,
    #    test_size=test_size, random_state=random_state)
    cv_split_filenames = []

    #scale the data
    
    for i, (train, test) in enumerate(cv):
        cv_split_filename = prefix+name + suffix % i
        cv_split_filename = os.path.abspath(cv_split_filename)
        cv_split_filenames.append(cv_split_filename)

        if os.path.isfile(cv_split_filename) and overwrite == False:
            continue

        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train])
            X_test_scaled = scaler.transform(X[test])
            #fold = cv_fold.cv_fold(X_train_scaled, y[train], w[train], X_test_scaled, y[test], w[test])
            fold = (X_train_scaled, y[train], w[train], X_test_scaled, y[test], w[test])
        else:
            #fold = cv_fold.cv_fold(X[train], y[train], w[train], X[test], y[test], w[test])
            fold = (X[train], y[train], w[train], X[test], y[test], w[test])

        joblib.dump(fold, cv_split_filename)
        rectrain = nf.append_fields(X[train], names='label', data=y[train], usemask=False)#, dtypes=int)#, usemask=False)
        array2root(rectrain, cv_split_filename.replace('.pkl','.root'), 'outputTree','recreate')
        rectest = nf.append_fields(X[test], names='label', data=y[test], usemask=False)#, dtypes=int)#, usemask=False)
        array2root(rectest, cv_split_filename.replace('.pkl','.root'), 'outputTree','recreate')
    
    return cv_split_filenames


def compute_evaluation(cv_split_filename, model, params, job_id = '', taggers = [], weighted=True, algorithm=''):
    """Function executed by a worker to evaluate a model on a CV split"""
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    import modelEvaluation as me

    import numpy as np
    #import cv_fold
    X_train, y_train, w_train, X_validation, y_validation, w_validation = joblib.load(
        cv_split_filename, mmap_mode='c')

    #fold = joblib.load(cv_split_filename)
    #X_train, y_train, w_train, X_validation, y_validation, w_validation = fold.returnDatasets()

    # get the scaler - if we need it!
    #if fold.hasScaler():
    #    scaler = fold.returnScaler()
    
    model.set_params(**params)
    if weighted:
        model.fit(X_train, y_train, w_train)
        #validation_score = model.score(X_validation, y_validation, w_validation)
    else:
        model.fit(X_train, y_train)
    validation_score = model.score(X_validation, y_validation)
    prob_predict_valid = model.predict_proba(X_validation)[:,1]
    fpr, tpr, thresholds = roc_curve(y_validation, prob_predict_valid)
    #roc = RocCurve_SingleSided(y_validation,prob_predict_valid,1,1)
    # find 0.5 tpr
    idx = (np.abs(tpr-0.5)).argmin()
    fpr_05 = fpr[idx]
    rej = 1-fpr_05
    if rej != 1:
        bkgrej = 1/(1-rej)
    else:
        bkgrej = -1

    m = me.modelEvaluation(fpr, tpr, thresholds, model, params, bkgrej, model.feature_importances_, job_id, taggers, algorithm, validation_score, cv_split_filename)
    sig_idx = y_validation == 1
    bkg_idx = y_validation == 0
    m.setProbas(prob_predict_valid, sig_idx, bkg_idx)

    #m.toROOT(sig_idx, bkg_idx, prob_predict_valid)
    f_name = 'evaluationObjects/'+job_id+'.pickle'
    # save the model for later
    import pickle 
    try:
        with open(f_name,'w') as d:
            pickle.dump(m, d)
    except:
        msg = 'unable to dump object'
        with open(f_name,'w') as d:
            pickle.dump(msg, d)
        print 'unable to dump object'

    return bkgrej#validation_score


def grid_search(lb_view, model, cv_split_filenames, param_grid, variables, algo):
    """Launch all grid search evaluation tasks."""
    from sklearn.grid_search import ParameterGrid
    all_tasks = []
    all_parameters = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(all_parameters):
        task_for_params = []
        
        for j, cv_split_filename in enumerate(cv_split_filenames):    
            t = lb_view.apply(
                compute_evaluation, cv_split_filename, model, params, job_id='paramID_'+str(i)+'cvID_'+str(j), taggers=variables, algorithm=algo)
            task_for_params.append(t) 
        
        all_tasks.append(task_for_params)
        
    return all_parameters, all_tasks


def progress(tasks):
    return np.mean([task.ready() for task_group in tasks
                                 for task in task_group])


def find_bests(all_parameters, all_tasks, n_top=5, save=False):
    """Compute the mean score of the completed tasks"""
    mean_scores = []
    param_id = 0
    for param, task_group in zip(all_parameters, all_tasks):
        scores = [t.get() for t in task_group if t.ready()]
        if len(scores) == 0:
            continue
        mean_scores.append((np.mean(scores), param, param_id))
        param_id+=1
    bests = sorted(mean_scores, reverse=True, key=lambda x: x[0])[:n_top]        
    if save:
        f = open('bests/bests.txt','w')
        for b in bests:
            f.write('mean_score: ' + str(b[0]) + ' params: ' + str(b[1]) + ' param id: ' + str(b[2])+'\n')
        f.close()
    return bests


def cross_validation(data, model, params, iterations, variables, ovwrite=True):
    X = data[variables].values
    y = data['label'].values
    w = data['weight'].values

    filenames = persist_cv_splits(X, y, w, n_cv_iter=iterations, name='data', suffix="_cv_%03d.pkl", test_size=0.25, random_state=None, overwrite=ovwrite)
    #all_parameters, all_tasks = grid_search(
     #   lb_view, model, filenames, params)
    return filenames
    #return all_parameters, all_tasks
    
def printProgress(tasks):
    prog = progress(tasks)
    print("Tasks completed: {0}%".format(100 * prog))
    return prog

print os.getcwd()
from sklearn.svm import SVC
from IPython.parallel import Client
from collections import OrderedDict

client = Client()
#with client[:].sync_imports():
#    import cv_fold
lb_view = client.load_balanced_view()
model = AdaBoostClassifier()
base_estimators = [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]
params = OrderedDict([
    ('base_estimator', base_estimators),
    ('n_estimators', np.linspace(5, 20, 10, dtype=np.int)),
    ('learning_rate', np.linspace(0.1, 1, 10))
])

algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedL_ranged_v2_1000_1500_nomw'
trainvars = ['Tau1','EEC_C2_1','EEC_C2_2','EEC_D2_1','TauWTA2','Tau2','EEC_D2_2','TauWTA1']

import pandas as pd
#data = pd.read_csv('/media/win/BoostedBosonFiles/csv/'+algorithm+'_merged.csv')
data = pd.read_csv('csv/'+algorithm+'_merged.csv')

trainvars_iterations = [trainvars]

for t in trainvars_iterations:
    filenames = cross_validation(data, model, params, 2, t, ovwrite=False)
    allparms, alltasks = grid_search(
        lb_view, model, filenames, params, t, algorithm)


    prog = printProgress(alltasks)
    while prog < 1:
        time.sleep(10)
        prog = printProgress(alltasks)
        pprint(find_bests(allparms,alltasks))


    pprint(find_bests(allparms,alltasks,len(allparms), True))
