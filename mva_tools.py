# A number of tools for doing MVA analysis.
# Includes:
# Cross validation 
# Grid search
# Learning curves
# Probability distributions
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import ShuffleSplit, StratifiedKFold
import os, sys, time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint

#import cv_fold
def persist_cv_splits(X, y, w, n_cv_iter=5, name='data',\
                      suffix="_cv_%03d.pkl", test_size=0.25, random_state=None, scale=True):
    """Materialize randomized train test splits of a dataset."""
    cv = StratifiedKFold(y,n_cv_iter)
    #cv = ShuffleSplit(X.shape[0], n_iter=n_cv_iter,
    #    test_size=test_size, random_state=random_state)
    cv_split_filenames = []

    #scale the data
    
    for i, (train, test) in enumerate(cv):
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train])
            X_test_scaled = scaler.transform(X[test])
            #fold = cv_fold.cv_fold(X_train_scaled, y[train], w[train], X_test_scaled, y[test], w[test])
            fold = (X_train_scaled, y[train], w[train], X_test_scaled, y[test], w[test])
        else:
            #fold = cv_fold.cv_fold(X[train], y[train], w[train], X[test], y[test], w[test])
            fold = (X[train], y[train], w[train], X[test], y[test], w[test])
        cv_split_filename = name + suffix % i
        cv_split_filename = os.path.abspath(cv_split_filename)
        joblib.dump(fold, cv_split_filename)
        cv_split_filenames.append(cv_split_filename)
    
    return cv_split_filenames


def compute_evaluation(cv_split_filename, model, params, weighted=True):
    """Function executed by a worker to evaluate a model on a CV split"""
    from sklearn.externals import joblib
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
        validation_score = model.score(X_validation, y_validation, w_validation)
    else:
        model.fit(X_train, y_train)
        validation_score = model.score(X_validation, y_validation)
        
    return validation_score


def grid_search(lb_view, model, cv_split_filenames, param_grid):
    """Launch all grid search evaluation tasks."""
    from sklearn.grid_search import ParameterGrid
    all_tasks = []
    all_parameters = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(all_parameters):
        task_for_params = []
        
        for j, cv_split_filename in enumerate(cv_split_filenames):    
            t = lb_view.apply(
                compute_evaluation, cv_split_filename, model, params)
            task_for_params.append(t) 
        
        all_tasks.append(task_for_params)
        
    return all_parameters, all_tasks


def progress(tasks):
    return np.mean([task.ready() for task_group in tasks
                                 for task in task_group])


def find_bests(all_parameters, all_tasks, n_top=5):
    """Compute the mean score of the completed tasks"""
    mean_scores = []
    
    for param, task_group in zip(all_parameters, all_tasks):
        scores = [t.get() for t in task_group if t.ready()]
        if len(scores) == 0:
            continue
        mean_scores.append((np.mean(scores), param))
                   
    return sorted(mean_scores, reverse=True, key=lambda x: x[0])[:n_top]


def cross_validation(data, model, params, iterations, variables):
    X = data[variables].values
    y = data['label'].values
    w = data['weight'].values

    filenames = persist_cv_splits(X, y, w, n_cv_iter=iterations, name='data', suffix="_cv_%03d.pkl", test_size=0.25, random_state=None)
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
with client[:].sync_imports():
    import cv_fold
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
data = pd.read_csv('/media/win/BoostedBosonFiles/csv/'+algorithm+'_merged.csv')


filenames = cross_validation(data, model, params, 2, trainvars)
allparms, alltasks = grid_search(
    lb_view, model, filenames, params)


#allparms, alltasks = cross_validation(data, model, params, 2, trainvars)

prog = printProgress(alltasks)
while prog < 1:
    time.sleep(10)
    prog = printProgress(alltasks)
    pprint(find_bests(allparms,alltasks))


pprint(find_bests(allparms,alltasks))
