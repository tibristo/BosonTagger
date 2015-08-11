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
from collections import OrderedDict        

#import cv_fold
def persist_cv_splits(X, y, w, n_cv_iter=5, name='data', prefix='persist/',\
                      suffix="_cv_%03d.pkl", test_size=0.25, random_state=None, scale=True, overwrite=True):
    """Materialize randomized train test splits of a dataset."""
    import os.path
    from root_numpy import array2root
    import numpy.lib.recfunctions as nf
    #cv = StratifiedKFold(y,n_cv_iter)
    cv = StratifiedShuffleSplit(y, n_cv_iter)#KFold(y,n_cv_iter)
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
            # should we scale the signal and background separately??? I think so!
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


def plotSamples(cv_split_filename, taggers, job_id=''):
    import os.path
    import matplotlib.pylab as plt
    
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc

    if not os.path.exists('fold_plots'):
        os.makedirs('fold_plots')
    import numpy as np

    X_train, y_train, w_train, X_validation, y_validation, w_validation = joblib.load(
        cv_split_filename, mmap_mode='c')
    
    # normalise the data first? should have been standardised....
    for i, t in enumerate(taggers):
        plt.hist(X_train[y_train==1][:,i],normed=1, bins=50,color='red',label='signal',alpha=0.5)
        plt.hist(X_train[y_train==0][:,i],normed=1, bins=50,color='blue',label='background',alpha=0.5)
        plt.xlabel(t)
        plt.ylabel('#events')
        plt.title(t)
        plt.legend()
        plt.savefig('fold_plots/'+job_id+t+'.pdf')
        plt.clf()


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

    # get the indices in the validation sample
    sig_idx = y_validation == 1
    bkg_idx = y_validation == 0
    # get the indices in the training sample
    sig_tr_idx = y_train == 1
    bkg_tr_idx = y_train == 0

    # set up the model
    model.set_params(**params)
    # if we are weighting I think that we need to have both the MC weights (or weights we have from our physics knowledge)
    # and the normalisation weights.
    # think that the correct way to do it would be to apply your weights as you would normally. You want to apply the weights so that you can correctly (according to our physics knowledge) represent the data.  I think that after that you can apply the normalisation.  This keeps the shape of the distribution the same, it just gives the bdt a better way of classifying the signal.  I mean I think you could get the same performance more or less if you had enough signal anyway.
    # this means that we then have to adjust the w_train sample so that we have w_train[sig_tr_idx] *= 1/np.count_nonzero(sig_tr_idx) and w_train[bkg_tr_idx] *= 1/np.count_nonzero(bkg_tr_idx)
    # set up array with the scaling factors
    sig_count = (1/float(np.count_nonzero(sig_tr_idx)))
    bkg_count = (1/float(np.count_nonzero(bkg_tr_idx)))
    sig_scaling = sig_tr_idx*sig_count
    bkg_scaling = bkg_tr_idx*bkg_count
    tot_scaling = sig_scaling+bkg_scaling

    #w_train = w_train*tot_scaling

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

    # print the feature importances in descending order
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    for f in range(len(feature_importances)):
        print("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], feature_importances[sorted_idx[f]]))

    m = me.modelEvaluation(fpr, tpr, thresholds, model, params, bkgrej, model.feature_importances_, job_id, taggers, algorithm, validation_score, cv_split_filename)
    m.setProbas(prob_predict_valid, sig_idx, bkg_idx)

    # create the output root file for this.
    m.toROOT()
    roc_bkg_rej = m.getRejPower()

    #m.toROOT(sig_idx, bkg_idx, prob_predict_valid)
    f_name = 'evaluationObjects/'+job_id+'.pickle'
    # save the model for later
    import pickle 
    try:
        with open(f_name,'w') as d:
            pickle.dump(m, d)
        d.close()
    except:
        msg = 'unable to dump object'
        with open(f_name,'w') as d:
            pickle.dump(msg, d)
        d.close()
        print 'unable to dump object'

    return roc_bkg_rej#bkgrej#validation_score


def grid_search(lb_view, model, cv_split_filenames, param_grid, variables, algo, id_tag = 'cv', weighted=True):
    """Launch all grid search evaluation tasks."""
    from sklearn.grid_search import ParameterGrid
    all_tasks = []
    all_parameters = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(all_parameters):
        task_for_params = []
       
        for j, cv_split_filename in enumerate(cv_split_filenames):    
            t = lb_view.apply(
                compute_evaluation, cv_split_filename, model, params, job_id='paramID_'+str(i)+id_tag+'ID_'+str(j), taggers=variables, weighted=weighted,algorithm=algo)
            task_for_params.append(t) 
        
        all_tasks.append(task_for_params)
        
    return all_parameters, all_tasks


def progress(tasks):
    return np.mean([task.ready() for task_group in tasks
                                 for task in task_group])


def find_bests(all_parameters, all_tasks, n_top=5, save=False, bests_tag='cv'):
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
        f = open('bests/bests'+bests_tag+'.txt','w')
        for b in bests:
            f.write('mean_score: ' + str(b[0]) + ' params: ' + str(b[1]) + ' param id: ' + str(b[2])+'\n')
        f.close()
    return bests


def cross_validation(data, model, params, iterations, variables, ovwrite=True, suffix_tag = 'cv', scale=True):
    X = data[variables].values
    y = data['label'].values
    w = data['weight'].values

    filenames = persist_cv_splits(X, y, w, n_cv_iter=iterations, name='data', suffix="_"+suffix_tag+"_%03d.pkl", test_size=0.25, scale=scale,random_state=None, overwrite=ovwrite)
    #all_parameters, all_tasks = grid_search(
     #   lb_view, model, filenames, params)
    return filenames
    #return all_parameters, all_tasks
    
def printProgress(tasks):
    prog = progress(tasks)
    print("Tasks completed: {0}%".format(100 * prog))
    return prog




def runTest(cv_split_filename, model, trainvars, algo):
    base_estimators = [DecisionTreeClassifier(max_depth=5)]
    params = OrderedDict([
            ('base_estimator', base_estimators),
            ('n_estimators', [50]),
            ('learning_rate', [1.0])
            ])

    from sklearn.grid_search import ParameterGrid
    all_parameters = list(ParameterGrid(params))
    
    for i, params in enumerate(all_parameters):
        compute_evaluation(cv_split_filename, model, params, job_id = 'test', taggers = trainvars, weighted=True, algorithm=algo)
        plotSamples(cv_split_filename, trainvars, 'test')
        return
        



print os.getcwd()
from sklearn.svm import SVC

#    import cv_fold

model = AdaBoostClassifier()

base_estimators = [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]
params = OrderedDict([
    ('base_estimator', base_estimators),
    ('n_estimators', np.linspace(20, 40, 10, dtype=np.int)),
    ('learning_rate', np.linspace(0.1, 1, 10))
])




#{'n_estimators': 20, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
#            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            random_state=None, splitter='best'), 'learning_rate': 0.70000000000000007} param id: 269

#algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedL_ranged_v2_1000_1500_mw'
algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedM_v2_500_1000_mw'
#trainvars = ['Tau1','EEC_C2_1','EEC_C2_2','EEC_D2_1','TauWTA2','Tau2','EEC_D2_2','TauWTA1']

#trainvars = ['Aplanarity','ThrustMin','Tau1','Sphericity','FoxWolfram20','Tau21','ThrustMaj','EEC_C2_1','EEC_C2_2','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','Angularity','ZCUT12','Tau2','EEC_D2_2','TauWTA1','PlanarFlow']
trainvars = ['Aplanarity','ThrustMin','Sphericity','Tau21','ThrustMaj','EEC_C2_1','EEC_C2_2','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','ZCUT12','Tau2','EEC_D2_2','PlanarFlow']
#trainvars = ['EEC_C2_1','EEC_C2_2','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','ZCUT12','Tau2','EEC_D2_2']

import pandas as pd
#data = pd.read_csv('/media/win/BoostedBosonFiles/csv/'+algorithm+'_merged.csv')
data = pd.read_csv('csv/'+algorithm+'_merged.csv')

test_case = 'features_5_10_'
#test_case = 'cv'

trainvars_iterations = [trainvars]


#runTest('persist/data_features_5_10__001.pkl', model, trainvars, algorithm)
#sys.exit(0)
#raw_input()

from IPython.parallel import Client


client = Client()
#with client[:].sync_imports():
lb_view = client.load_balanced_view()

for t in trainvars_iterations:
    filenames = cross_validation(data, model, params, 3, t, ovwrite=True, suffix_tag=test_case, scale=True)
    allparms, alltasks = grid_search(
        lb_view, model, filenames, params, t, algorithm, id_tag=test_case, weighted=True)


    prog = printProgress(alltasks)
    while prog < 1:
        time.sleep(10)
        prog = printProgress(alltasks)
        pprint(find_bests(allparms,alltasks))


    pprint(find_bests(allparms,alltasks,len(allparms), True))
