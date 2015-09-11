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
def persist_cv_splits(X, y, w, variables, n_cv_iter=5, name='data', prefix='persist/',\
                      suffix="_cv_%03d.pkl", test_size=0.25, random_state=None, scale=True, overwrite=True, overwrite_full=True,signal_eff=1.0, bkg_eff=1.0):
    """Materialize randomized train test splits of a dataset."""
    import os.path
    from root_numpy import array2root
    import numpy.lib.recfunctions as nf
    #cv = StratifiedKFold(y,n_cv_iter)
    cv = StratifiedShuffleSplit(y, n_cv_iter)#KFold(y,n_cv_iter)
    #cv = ShuffleSplit(X.shape[0], n_iter=n_cv_iter,
    #    test_size=test_size, random_state=random_state)
    cv_split_filenames = []

    # persist the original files as well.
    # first check if the file exists already
    full_fname = os.path.abspath(prefix+name+suffix % 100)
    print full_fname
    if overwrite_full or not os.path.isfile(full_fname):
        full_set = (X,y,w,[signal_eff,bkg_eff])
        joblib.dump(full_set,full_fname)

    #return
        
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
        cv_split_train = os.path.abspath(prefix+name+'train'+suffix%i)
        cv_split_test = os.path.abspath(prefix+name+'test'+suffix%i)

        # have to do this annoying .copy() to be able to add the dtype.names to any
        # arrays that come from a slice.
        XX = X.copy().view(dtype=[(n, np.float64) for n in variables]).reshape(len(X))

        # we want to add the weights as well        
        # add the label to the array
        rectrain = nf.append_fields(XX[train], names=['label','weight'], data=[y[train],w[train]], usemask=False)
        array2root(rectrain, cv_split_train.replace('.pkl','.root'), 'outputTree','recreate')
        rectest = nf.append_fields(XX[test], names=['label','weight'], data=[y[test],w[test]], usemask=False)
        array2root(rectest, cv_split_test.replace('.pkl','.root'), 'outputTree','recreate')
    
    return cv_split_filenames


def plotSamples(cv_split_filename, full_dataset, taggers, first_tagger = False):
    '''
    Method for plotting the variables in the cv folds.  It also records stats of the cv folds including the number
    of events and the mean and std of the different taggers.

    cv_split_filename -- cv fold filename
    full_dataset --- the full dataset from which the cv folds were created
    taggers --- the variables used in the dataset for training
    first_tagger --- this is used to create a stats file for all of the cv folds combined. If this is true then it will recreate the file
    and it will also write the full dataset stats at the top of the file.
    '''
    import os.path
    import matplotlib.pylab as plt
    
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc

    if not os.path.exists('fold_plots'):
        os.makedirs('fold_plots')
    import numpy as np

    # file mode for tagger_stats and event_count
    file_mode = 'a'
    if first_tagger:
        file_mode = 'w'
    
    # load the cross validation folds
    X_train, y_train, w_train, X_validation, y_validation, w_validation = joblib.load(
        'persist/'+cv_split_filename, mmap_mode='c')
    # load the full dataset
    X,y,w,eff = joblib.load(full_dataset,mmap_mode='c')
    # only want the file not the folder
    stats_fname = os.path.basename(cv_split_filename)
    stats_fname = stats_fname.replace('pkl','txt')
    # get the cross validation fold number from the file name.  The file will always end in _XYZ.pkl, where XYZ are integers.
    cv_num = stats_fname.split('_')[-1].replace('.txt','')
    # create a stats file with the std, mean of each variable, number of entries
    if not os.path.exists('fold_stats'):
        os.makedirs('fold_stats')
    stats = open('fold_stats/'+stats_fname,'w')
    
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Sample','Signal','Background','Total')+'\n')
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Full',str(X[y==1].shape[0]), str(X[y==0].shape[0]),str(X.shape[0]))+'\n')
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Training',str(X_train[y_train==1].shape[0]), str(X_train[y_train==0].shape[0]),str(X_train.shape[0]))+'\n')
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Valid',str(X_validation[y_validation==1].shape[0]), str(X_validation[y_validation==0].shape[0]),str(X_validation.shape[0]))+'\n\n')
    
    event_count = open('fold_stats/event_counts.txt',file_mode)
    if first_tagger:
        event_count.write('{0:15}  {1:10} {2:14}{3:10}'.format('Full',str(X[y==1].shape[0]), str(X[y==0].shape[0]),str(X.shape[0]))+'\n')
    event_count.write('{0:15}  {1:10} {2:14}{3:10}'.format('Train cv '+cv_num,str(X_train[y_train==1].shape[0]), str(X_train[y_train==0].shape[0]),str(X_train.shape[0]))+'\n')
    event_count.write('{0:15}  {1:10} {2:14}{3:10}'.format('Valid cv '+cv_num,str(X_validation[y_validation==1].shape[0]), str(X_validation[y_validation==0].shape[0]),str(X_validation.shape[0]))+'\n')
    event_count.close()
    
    stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Variable','Mean','Std','Mean Sig','Std Sig','Mean Bkg','Std Bkg')+'\n\n')

    
    for i,t in enumerate(taggers):
        tagger_stats = open('fold_stats/'+t+'.txt',file_mode)
        stats.write(t+'\n')
        # training
        mean_tr = "{0:.4f}".format(float(np.mean(X_train[:,i])))
        mean_signal_tr = '{0:.4f}'.format(float(np.mean(X_train[y_train==1][:,i])))
        mean_bkg_tr = '{0:.4f}'.format(float(np.mean(X_train[y_train==0][:,i])))
        std_tr = '{0:.4f}'.format(float(np.std(X_train[:,i])))
        std_signal_tr = '{0:.4f}'.format(float(np.std(X_train[y_train==1][:,i])))
        std_bkg_tr = '{0:.4f}'.format(float(np.std(X_train[y_train==0][:,i])))
        # validation
        mean_val = '{0:.4f}'.format(float(np.mean(X_validation[:,i])))
        mean_signal_val = '{0:.4f}'.format(float(np.mean(X_validation[y_validation==1][:,i])))
        mean_bkg_val = '{0:.4f}'.format(float(np.mean(X_validation[y_validation==0][:,i])))
        std_val = '{0:.4f}'.format(float(np.std(X_validation[:,i])))
        std_signal_val = '{0:.4f}'.format(float(np.std(X_validation[y_validation==1][:,i])))
        std_bkg_val = '{0:.4f}'.format(float(np.std(X_validation[y_validation==0][:,i])))
        # full dataset
        mean = '{0:.4f}'.format(float(np.mean(X[:,i])))
        mean_signal = '{0:.4f}'.format(float(np.mean(X[y==1][:,i])))
        mean_bkg = '{0:.4f}'.format(float(np.mean(X[y==0][:,i])))
        std = '{0:.4f}'.format(float(np.std(X[:,i])))
        std_signal = '{0:.4f}'.format(float(np.std(X[y==1][:,i])))
        std_bkg = '{0:.4f}'.format(float(np.std(X[y==0][:,i])))

        result = '{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Full',str(mean),str(std),str(mean_signal),str(std_signal),str(mean_bkg),str(std_bkg))
        result_tr = '{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Train',str(mean_tr),str(std_tr),str(mean_signal_tr),str(std_signal_tr),str(mean_bkg_tr),str(std_bkg_tr))
        result_val = '{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Valid',str(mean_val),str(std_val),str(mean_signal_val),str(std_signal_val),str(mean_bkg_val),str(std_bkg_val))
        
        stats.write(result+'\n'+result_tr+'\n'+result_val+'\n\n')
        if first_tagger:
            tagger_stats.write(result+'\n')
        tagger_stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Train cv '+cv_num,str(mean_tr),str(std_tr),str(mean_signal_tr),str(std_signal_tr),str(mean_bkg_tr),str(std_bkg_tr))+'\n')
        tagger_stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Valid cv '+cv_num,str(mean_val),str(std_val),str(mean_signal_val),str(std_signal_val),str(mean_bkg_val),str(std_bkg_val))+'\n')
        tagger_stats.close()

    stats.close()
    # normalise the data first? should have been standardised....
    # filename for plots:
    plt_fname = stats_fname.replace('.txt','')
    for i, t in enumerate(taggers):
        plt.hist(X_train[y_train==1][:,i],normed=1, bins=50,color='red',label='signal',alpha=0.5)
        plt.hist(X_train[y_train==0][:,i],normed=1, bins=50,color='blue',label='background',alpha=0.5)
        plt.xlabel(t)
        plt.ylabel('#events')
        plt.title(t)
        plt.legend()
        plt.savefig('fold_plots/'+plt_fname+'_'+t+'.pdf')
        plt.clf()


def compute_evaluation(cv_split_filename, model, params, job_id = '', taggers = [], weighted=True, algorithm='', full_dataset=''):
    """Function executed by a worker to evaluate a model on a CV split"""
    import os
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
    import modelEvaluation as me
    import sys

    import numpy as np
    #import cv_fold
    print cv_split_filename
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

    # we want to do this for both the validation sample AND the full sample so that we
    # can compare it with the cut-based tagger.
    validation_score = model.score(X_validation, y_validation)
    prob_predict_valid = model.predict_proba(X_validation)[:,1]
    fpr, tpr, thresholds = roc_curve(y_validation, prob_predict_valid)
    

    m = me.modelEvaluation(fpr, tpr, thresholds, model, params, job_id, taggers, algorithm, validation_score, cv_split_filename, feature_importances=model.feature_importances_, decision_function=model.decision_function(X_validation))
    m.setProbas(prob_predict_valid, sig_idx, bkg_idx)
    # set all of the scores
    y_val_pred = model.predict(X_validation)
    m.setScores('test',accuracy=accuracy_score(y_val_pred, y_validation), precision=precision_score(y_val_pred, y_validation), recall=recall_score(y_val_pred, y_validation), f1=f1_score(y_val_pred, y_validation))
    y_train_pred = model.predict(X_train)
    m.setScores('train',accuracy=accuracy_score(y_train_pred, y_train), precision=precision_score(y_train_pred, y_train), recall=recall_score(y_train_pred, y_train), f1=f1_score(y_train_pred, y_train)
)
    # create the output root file for this.
    m.toROOT()
    # score to return
    roc_bkg_rej = m.getRejPower()
    # calculate the training score as well
    prob_predict_train = model.predict_proba(X_train)[:,1]
    bkg_rej_train = m.calculateBkgRej(prob_predict_train, sig_tr_idx, bkg_tr_idx)
    m.setTrainRejection(bkg_rej_train)

    # save the model for later
    f_name = 'evaluationObjects/'+job_id+'.pickle'
    import pickle 
    try:
        with open(f_name,'w') as d:
            pickle.dump(m, d)
        d.close()
    except:
        msg = 'unable to dump '+job_id+ ' object'
        with open(f_name,'w') as d:
            pickle.dump(msg, d)
        d.close()

    
    # do this for the full dataset
    # try reading in the memmap file
    # the easiest way to find the name of the file is to take the cv_split_filename
    # and then search backwards to find an underscore. The number between this underscore
    # and the file extension, pkl, should be 100 for this file. It is written this
    # way in the persist_cv_ method in this file.
    if full_dataset == '':
        underscore_idx = cv_split_filename.rfind('_')
        if underscore_idx == -1:
            print 'could not locate the full dataset'
            # return the rejection on the validation set anyway
            return roc_bkg_rej
        file_full = cv_split_filename[:underscore_idx+1]+'100.pkl'
    else:
        file_full = full_dataset
    # check that this file exists
    if not os.path.isfile(file_full):
        print 'could not locate the full dataset'
        return roc_bkg_rej
    print file_full
    X_full, y_full, w_full, efficiencies = joblib.load(file_full, mmap_mode='c')
    print efficiencies
    full_score = model.score(X_full, y_full)
    prob_predict_full = model.predict_proba(X_full)[:,1]
    fpr_full, tpr_full, thresh_full = roc_curve(y_full, prob_predict_full)
    # need to set the maximum efficiencies for signal and bkg
    m_full = me.modelEvaluation(fpr_full, tpr_full, thresh_full, model, params, job_id+'_full', taggers, algorithm, full_score, file_full,feature_importances=model.feature_importances_, decision_function=model.decision_function(X_full))
    m_full.setSigEff(efficiencies[0])
    m_full.setBkgEff(efficiencies[1])
    # get the indices in the full sample
    sig_full_idx = y_full == 1
    bkg_full_idx = y_full == 0
    # set the probabilities and the true indices of the signal and background
    m_full.setProbas(prob_predict_full, sig_full_idx, bkg_full_idx)
    # set the different scoresx
    y_pred_full = model.predict(X_full)
    m_full.setScores('test',accuracy=accuracy_score(y_pred_full, y_full), precision=precision_score(y_pred_full, y_full), recall=recall_score(y_pred_full, y_full), f1=f1_score(y_pred_full, y_full))
    m_full.setScores('train',accuracy=accuracy_score(y_train_pred, y_train), precision=precision_score(y_train_pred, y_train), recall=recall_score(y_train_pred, y_train), f1=f1_score(y_train_pred, y_train))
    # write this into a root file
    m_full.toROOT()
    # save the train score
    m_full.setTrainRejection(bkg_rej_train)

    f_name_full = 'evaluationObjects/'+job_id+'_full.pickle'

    # save the model to use later.
    try:
        with open(f_name_full,'w') as d2:
            pickle.dump(m_full, d2)
        d2.close()
    
    except:
        msg = 'unable to dump '+job_id+ '_full object'
        with open(f_name_full,'w') as d2:
            pickle.dump(msg, d2)
        d2.close()
        print 'unable to dump '+job_id+ '_full object:', sys.exc_info()[0]
    
    return roc_bkg_rej#bkgrej#validation_score


def grid_search(lb_view, model, cv_split_filenames, param_grid, variables, algo, id_tag = 'cv', weighted=True, full_dataset=''):
    """Launch all grid search evaluation tasks."""
    from sklearn.grid_search import ParameterGrid
    all_tasks = []
    all_parameters = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(all_parameters):
        task_for_params = []
       
        for j, cv_split_filename in enumerate(cv_split_filenames):    
            t = lb_view.apply(
                compute_evaluation, cv_split_filename, model, params, job_id='paramID_'+str(i)+id_tag+'ID_'+str(j), taggers=variables, weighted=weighted,algorithm=algo, full_dataset=full_dataset)
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


def cross_validation(data, model, params, iterations, variables, ovwrite=True, ovwrite_full=True,suffix_tag = 'cv', scale=True):
    X = data[variables].values
    y = data['label'].values
    w = data['weight'].values

    # find the efficiency for both signal and bkg
    signal_eff = data.loc[data['label']==1]['eff'].values[0]
    bkg_eff = data.loc[data['label']==0]['eff'].values[0]

    # create the cross validation splits and write them to disk
    filenames = persist_cv_splits(X, y, w, variables, n_cv_iter=iterations, name='data', suffix="_"+suffix_tag+"_%03d.pkl", test_size=0.25, scale=scale,random_state=None, overwrite=ovwrite, overwrite_full=ovwrite_full, signal_eff=signal_eff, bkg_eff=bkg_eff)

    return filenames
    
def printProgress(tasks):
    prog = progress(tasks)
    print("Tasks completed: {0}%".format(100 * prog))
    return prog




def runTest(cv_split_filename, model, trainvars, algo, full_dataset=''):
    base_estimators = [DecisionTreeClassifier(max_depth=5)]
    params = OrderedDict([
            ('base_estimator', base_estimators),
            ('n_estimators', [50]),
            ('learning_rate', [1.0])
            ])

    from sklearn.grid_search import ParameterGrid
    all_parameters = list(ParameterGrid(params))
    
    for i, params in enumerate(all_parameters):
        compute_evaluation(cv_split_filename, model, params, job_id = 'test', taggers = trainvars, weighted=True, algorithm=algo, full_dataset=full_dataset)
        plotSamples(cv_split_filename, trainvars)
        return
        
print os.getcwd()
from sklearn.svm import SVC

#    import cv_fold

model = AdaBoostClassifier()

base_estimators = [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]
params = OrderedDict([
    ('base_estimator', base_estimators),
    ('n_estimators', np.linspace(20, 100, 10, dtype=np.int)),
    ('learning_rate', np.linspace(0.1, 1, 10))
])

#{'n_estimators': 20, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
#            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            random_state=None, splitter='best'), 'learning_rate': 0.70000000000000007} param id: 269

#algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedL_ranged_v2_1000_1500_mw'
algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedM_loose_v2_200_1000_mw'
#algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedM_notcleaned_v2_200_1000_mw'

#trainvars = ['Aplanarity','ThrustMin','Tau1','Sphericity','FoxWolfram20','Tau21','ThrustMaj','EEC_C2_1','EEC_C2_2','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','Angularity','ZCUT12','Tau2','EEC_D2_2','TauWTA1','PlanarFlow']
#trainvars = ['Aplanarity','ThrustMin','Sphericity','Tau21','ThrustMaj','EEC_C2_1','EEC_C2_2','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','ZCUT12','Tau2','EEC_D2_2','PlanarFlow']# features v1 
trainvars = ['EEC_C2_1','EEC_C2_2','SPLIT12','Aplanarity','EEC_D2_1','TauWTA2'] # features_l_2_10_v2

key = 'features_l_2_10_v3'
#test_case = 'features_nc_2_10_v1'
#test_case = 'cv'
#test_case = 'test_tgraph'

trainvars_iterations = [trainvars]
#full_dataset = 'persist/data_features_nc_2_10_v3_100.pkl'
full_dataset = 'persist/data_features_l_2_10_v3_100.pkl'

plotCV = True
if plotCV:
    filenames = [f for f in os.listdir('persist/') if f.find(key) != -1 and f.find('100.pkl')==-1 and f.endswith('pkl')]
    for i,f in enumerate(filenames):
        plotSamples(f,full_dataset,trainvars, i == 0)
    # create the combined stats file for all taggers and all cv splits
    combined_stats = open('fold_stats/combined_stats_'+key+'.txt','w')
    combined_stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Sample','Signal','Background','Total')+'\n')
    with open('fold_stats/event_counts.txt') as infile:
        combined_stats.write(infile.read())
    combined_stats.write('\n')
    combined_stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10}'.format('Variable','Mean','Std','Mean Sig','Std Sig','Mean Bkg','Std Bkg')+'\n\n')
    for t in trainvars:
        combined_stats.write(t+'\n')
        with open('fold_stats/'+t+'.txt') as tfile:
            combined_stats.write(tfile.read())
        combined_stats.write('\n')
    sys.exit()


import pandas as pd
#data = pd.read_csv('/media/win/BoostedBosonFiles/csv/'+algorithm+'_merged.csv')
data = pd.read_csv('csv/'+algorithm+'_merged.csv')

#runTest('persist/data_features_5_10__001.pkl', model, trainvars, algorithm)
#runTest('persist/data_features_l_2_10_v3_001.pkl', model, trainvars, algorithm, full_dataset)

#sys.exit(0)
#raw_input()

# just create the folds
createFoldsOnly = False
if createFoldsOnly:
    # we need to add some extra variables that might not get used for training, but we want in there anyway!
    filenames = cross_validation(data, model, params, 3, trainvars, ovwrite=True, ovwrite_full=True,suffix_tag=key, scale=False)
    sys.exit()


    
from IPython.parallel import Client

client = Client()
#with client[:].sync_imports():
lb_view = client.load_balanced_view()


    
for t in trainvars_iterations:
    filenames = cross_validation(data, model, params, 3, t, ovwrite=True, ovwrite_full=False, suffix_tag=key, scale=False)
    allparms, alltasks = grid_search(
        lb_view, model, filenames, params, t, algorithm, id_tag=key, weighted=True, full_dataset=full_dataset)


    prog = printProgress(alltasks)
    while prog < 1:
        time.sleep(10)
        prog = printProgress(alltasks)
        pprint(find_bests(allparms,alltasks))


    pprint(find_bests(allparms,alltasks,len(allparms), True, key))
