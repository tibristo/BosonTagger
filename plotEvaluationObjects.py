import sys
import modelEvaluation as ev
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import operator
import re
job_id = 'features_l_2_10'

def recreateFull(job_id, full_dataset, suffix = 'v2'):
    
    # load the model from the cv model
    with open('evaluationObjects/'+job_id,'r') as p:
        model = pickle.load(p)    
    
    # do this for the full dataset
    file_full = full_dataset
    # check that this file exists
    if not os.path.isfile(file_full):
        print 'could not locate the full dataset'
        return roc_bkg_rej
    print file_full
    X_full, y_full, w_full, efficiencies = joblib.load(file_full, mmap_mode='c')
    print efficiencies
    full_score = model.model.score(X_full, y_full)
    prob_predict_full = model.model.predict_proba(X_full)[:,1]
    fpr_full, tpr_full, thresh_full = roc_curve(y_full, prob_predict_full)
    # need to set the maximum efficiencies for signal and bkg
    m_full = ev.modelEvaluation(fpr_full, tpr_full, thresh_full, model.model, model.params, job_id+'_full', model.taggers, model.Algorithm, full_score, file_full,feature_importances=model.feature_importances, decision_function=model.model.decision_function(X_full))
    m_full.setSigEff(efficiencies[0])
    m_full.setBkgEff(efficiencies[1])
    # get the indices in the full sample
    sig_full_idx = y_full == 1
    bkg_full_idx = y_full == 0
    # set the probabilities and the true indices of the signal and background
    m_full.setProbas(prob_predict_full, sig_full_idx, bkg_full_idx)
    # set the different scores
    y_pred_full = model.predict(X_full)
    m_full.setScores(accuracy=accuracy_score(y_pred_full, y_full), precision=precision_score(y_pred_full, y_full), recall=recall_score(y_pred_full, y_full), f1=f1_score(y_pred_full, y_full))

    # write this into a root file
    m_full.toROOT()

    f_name_full = 'evaluationObjects/'+job_id.replace('.pickle','_')+'_'+suffix+'.pickle'

    try:
        with open(f_name_full,'w') as d2:
            pickle.dump(m_full, d2)
        d2.close()
        print 'pickled ' + f_name_full
    
    except:
        msg = 'unable to dump '+job_id+ '_'+suffix+' object'
        with open(f_name_full,'w') as d2:
            pickle.dump(msg, d2)
        d2.close()
        print 'unable to dump '+job_id+ '_full object:', sys.exc_info()[0]


# set up the job ids
#key = 'features_l_2_10_v2'
key = 'features_l_2_10ID'
full_dataset = 'persist/data_features_nc_2_10_v2_100.pkl'
jobids = [f for f in os.listdir('evaluationObjects/') if f.find(key)!=-1 and f.find('_full.pickle')==-1]
print jobids
#raw_input()
#for j in jobids:
#    recreateFull(j,full_dataset, 'full_v2')

print 'finished creating new full objects'
#sys.exit()
#raw_input()
files = [f for f in os.listdir('evaluationObjects/') if f.find(key)!=-1 and f.endswith('full_v2.pickle')]


data = []
for f in files:
    print 'plotting file: ' + f
    with open('evaluationObjects/'+f,'r') as p:
        model = pickle.load(p)
    # the filename contains the iteration number and the cv number which together form a unique id.
    #  we have a string like paramID_iternum_key_cvnum[_full].pickle
    # first number in the filename is the iternum
    numbers = re.findall(r'\d+', f)
    iter_num = numbers[0]
    # last number is the cv number, unless we have _full_v2.pickle
    if f.find('full_v') != -1:
        cv_num = numbers[-2]
    else:
        cv_num = numbers[-1]
    # now we want to get max_depth parameter
    params = model.model.params
    max_depth = -1
    for k in params.keys():
        if k.find('max_depth') != -1:
            max_depth = int(params[k])

    # get the f1 scores
    f1_train = model.train_f1
    f1_test = model.test_f1
    # get the recall scores
    recall_train = model.train_recall
    recall_test = model.test_recall
    # get the accuracy scores
    accuracy_train = model.train_accuracy
    accuracy_test = model.test_accuracy
    # get the precision scores
    precision_train = model.train_precision
    precision_test = model.test_precision
    # get the bkg rejection power scores
    bkg_rej_train = model.ROC_rej_power_05_train
    bkg_rej_test = model.ROC_rej_power_05
    thedict  = {'test_id':iter_num, 'cv_id': cv_num, 'max_depth':max_depth,'f1_train':f1_train,'f1_test':f1_test,'recall_train':recall_train,'recall_test':recall_test,'precision_train':precision_train,'precision_test':precision_test,'accuracy_train':accuracy_train,'accuracy_test':accuracy_test,'bkg_rej_train':bkg_rej_train,'bkg_rej_test':bkg_rej_test,'filename': f}
    data.append(thedict)

# now create a dataframe out of the data
df = pd.DataFrame(data)
# now we want to be able to sort on a number of criteria and create meaningful stats



def getTaggerScores(files):
    all_taggers_scores = {}
    all_taggers_positions = {}
    scores = {}
    max_score = 0
    max_id = ""
    for f in files:
        print 'plotting file: ' + f
        with open('evaluationObjects/'+f,'r') as p:
            model = pickle.load(p)
        taggers = model.taggers
        for t in taggers:
            if t not in all_taggers_scores.keys():
                all_taggers_scores[t] = []
                all_taggers_positions[t] = []
        feature_importances = 100.0 * (model.feature_importances / model.feature_importances.max())

        #feature_importances = model.feature_importances
        #print feature_importances
        sorted_idx = np.argsort(feature_importances)[::-1]
        for x in range(len(sorted_idx)):
            # add the feature importance score and the position
            all_taggers_scores[taggers[sorted_idx[x]]].append(feature_importances[sorted_idx[x]])
            all_taggers_positions[taggers[sorted_idx[x]]].append(x)
            #print 'feature: ' + taggers[sorted_idx[x]]+' score: ' + str(feature_importances[sorted_idx[x]])

        scores[model.job_id] = model.ROC_rej_power_05
        #print 'sum: ' + str(np.sum(feature_importances))
        if model.score > max_score:
            max_score = model.ROC_rej_power_05
            max_id = model.job_id



    print 'max score: ' + str(max_score)
    print 'id: ' + max_id

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
    print sorted_scores
    print 'press a key to get tagger scores'
    # write to file
    f = open('scores'+key+'.txt','w')
    for s in sorted_scores:
    f.write(s[0] + ': ' + str(s[1]) +'\n')
    f.close()

    # find median of each
    medians = {}
    for t in all_taggers_scores.keys():
        #print t
        #print np.median(all_taggers_scores[t])
        medians[t] = np.median(all_taggers_scores[t])
        #print '***********'

        #print all_taggers_scores


        # sort medians

    sorted_medians = sorted(medians.items(), key=operator.itemgetter(1))
    print sorted_medians
