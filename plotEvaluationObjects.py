import sys
import modelEvaluation as ev
import os
import pickle
import numpy as np


job_id = 'features_5_10'


files = [f for f in os.listdir('evaluationObjects/') if f.find(job_id)!=-1]


all_taggers_scores = {}
all_taggers_positions = {}
scores = {}
max_score = 0
max_id = ""
for f in files:
    print 'plotting file: ' + f
    with open('evaluationObjects/'+f,'r') as p:
        model = pickle.load(p)
    
    #try:
    #print 'saving ROOT file'
    #print model.sig_idx
    #print model.bkg_idx
    #print model.tpr
    #model.toROOT()
    #print model.feature_importances
    #print model.taggers
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


# find median of each
medians = {}
for t in all_taggers_scores.keys():
    print t
    print np.median(all_taggers_scores[t])
    medians[t] = np.median(all_taggers_scores[t])
    print '***********'

#print all_taggers_scores


# sort medians
import operator
sorted_medians = sorted(medians.items(), key=operator.itemgetter(1))
print sorted_medians
