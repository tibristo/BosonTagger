import sys
import modelEvaluation as ev
import os
import pickle
import numpy as np


job_id = 'features'


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
    taggers = model.taggers
    for t in taggers:
        if t not in all_taggers_scores.keys():
            all_taggers_scores[t] = []
            all_taggers_positions[t] = []
    feature_importances = model.feature_importances
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
    #if np.min(model.discriminant) < 0:
    #    print model.discriminant
    #    raw_input()
    
    #raw_input()
    #sys.exit()
    #except:
     #   print 'error saving ROOT file'



print 'max score: ' + str(max_score)
print 'id: ' + max_id




# find median of each
for t in all_taggers_scores.keys():
    print t
    print np.median(all_taggers_scores[t])
    print '***********'
