import sys
import modelEvaluation as ev
import os
import pickle
import numpy as np

files = os.listdir('evaluationObjects/')
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
    model.toROOT()
    scores[model.job_id] = model.score
    if model.score > max_score:
        max_score = model.score
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
