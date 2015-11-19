# coding: utf-8
import modelEvaluation as me
import numpy as np

from sklearn.externals import joblib
import bz2
import pickle
with bz2.BZ2File('evaluationObjects/test.pbz2','r') as p:
    model = pickle.load(p)
    
model.feature_importances
taggers = model.taggers
features = model.feature_importances
sorted_idx = np.argsort(features)[::-1]
outf = open('test_features.txt','w')
for f in range(len(features)):
    print("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], features[sorted_idx[f]]))
    outf.write("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], features[sorted_idx[f]])+ '\n')
outf.close()
#get_ipython().magic(u'save featureImportances')
