# coding: utf-8
import modelEvaluation as me
import numpy as np
import sys

in_file = sys.argv[1]
if len(sys.argv) > 2:
    out_file = sys.arv[2]
else:
    out_file = in_file.replace('.pbz2','_featureImportances.txt')

from sklearn.externals import joblib
import bz2
import pickle
with bz2.BZ2File(in_file,'r') as p:
    model = pickle.load(p)
    
model.feature_importances
taggers = model.taggers
features = model.feature_importances
sorted_idx = np.argsort(features)[::-1]
outf = open(out_file,'w')
for f in range(len(features)):
    print("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], features[sorted_idx[f]]))
    outf.write("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], features[sorted_idx[f]])+ '\n')
outf.close()
#get_ipython().magic(u'save featureImportances')
