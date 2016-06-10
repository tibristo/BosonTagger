# coding: utf-8
import modelEvaluation as me
import numpy as np
import sys
import matplotlib.pylab as plt
plt.rc('text',usetex=True)
import mva_tools as tools

in_file = sys.argv[1]
if len(sys.argv) > 2:
    out_file = sys.argv[2]
else:
    out_file = in_file.replace('.pbz2','_featureImportances.txt')

from sklearn.externals import joblib
import bz2
import pickle
with bz2.BZ2File(in_file,'r') as p:
    model = pickle.load(p)
    
model.feature_importances
taggers = model.taggers
features = model.model.feature_importances_
sorted_idx = np.argsort(features)[::-1]
taggers_sorted = [tools.label_dict[taggers[sorted_idx[f]]] for f in range(len(features))]
outf = open(out_file,'w')
std = np.std([tree.feature_importances_ for tree in model.model.estimators_],axis=0)
max_feature = features[sorted_idx[0]]
for f in range(len(features)):
    print("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], 100*features[sorted_idx[f]]/max_feature))
    outf.write("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], 100*features[sorted_idx[f]]/max_feature)+ '\n')
outf.close()
#get_ipython().magic(u'save featureImportances')


plt.title('Feature Importances')
plt.bar(range(len(features)), #X_train.shape[1]
        features[sorted_idx],
        color='r',
        align='center')#, yerr=std[sorted_idx])
plt.xticks(range(len(features)),
           taggers_sorted, rotation=90)
plt.xlim([-1, len(features)])
plt.tight_layout()
#plt.show()
plt.savefig('testfeatures.pdf')
