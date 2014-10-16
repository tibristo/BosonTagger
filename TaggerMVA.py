import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc
import itertools

print 'Loading training data ...'

Algorithm = 'CamKt12LCTopoSplitFilteredMu67SmallR0YCut9'

data_train = pd.read_csv(Algorithm+'merged.csv')
np.random.seed(42)
r =np.random.rand(data_train.shape[0])
    	#Algorithm = 'AKT10LCTRIM530'

plt.figure(1)
  #  Y_train = data_train['label'][r<0.9]
#    W_train = data_train['weight'][r<0.9]
   # Y_valid = data_train['label'][r>=0.9]
#    W_valid = data_train['weight'][r>=0.9]
#    data_train.drop('AKT10LCTRIM530_MassDropSplit', axis=1, inplace=True)
varcombinations = itertools.combinations(data_train.columns.values[1:-1],2)
fac = lambda n: 1 if n < 2 else n * fac(n - 1)
combos = lambda n, k: fac(n) / fac(k) / fac(n - k)

colors = plt.get_cmap('jet')(np.linspace(0, 1.0,combos(len(data_train.columns.values[1:-1]),2) ))
for varset,color in zip(varcombinations, colors):
    print list(varset)
    X_train = data_train[list(varset)].values[r<0.5]
    X_valid = data_train[list(varset)].values[r>=0.5]
   # X_train = data_train.values[:,1:3][r<0.5]
   # X_valid = data_train.values[:,1:3][r>=0.5]
    Y_train = data_train['label'].values[r<0.5]
    Y_valid = data_train['label'].values[r>=0.5]

    print X_train
	  #gbc = Pipeline([("scale", StandardScaler()), ("gbc",GBC(n_estimators=1,verbose=1, max_depth=10,min_samples_leaf=50))])
	  #	  gbc = GBC(n_estimators=20,verbose=1, max_depth=10,min_samples_leaf=50)
#gbc = GaussianNB()
    dt = DC(max_depth=3,min_samples_leaf=0.05*len(X_train))
    abc = ABC(dt,algorithm='SAMME',
                 n_estimators=800,
                 learning_rate=0.5)
    print 'Training classifier with all the data..'
    abc.fit(X_train, Y_train)
#    sample_weight=W_train.values 
    print 'Done.. Applying to validation sample and drawing ROC' 
    prob_predict_valid = abc.predict_proba(X_valid)[:,1]
    Y_score = abc.decision_function(X_valid)
    fpr, tpr, _ = roc_curve(Y_valid, prob_predict_valid)
    #    W_valid.values
    labelstring = ' And '.join(var.replace('_','') for var in varset)
    print labelstring
    plt.plot(tpr, (1-fpr), label=labelstring, color=color)
    #fpr, tpr, _ = roc_curve(Y_valid,Y_score )
     
#    plt.figure(2)       
#    plt.hist(abc.decision_function(X_valid[Y_valid==1.]).ravel(),
#             color='r', alpha=0.5, range=(-0.6,0.0), bins=20)
#    plt.hist(abc.decision_function(X_valid[Y_valid==0.]).ravel(),
#             color='b', alpha=0.5, range=(-0.6,0.0), bins=20)
#    plt.xlabel("BDT output")
#    plt.savefig(labelstring+'bdtout.pdf')        
#  
	  #	  labelstring = ' and '.join(var.replace(Algorithm,'') for var in varset)
plt.figure(1)   
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('1- Background Efficiency')
plt.xlabel('Signal Efficiency')
plt.title(Algorithm+' ROC Curve')
plt.legend(loc="lower left",prop={'size':6})
#plt.show()
plt.savefig(Algorithm+'rocmva.pdf')
