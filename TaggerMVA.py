import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc
import itertools

def main():
        print 'Loading training data ...'
        data_train = pd.read_csv('merged.csv')
	r =np.random.rand(data_train.shape[0])
	Algorithm = 'AKT10LCTRIM530'

	plt.figure()
	Y_train = data_train['label'][r<0.9]
	W_train = data_train['weight'][r<0.9]
	Y_valid = data_train['label'][r>=0.9]
	W_valid = data_train['weight'][r>=0.9]
	data_train.drop('AKT10LCTRIM530_MassDropSplit', axis=1, inplace=True)
	for varset in itertools.combinations(data_train.columns.values[4:-2],2):
	  print list(varset)
	  X_train = data_train[list(varset)][r<0.9]
	  X_valid = data_train[list(varset)][r>=0.9]

	  #gbc = Pipeline([("scale", StandardScaler()), ("gbc",GBC(n_estimators=1,verbose=1, max_depth=10,min_samples_leaf=50))])
	  #	  gbc = GBC(n_estimators=20,verbose=1, max_depth=10,min_samples_leaf=50)
		  #gbc = GaussianNB()
	  abc = ABC(n_estimators=20)
	  print 'Training classifier with all the data..'
	  abc.fit(X_train.values, Y_train.values, sample_weight=W_train.values )

	  print 'Done.. Applying to validation sample and drawing ROC' 
	  #prob_predict_valid = gbc.predict_proba(X_valid)[:,1]
	  Y_score = abc.decision_function(X_valid.values)
	  fpr, tpr, _ = roc_curve(Y_valid.values, Y_score,W_valid.values)

	  #	  labelstring = ' and '.join(var.replace(Algorithm,'') for var in varset)
	  plt.plot(tpr, (1-fpr), label=' and '.join(varset))
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.ylabel('1- Background Efficiency')
	plt.xlabel('Signal Efficiency')
	plt.title('ROC Curve')
	plt.legend(loc="lower left",prop={'size':6})
	#plt.show()
	plt.savefig('rocmva.pdf')
        
	#        print 'Saving model\n'
	#        joblib.dump(gbc, 'MyModel.pkl')         
        
if __name__ == '__main__':
    main()
