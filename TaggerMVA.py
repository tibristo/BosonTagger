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

def main():
    print 'Loading training data ...'
    data_train = pd.read_csv('merged.csv')
    r =np.random.rand(data_train.shape[0])
        	#Algorithm = 'AKT10LCTRIM530'
    
    plt.figure(1)
    Y_train = data_train['label'][r<0.9]
#    W_train = data_train['weight'][r<0.9]
    Y_valid = data_train['label'][r>=0.9]
#    W_valid = data_train['weight'][r>=0.9]
#    data_train.drop('AKT10LCTRIM530_MassDropSplit', axis=1, inplace=True)
    for varset in itertools.combinations(data_train.columns.values[1:-1],2):
        print list(varset)
        X_train = data_train[list(varset)][r<0.9]
        X_valid = data_train[list(varset)][r>=0.9]
    
    	  #gbc = Pipeline([("scale", StandardScaler()), ("gbc",GBC(n_estimators=1,verbose=1, max_depth=10,min_samples_leaf=50))])
    	  #	  gbc = GBC(n_estimators=20,verbose=1, max_depth=10,min_samples_leaf=50)
        #gbc = GaussianNB()
        dt = DC(max_depth=3,min_samples_leaf=0.05*len(X_train))
        abc = ABC(dt,algorithm='SAMME',
                         n_estimators=800,
                         learning_rate=0.5)
        print 'Training classifier with all the data..'
        abc.fit(X_train.values, Y_train.values)
#    sample_weight=W_train.values 
        print 'Done.. Applying to validation sample and drawing ROC' 
        prob_predict_valid = abc.predict(X_valid)
        #[:,1]
        #
        print prob_predict_valid
        Y_score = abc.decision_function(X_valid.values)
        print Y_score
        fpr, tpr, _ = roc_curve(Y_valid.values, Y_score)
#        W_valid.values
        labelstring = 'And'.join(var.replace('_','') for var in varset)
        print labelstring    
        plt.plot(tpr, (1-fpr), label=labelstring)
        plt.figure(2)       
        plt.hist(abc.decision_function(X_valid[Y_valid==1.]).ravel(),
         color='r', alpha=0.5, range=(-1.0,1.0), bins=50)
        plt.hist(abc.decision_function(X_valid[Y_valid==0.]).ravel(),
         color='b', alpha=0.5, range=(-1.0,1.0), bins=50)
        plt.xlabel("scikit-learn BDT output")
        plt.savefig(labelstring+'bdtout.pdf')        
    	  #	  labelstring = ' and '.join(var.replace(Algorithm,'') for var in varset)
    plt.figure(1)   
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('1- Background Efficiency')
    plt.xlabel('Signal Efficiency')
    plt.title('ROC Curve')
    plt.legend(loc="lower left",prop={'size':6})
    #plt.show()
    plt.savefig('rocmva.pdf')
    
#    print('Histogramming signal probabilities...')
#    validation_b_indices = (['Label'] == 'b').as_matrix()
#    validation_b_histo = np.histogram(
#        validation_signal_probabilities[validation_b_indices],
#        bins = 50,
#        range = (0.0, 1.0)
#    )[0]
#    validation_s_histo = np.histogram(
#        validation_signal_probabilities[~validation_b_indices],
#        bins = 50,
#        range = (0.0, 1.0)
#    )[0]
 
	#        print 'Saving model\n'
	#        joblib.dump(gbc, 'MyModel.pkl')         
        
if __name__ == '__main__':
    main()
