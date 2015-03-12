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
    #Algorithm = 'CamKt12LCTopoSplitFilteredMu67SmallR0YCut9'
    Algorithm = 'CamKt12LCTopoSplitFilteredMu100SmallR30YCut414tev_350_500_vxp_0_99'
    print 'Loading training data ...'

    data_train = pd.read_csv('csv/'+Algorithm+'-merged.csv')   
    r =np.random.rand(data_train.shape[0])
    
    #Set label and weight vectors - and drop any unwanted tranining one
    Y_train = data_train['label'].values[r<0.5]
    # W_train = data_train['weight'].values[r<0.9]
    Y_valid = data_train['label'].values[r>=0.5]
    # W_valid = data_train['weight'].values[r>=0.9]
    # data_train.drop('AKT10LCTRIM530_MassDropSplit', axis=1, inplace=True)

    varcombinations = itertools.combinations(data_train.columns.values[1:-1],2)
    fac = lambda n: 1 if n < 2 else n * fac(n - 1)
    combos = lambda n, k: fac(n) / fac(k) / fac(n - k)

    colors = plt.get_cmap('jet')(np.linspace(0, 1.0,combos(len(data_train.columns.values[1:-1]),2) ))

    for varset,color in zip(varcombinations, colors):
        print list(varset)
        X_train = data_train[list(varset)].values[r<0.5]
        X_valid = data_train[list(varset)].values[r>=0.5]


        dt = DC(max_depth=3,min_samples_leaf=0.05*len(X_train))
        abc = ABC(dt,algorithm='SAMME',
                 n_estimators=8,
                 learning_rate=0.5)
        print 'Training classifier with all the data..'
        abc.fit(X_train, Y_train)
        print 'Done.. Applying to validation sample and drawing ROC' 
        prob_predict_valid = abc.predict_proba(X_valid)[:,1]
        Y_score = abc.decision_function(X_valid)
        fpr, tpr, _ = roc_curve(Y_valid, prob_predict_valid)
        labelstring = ' And '.join(var.replace('_','') for var in varset)
        print labelstring
        plt.plot(tpr, (1-fpr), label=labelstring, color=color)

        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('1- Background Efficiency')
    plt.xlabel('Signal Efficiency')
    plt.title(Algorithm+' ROC Curve')
    plt.legend(loc="lower left",prop={'size':6})
    plt.savefig(Algorithm+'rocmva.pdf')

if __name__ == '__main__':
    main()
