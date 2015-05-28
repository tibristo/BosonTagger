import sys
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
from sklearn.covariance import EmpiricalCovariance, MinCovDet

trainvars = ['Aplanarity','ThrustMin','Tau1','Sphericity','m','FoxWolfram20','Tau21','ThrustMaj','EEC_C2_1','pt','EEC_C2_2','Dip12','phi','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','Angularity','ZCUT12','Tau2','EEC_D2_2','eta','TauWTA1','PlanarFlow']

def plotVars(algorithm):
    data = pd.read_csv('csv/'+algorithm+'_merged.csv')
    plt.figure()
    #data.diff().hist()
    #plt.show()
    #raw_input()
    for v in trainvars:
        data[v].hist(bins=50)
        print v
        #data[v].diff().hist(bins=20)
        plt.xlabel(v)
        #plt.show()
        plt.savefig(str(v+'.png'))
        plt.close()
        #raw_input()

def pca(algorithm):
    data = pd.read_csv('csv/'+algorithm+'_merged.csv')
    #print data[trainvars]
    #cov = np.cov(data['Aplanarity'],data['ThrustMin'],data['Tau1'],data['Sphericity'],data['m'],data['FoxWolfram20'],data['Tau21'],data['ThrustMaj'],data['EEC_C2_1'],data['pt'],data['EEC_C2_2'],data['Dip12'],data['phi'],data['SPLIT12'],data['TauWTA2TauWTA1'],data['EEC_D2_1'],data['YFilt'],data['Mu12'],data['TauWTA2'],data['Angularity'],data['ZCUT12'],data['Tau2'],data['EEC_D2_2'],data['eta'],data['TauWTA1'],data['PlanarFlow'])
    ec = EmpiricalCovariance().fit(data[trainvars])
    #ec = MinCovDet().fit(data[trainvars])
    #print ec.covariance_
    # get the eigenvectors/values from the covariance matrix.
    eig_val_cov, eig_vec_cov = np.linalg.eig(ec.covariance_)
    

    for i in range(len(eig_val_cov)):
        eigvec_cov = eig_vec_cov[:,i].reshape(1,26).T
        #print('Eigenvector {}: \n{}'.format(trainvars[i], eigvec_cov))
        print('Eigenvalue {} from covariance matrix: {}'.format(trainvars[i], eig_val_cov[i]))
        print(40 * '-')

    # check that the eigenvalues and vectors are correction: covmatrix.eigenvec = eigenval.eigenvector
    for i in range(len(eig_val_cov)):
        eigv = eig_vec_cov[:,i].reshape(1,26).T
        np.testing.assert_array_almost_equal(ec.covariance_.dot(eigv),\
                                                 eig_val_cov[i] * eigv, decimal=2,\
                                                 err_msg='', verbose=True)

    # list of eigenval/vec tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    
    eig_pairs, trainvars_eig = (list(t) for t in zip(*sorted(zip(eig_pairs, trainvars))))
    #eig_pairs.sort()
    #eig_pairs.reverse()

    #for i in zip(trainvars_eig, eig_pairs):
    #    print i[0]+':' + str(i[1][0])

    # get d X k eigenvector matrix W 
    # choose k to be teh best k eigenvectors
    # matrix_w = np.hstack((eig_pairs[0][1].reshape(26,1), eig_pairs[1][1].reshape(26,1)))
    #transformed = matrix_w.T.dot(data[trainvars])
    #print eig_val_cov
    #print cov
    #print data['Tau1']
    #np.cov[]

def main():
    
    Algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedL_v1_1000_1500'
    plotVars(Algorithm)
    return
    #Algorithm = sys.argv[1]
    #Algorithm = 'CamKt12LCTopoSplitFilteredMu100SmallR30YCut414tev_350_500_vxp_0_99'
    print 'Loading training data ...'

    data_train = pd.read_csv('csv/'+Algorithm+'_merged.csv')   
    r =np.random.rand(data_train.shape[0])
    
    #Set label and weight vectors - and drop any unwanted tranining one
    Y_train = data_train['label'].values[r<0.5]
    # W_train = data_train['weight'].values[r<0.9]
    Y_valid = data_train['label'].values[r>=0.5]
    # W_valid = data_train['weight'].values[r>=0.9]
    # data_train.drop('AKT10LCTRIM530_MassDropSplit',axis=1, inplace=True)
    print data_train.columns.values[1:-1]
    #varcombinations = itertools.combinations(data_train.columns.values[1:-1],2)
    varcombinations = itertools.combinations(trainvars[:],2)
    fac = lambda n: 1 if n < 2 else n * fac(n - 1)
    combos = lambda n, k: fac(n) / fac(k) / fac(n - k)

    #colors = plt.get_cmap('jet')(np.linspace(0, 1.0,combos(len(data_train.columns.values[1:-1]),2) ))
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0,combos(len(trainvars),2) ))

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

        # if we want to compare directly with the cut-based method we need to calculate 1/(1-roc(0.5)).
        # however, this is what we do when we've already applied the mass window. This does not do so.
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
