import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc
import itertools
from sklearn.covariance import EmpiricalCovariance, MinCovDet

#trainvars = ['Aplanarity','ThrustMin','Tau1','Sphericity','m','FoxWolfram20','Tau21','ThrustMaj','EEC_C2_1','pt','EEC_C2_2','Dip12','phi','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','Angularity','ZCUT12','Tau2','EEC_D2_2','eta','TauWTA1','PlanarFlow']
trainvars = ['Tau1','m','EEC_C2_1','pt','EEC_C2_2','phi','TauWTA2TauWTA1','EEC_D2_1','TauWTA2','Tau2','EEC_D2_2','eta','TauWTA1']
#trainvars = trainvars[::-1]

def svd(algorithm):
    from scipy.linalg import svd
    from scipy.misc import lena
    
    data = pd.read_csv('csv/'+algorithm+'_bkg.csv')
    #print data[trainvars]
    #cov = np.cov(data['Aplanarity'],data['ThrustMin'],data['Tau1'],data['Sphericity'],data['m'],data['FoxWolfram20'],data['Tau21'],data['ThrustMaj'],data['EEC_C2_1'],data['pt'],data['EEC_C2_2'],data['Dip12'],data['phi'],data['SPLIT12'],data['TauWTA2TauWTA1'],data['EEC_D2_1'],data['YFilt'],data['Mu12'],data['TauWTA2'],data['Angularity'],data['ZCUT12'],data['Tau2'],data['EEC_D2_2'],data['eta'],data['TauWTA1'],data['PlanarFlow'])
    datatrain = data[trainvars]
    #print datatrain['Aplanarity']
    print datatrain['Aplanarity']
    print np.mean(datatrain)
    print np.std(datatrain)
    datatrain = (datatrain - np.mean(datatrain))/np.std(datatrain)

    print datatrain['Aplanarity']
    # singular value decomposition factorises your data matrix such that:
    # 
    #   M = U*S*V.T     (where '*' is matrix multiplication)
    # 
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    #
    # * S is a diagonal matrix containing the singular values of M - these 
    #   values squared divided by the number of observations will give the 
    #   variance explained by each PC.
    #
    # * if M is considered to be an (observations, features) matrix, the PCs
    #   themselves would correspond to the rows of S^(1/2)*V.T. if M is 
    #   (features, observations) then the PCs would be the columns of
    #   U*S^(1/2).
    #
    # * since U and V both contain orthonormal vectors, U*V.T is equivalent 
    #   to a whitened version of M.
    
    U, s, Vt = svd(datatrain, full_matrices=False)
    V = Vt.T
    
    # sort the PCs by descending order of the singular values (i.e. by the
    # proportion of total variance they explain)
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    print s
    V = V[:, ind]
    print ind
    vars_srt = trainvars[ind]
    # if we use all of the PCs we can reconstruct the noisy signal perfectly
    S = np.diag(s)
    for i in range(0,vars_srt):
        print vars_srt[i] + ': ' + str(S[i][i])
    #Mhat = np.dot(U, np.dot(S, V.T))
    #print "Using all PCs, MSE = %.6G" %(np.mean((datatrain - Mhat)**2))

    # if we use only the first 20 PCs the reconstruction is less accurate
    #Mhat2 = np.dot(U[:, :20], np.dot(S[:20, :20], V[:,:20].T))
    #print "Using first 20 PCs, MSE = %.6G" %(np.mean((datatrain - Mhat2)**2))

    
    
def plotVars(algorithm):
    data = pd.read_csv('csv/'+algorithm+'_bkg.csv')
    plt.figure()
    data = (data - np.mean(data))/np.std(data)
    #data.diff().hist()
    #plt.show()
    #raw_input()
    for v in trainvars:
        data[v].hist(bins=50)
        print v
        #data[v].diff().hist(bins=20)
        plt.xlabel(v)
        #plt.show()
        plt.savefig(str(algorithm+'_'+v+'_norm.png'))
        plt.close()
        #raw_input()

def pca(algorithm):
    plt.rcParams['figure.figsize'] = 10, 7.5
    plt.rcParams['axes.grid'] = True
    plt.gray()
    data = pd.read_csv('/media/win/BoostedBosonFiles/csv/'+algorithm+'_merged.csv')
    print trainvars
    datatrain = data[trainvars]
    # standardise the data
    datatrain = (datatrain - np.mean(datatrain))/np.std(datatrain)
    y = data['label'].values
    from sklearn.decomposition import RandomizedPCA
    #pca = PCA().fit(datatrain, datalabels)
    pca = RandomizedPCA(n_components=2)
    x_pca = pca.fit_transform(datatrain)
    print x_pca.shape
    from itertools import cycle

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    markers = ['+', 'o', '^', 'v', '<', '>', 'D', 'h', 's']
    for i, c, m in zip(np.unique(y), cycle(colors), cycle(markers)):
        plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1],c=c, marker=m, label=i, alpha=0.5)
        
    _ = plt.legend(loc='best')
    plt.show()

    from sklearn.decomposition import PCA
    pca_big = PCA().fit(datatrain, y)
    plt.title("Explained Variance")
    plt.ylabel("Percentage of explained variance")
    plt.xlabel("PCA Components")
    plt.plot(pca_big.explained_variance_ratio_);
    plt.show()

def main():
    
    Algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedL_ranged_v2_1000_1500'
    pca(Algorithm)
    #plotVars(Algorithm)
    return
    #Algorithm = sys.argv[1]
    #Algorithm = 'CamKt12LCTopoSplitFilteredMu100SmallR30YCut414tev_350_500_vxp_0_99'
    print 'Loading training data ...'

    data_train = pd.read_csv('csv/'+Algorithm+'_merged.csv')   
    #standardise data
    for t in trainvars:
        minx = np.amin(data_train[t])
        maxx = np.amax(data_train[t])
        data_train[t] = (data_train[t] - minx)/(maxx-minx)
        #data_train[t] = (data_train[t] - np.mean(data_train[t]))/np.std(data_train[t])
    r =np.random.rand(data_train.shape[0])
    
    #Set label and weight vectors - and drop any unwanted tranining one
    Y_train = data_train['label'].values[r<0.5]
    # W_train = data_train['weight'].values[r<0.9]
    Y_valid = data_train['label'].values[r>=0.5]
    # W_valid = data_train['weight'].values[r>=0.9]
    # data_train.drop('AKT10LCTRIM530_MassDropSplit',axis=1, inplace=True)
    print data_train.columns.values[1:-1]
    #varcombinations = itertools.combinations(data_train.columns.values[1:-1],2)
    varcombinations = itertools.combinations(trainvars[:],26)
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
        print abc.feature_importances_

        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('1- Background Efficiency')
    plt.xlabel('Signal Efficiency')
    plt.title(Algorithm+' ROC Curve')
    plt.legend(loc="lower left",prop={'size':6})
    plt.savefig(Algorithm+'rocmva.pdf')

if __name__ == '__main__':
    main()
