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
from matplotlib.colors import ListedColormap

print 'Loading training data ...'

#Algorithm = 'CamKt12LCTopoSplitFilteredMu67SmallR0YCut9'
Algorithm = 'CamKt12LCTopoSplitFilteredMu100SmallR30YCut4'
Energy = '14TeV'
data_train = pd.read_csv(Algorithm+Energy+'merged.csv')
np.random.seed(42)
r =np.random.rand(data_train.shape[0])
    	#Algorithm = 'AKT10LCTRIM530'

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
    X = data_train[list(varset)].values
    Y = data_train['label'].values
    
    X_train = data_train[list(varset)].values[r<0.5]
    X_valid = data_train[list(varset)].values[r>=0.5]
   # X_train = data_train.values[:,1:3][r<0.5]
   # X_valid = data_train.values[:,1:3][r>=0.5]
    Y_train = data_train['label'].values[r<0.5]
    Y_valid = data_train['label'].values[r>=0.5]
    
     #	  gbc = GBC(n_estimators=20,verbose=1, max_depth=10,min_samples_leaf=50)
    #gbc = GaussianNB()
    dt = DC(max_depth=3,min_samples_leaf=0.05*len(X_train))
    abc = ABC(dt,algorithm='SAMME',
                 n_estimators=8,
                 learning_rate=0.5)
   
    print 'Training classifier with all the data..'
    abc.fit(X_train, Y_train)
    
    print 'Drawing fancy plots'

    plot_colors = "br"
    plot_step = 0.2
    class_names = ["W Jets","QCD"]
    
    plt.figure(figsize=(10, 5))
    
    # Plot the decision boundaries
    plt.subplot(121)
    x_min, x_max = X[:, 0].min() , X[:, 0].max()
    y_min, y_max = X[:, 1].min() , X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/10000),
                         np.arange(y_min, y_max, (y_max-y_min)/10000))
    print 'made mesh'

    Z = abc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    plt.axis("tight")
    print 'Drawing fancy plots - train points'

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(Y_valid == i)
        plt.scatter(X_valid[idx, 0], X_valid[idx, 1],
                    c=c, cmap=plt.cm.Paired,
                    label="%s" % n)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.ylabel(list(varset)[1])
    plt.xlabel(list(varset)[0])
    print 'Drawing fancy plots - decision scores'
#
#    # Plot the two-class decision scores
    twoclass_output = abc.decision_function(X_valid)
    plot_range = (twoclass_output.min(), twoclass_output.max())
    plt.subplot(122)
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[Y_valid == i],
                 bins=10,
                 range=plot_range,
                 facecolor=c,
                 label='%s' % n,
                 alpha=.5)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
  #  plt.ylabel(list(varset)[1])
    plt.xlabel('Decision Scores')
    plt.title('Decision Scores')
    
    plt.subplots_adjust(wspace=0.25)
    labelstring = 'And'.join(var.replace('_','') for var in varset)
    print labelstring
    plt.savefig(labelstring+'fancyplot.pdf')
    #break