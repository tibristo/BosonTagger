# A number of tools for doing MVA analysis.
# Includes:
# Cross validation 
# Grid search
# Learning curves
# Probability distributions
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
import os, sys, time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
from collections import OrderedDict
import bz2
import math
import argparse

# some common variables and their latex format
label_dict = {'TauWTA2':r"$\tau^{WTA}_{2}$",'EEC_C2_1':r"$C^{(\beta=1)}_{2}$",'EEC_C2_2':r"$C^{(\beta=2)}_{2}$",'EEC_D2_1':r"$D^{(\beta=1)}_{2}$",'EEC_D2_2':r"$D^{(\beta=2)}_{2}$", 'SPLIT12':r"$\sqrt{d_{12}}$",'Aplanarity':r"$\textit{A}$", 'PlanarFlow':r"\textit{P}", 'ThrustMin':r"$T_{min}$",'Sphericity':r"$\textit{S}$",'Tau21':r"$\tau_{21}$",'ThrustMaj':r"$T_{maj}$",'Dip12':r"$D_{12}$",'TauWTA2TauWTA1':r"$\tau^{WTA}_{21}$",'YFilt':r"$YFilt$",'Mu12':r"$\mu_{12}$",'ZCUT12':r"$\sqrt{z_{12}}$",'Tau2':r"$\tau_2$",'nTracks':r"$nTrk$"}

   

#import cv_fold
def persist_cv_splits(X, y, w, variables, n_cv_iter=5, name='data', prefix='persist/',\
                      suffix="_cv_%03d.pkl", test_size=0.25, random_state=None, scale=True, overwrite=True, overwrite_full=True,signal_eff=1.0, bkg_eff=1.0, onlyFull = False):
    """Materialize randomized train test splits of a dataset."""
    import os.path
    from root_numpy import array2root
    import numpy.lib.recfunctions as nf
    #cv = StratifiedKFold(y,n_cv_iter)
    cv = StratifiedShuffleSplit(y, n_cv_iter, test_size = test_size)#KFold(y,n_cv_iter)
    #cv = ShuffleSplit(X.shape[0], n_iter=n_cv_iter,
    #    test_size=test_size, random_state=random_state)
    cv_split_filenames = []

    # persist the original files as well.
    # first check if the file exists already
    full_fname = os.path.abspath(prefix+name+suffix % 100)
    print full_fname
    if overwrite_full or not os.path.isfile(full_fname):
        full_set = (X,y,w,[signal_eff,bkg_eff])
        joblib.dump(full_set,full_fname)
    if onlyFull:
        return
        
    for i, (train, test) in enumerate(cv):
        cv_split_filename = prefix+name + suffix % i
        cv_split_filename = os.path.abspath(cv_split_filename)
        cv_split_filenames.append(cv_split_filename)

        if os.path.isfile(cv_split_filename) and overwrite == False:
            continue

        if scale:
            # should we scale the signal and background separately??? I think so!
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train])
            X_test_scaled = scaler.transform(X[test])
            #fold = cv_fold.cv_fold(X_train_scaled, y[train], w[train], X_test_scaled, y[test], w[test])
            fold = (X_train_scaled, y[train], w[train], X_test_scaled, y[test], w[test])
        else:
            #fold = cv_fold.cv_fold(X[train], y[train], w[train], X[test], y[test], w[test])
            fold = (X[train], y[train], w[train], X[test], y[test], w[test])

        joblib.dump(fold, cv_split_filename)
        cv_split_train = os.path.abspath(prefix+name+'train'+suffix%i)
        cv_split_test = os.path.abspath(prefix+name+'test'+suffix%i)

        # have to do this annoying .copy() to be able to add the dtype.names to any
        # arrays that come from a slice.
        XX = X.copy().view(dtype=[(n, np.float64) for n in variables]).reshape(len(X))

        # we want to add the weights as well        
        # add the label to the array
        rectrain = nf.append_fields(XX[train], names=['label','weight'], data=[y[train],w[train]], usemask=False)
        array2root(rectrain, cv_split_train.replace('.pkl','.root'), 'outputTree','recreate')
        rectest = nf.append_fields(XX[test], names=['label','weight'], data=[y[test],w[test]], usemask=False)
        array2root(rectest, cv_split_test.replace('.pkl','.root'), 'outputTree','recreate')
    
    return cv_split_filenames

def round_sigfigs(num, sig_figs):
    '''
    Round number to specified number of significant digits
    '''
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs-1)))
    else:
        return 0 # can't take the log of 0

def drawMatrix(key, corr_matrix, title, taggers, matrix_size, file_id = '', drawATLAS = False, sampletype = 'combined'):
    import ROOT
    ROOT.gROOT.SetBatch(1)
    #ROOT.gStyle.SetPaintTextFormat("4.1f")
    corr_hist = ROOT.TH2F(title,title,matrix_size, 1, matrix_size+1, matrix_size, 1, matrix_size+1)
    # we can set the labels of the covariance matrices from label_dict
    # right now label_dict has the latex versions of the variables, but this doesn't work with th2f, need
    # to replace \ with #
    for i, t in enumerate(taggers):
        label = label_dict[t].replace('$','').replace("\\","#").replace("text","")
        #corr_hist.GetXaxis().SetBinLabel(i+1, label_dict[t])
        corr_hist.GetXaxis().SetBinLabel(i+1, label)
        #corr_hist.GetYaxis().SetBinLabel(i+1, label_dict[t])
        corr_hist.GetYaxis().SetBinLabel(i+1, label)
    # the set bin content does set(x,y), but the loop here goes through the rows and columns - y and x.
    for row in range(len(corr_matrix)-1,-1,-1): # row = y
        # only want to draw the top diagonal!
        for col in range(len(corr_matrix)): # col = x
            # also, axis in corr_matrix is 0->n down the y axis, but n+1->1 in corr_hist
            if col < row +1:
                corr_hist.SetBinContent(col+1, row+1, round_sigfigs(corr_matrix[row][col], 3))
            else:
                corr_hist.SetBinContent(col+1, row+1, -1.1)
    # create the canvas and set up the latex and legends
    #corr_hist.SetAxisColor(1,"Z")
    tc = ROOT.TCanvas()
    # turn on the colours
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPadBorderSize(0)
    ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetLeftMargin(0.1)
    ROOT.gPad.SetRightMargin(0.15)
    corr_hist.SetTitle("")
    corr_hist.SetStats(0)
    corr_hist.SetMarkerSize(0.7)
    corr_hist.Draw("TEXTCOLZ")
    corr_hist.GetZaxis().SetTitle('Linear Correlation')
    corr_hist.GetZaxis().SetRangeUser(-1.0,1.0)
    from ROOT import TLatex
    # need to put the pt range on here....
    # okay, so this is nasty and poor form, but I'm super stressed and running out of time
    # to finish my thesis, so whatever.  The algorithm name should have the pt range in it in gev
    # At this point things are narrowed down to the point where we are only considering two
    # pt ranges: 400-1600 GeV or 800-1200 GeV, so just look for those.
    ptrange = ''
    if key.find('400_1600') != -1:
        ptrange = '400<p_{T}^{Truth}<1600 GeV'
    elif key.find('800_1200') != -1:
        ptrange = '800<p_{T}^{Truth}<1200 GeV'

        
    if ptrange != '':
        # draw it
        ptl = TLatex()
        ptl.SetNDC()
        ptl.SetTextFont(42)
        ptl.SetTextSize(0.035)
        ptl.SetTextColor(ROOT.kBlack)
        ptl.DrawLatex(0.6,0.36,ptrange);#"Internal Simulation");
    # the bdt parameters too?!

    sample = TLatex();sample.SetNDC();sample.SetTextFont(42);sample.SetTextSize(0.035);sample.SetTextColor(ROOT.kBlack)
    if sampletype == 'bkg':
        sample.DrawLatex(0.6,0.41, "Bkg (QCD jets)")
    elif sampletype == 'sig':
        sample.DrawLatex(0.6,0.41, "Signal (W jets)")
    else:
        sample.DrawLatex(0.6,0.41, "Merged Signal & Bkg.")
        
    e = TLatex();e.SetNDC();e.SetTextFont(42);e.SetTextSize(0.035);e.SetTextColor(ROOT.kBlack)
    e.DrawLatex(0.6,0.46, "#sqrt{s}=13 TeV")

    m = TLatex();m.SetNDC();m.SetTextFont(42);m.SetTextSize(0.035);m.SetTextColor(ROOT.kBlack)
    m.DrawLatex(0.6,0.31,"68% mass window")

    #param = TLatex();param.SetNDC();param.SetTextFont(42);param.SetTextSize(0.035);param.SetTextColor(ROOT.kBlack)
    #param.DrawLatex(0.3,0.3,"BDT maxdepth=, rate=, est=")
    
    if drawATLAS: # also, in my hurry to do this, you'll notice that atlas is now drawn in the same place as the pt range above
        texw = TLatex();
        texw.SetNDC();
        texw.SetTextSize(0.035);
        texw.SetTextFont(72);
        texw.DrawLatex(0.6,0.91,"ATLAS");
        p = TLatex();
        p.SetNDC();
        p.SetTextFont(42);
        p.SetTextSize(0.035);
        p.SetTextColor(ROOT.kBlack);
        p.DrawLatex(0.68,0.91,"Simulation Work in Progress");#"Internal Simulation");
    # check that the matrix folder exists, if not, create it
    tc.SaveAs("corr_matrices/corr_matrix_"+file_id+".pdf")



def plotCorrelation(cv_split_filenames, full_dataset, taggers, key = '', sampletype='combined'):
    '''
    Method for plotting the covariance or correlation of all variables.

    cv_split_filenames --- cv fold filenames
    full_dataset --- full dataset from which the cv folds are created
    taggers --- the variable names
    key --- add this to the output file id
    '''

    import os.path
    import matplotlib.pylab as plt
    plt.rc('text',usetex=True)
    from sklearn.externals import joblib
    import numpy as np

    # create an output folder for the correlation matrices
    if not os.path.exists('corr_matrices'):
        os.makedirs('corr_matrices')

    # load the cross validation folds
    for cv in cv_split_filenames:
        X_train, y_train, w_train, X_validation, y_validation, w_validation = joblib.load(
            'persist/'+cv, mmap_mode='c')
        # the number of cols gives us the number of variables and size of the matrix
        matrix_size = len(taggers)
        # get the covariance matrix for training data
        corr_train = np.corrcoef(X_train, rowvar=0) #rowvar transposes the matrix to get rows, cols to be cols, rows to work with the corr method
        drawMatrix(key, corr_train, "Correlation Matrix for training data cv split "+str(cv), taggers, matrix_size, file_id='train_'+str(cv)+key, sampletype = sampletype)
        corr_valid = np.corrcoef(X_validation, rowvar=0)
        drawMatrix(key, corr_valid, "Correlation Matrix for validation data cv split "+str(cv), taggers, matrix_size, file_id='valid_'+str(cv)+key, sampletype = sampletype)
        
    # load the full dataset
    X,y,w,eff = joblib.load(full_dataset,mmap_mode='c')
    corr_full = np.corrcoef(X, rowvar=0)
    drawMatrix(key, corr_full, "Correlation Matrix for full dataset", taggers, matrix_size, file_id="full", sampletype =sampletype)
    
    # get the covariance matrix for the 


def plotSamples(cv_split_filename, full_dataset, taggers, key = '', first_tagger = False, weight_plots = False):
    '''
    Method for plotting the variables in the cv folds.  It also records stats of the cv folds including the number
    of events and the mean and std of the different taggers.

    cv_split_filename -- cv fold filename
    full_dataset --- the full dataset from which the cv folds were created
    taggers --- the variables used in the dataset for training
    first_tagger --- this is used to create a stats file for all of the cv folds combined. If this is true then it will recreate the file
    and it will also write the full dataset stats at the top of the file.
    weight_plots --- if the plots should be weighted or not
    '''


    
    import os.path
    import matplotlib.pylab as plt
    plt.rc('text',usetex=True)
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc

    if not os.path.exists('fold_plots'):
        os.makedirs('fold_plots')

    weight_flag = '_weighted' if weight_plots else ''
    
    # file mode for tagger_stats and event_count
    file_mode = 'w' if first_tagger else 'a'
    
    # load the cross validation folds
    X_train, y_train, w_train, X_validation, y_validation, w_validation = joblib.load(
        'persist/'+cv_split_filename, mmap_mode='c')
    # load the full dataset
    X,y,w,eff = joblib.load(full_dataset,mmap_mode='c')
    # only want the file not the folder
    stats_fname = os.path.basename(cv_split_filename)
    stats_fname = stats_fname.replace('.pkl','.txt')
    # get the cross validation fold number from the file name.  The file will always end in _XYZ.pkl, where XYZ are integers.
    cv_num = stats_fname.split('_')[-1].replace('.txt','')
    # create a stats file with the std, mean of each variable, number of entries
    if not os.path.exists('fold_stats'):
        os.makedirs('fold_stats')
    # update stats_fname with the weight flag
    stats_fname = stats_fname.replace('.txt', weight_flag+'.txt')
    stats = open('fold_stats/'+stats_fname,'w')
    
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Sample','Signal','Background','Total')+'\n')
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Full',str(X[y==1].shape[0]), str(X[y==0].shape[0]),str(X.shape[0]))+'\n')
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Training',str(X_train[y_train==1].shape[0]), str(X_train[y_train==0].shape[0]),str(X_train.shape[0]))+'\n')
    stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Valid',str(X_validation[y_validation==1].shape[0]), str(X_validation[y_validation==0].shape[0]),str(X_validation.shape[0]))+'\n\n')
    
    event_count = open('fold_stats/event_counts'+weight_flag+'.txt',file_mode)
    if first_tagger:
        event_count.write('{0:15}  {1:10} {2:14}{3:10}'.format('Full',str(X[y==1].shape[0]), str(X[y==0].shape[0]),str(X.shape[0]))+'\n')
    event_count.write('{0:15}  {1:10} {2:14}{3:10}'.format('Train cv '+cv_num,str(X_train[y_train==1].shape[0]), str(X_train[y_train==0].shape[0]),str(X_train.shape[0]))+'\n')
    event_count.write('{0:15}  {1:10} {2:14}{3:10}'.format('Valid cv '+cv_num,str(X_validation[y_validation==1].shape[0]), str(X_validation[y_validation==0].shape[0]),str(X_validation.shape[0]))+'\n')
    event_count.close()
    
    stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Variable','Mean','Std','Mean Sig','Std Sig','Mean Bkg','Std Bkg')+'\n\n')

    
    for i,t in enumerate(taggers):
        tagger_label = t
        if t in label_dict.keys():
            tagger_label = label_dict[t]
        tagger_stats = open('fold_stats/'+t+'_'+key+weight_flag+'.txt',file_mode)
        stats.write(tagger_label+'\n')
        # training
        mean_tr = "{0:.4f}".format(float(np.mean(X_train[:,i])))
        mean_signal_tr = '{0:.4f}'.format(float(np.mean(X_train[y_train==1][:,i])))
        mean_bkg_tr = '{0:.4f}'.format(float(np.mean(X_train[y_train==0][:,i])))
        std_tr = '{0:.4f}'.format(float(np.std(X_train[:,i])))
        std_signal_tr = '{0:.4f}'.format(float(np.std(X_train[y_train==1][:,i])))
        std_bkg_tr = '{0:.4f}'.format(float(np.std(X_train[y_train==0][:,i])))
        # validation
        mean_val = '{0:.4f}'.format(float(np.mean(X_validation[:,i])))
        mean_signal_val = '{0:.4f}'.format(float(np.mean(X_validation[y_validation==1][:,i])))
        mean_bkg_val = '{0:.4f}'.format(float(np.mean(X_validation[y_validation==0][:,i])))
        std_val = '{0:.4f}'.format(float(np.std(X_validation[:,i])))
        std_signal_val = '{0:.4f}'.format(float(np.std(X_validation[y_validation==1][:,i])))
        std_bkg_val = '{0:.4f}'.format(float(np.std(X_validation[y_validation==0][:,i])))
        # full dataset
        mean = '{0:.4f}'.format(float(np.mean(X[:,i])))
        mean_signal = '{0:.4f}'.format(float(np.mean(X[y==1][:,i])))
        mean_bkg = '{0:.4f}'.format(float(np.mean(X[y==0][:,i])))
        std = '{0:.4f}'.format(float(np.std(X[:,i])))
        std_signal = '{0:.4f}'.format(float(np.std(X[y==1][:,i])))
        std_bkg = '{0:.4f}'.format(float(np.std(X[y==0][:,i])))

        result = '{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Full',str(mean),str(std),str(mean_signal),str(std_signal),str(mean_bkg),str(std_bkg))
        result_tr = '{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Train',str(mean_tr),str(std_tr),str(mean_signal_tr),str(std_signal_tr),str(mean_bkg_tr),str(std_bkg_tr))
        result_val = '{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Valid',str(mean_val),str(std_val),str(mean_signal_val),str(std_signal_val),str(mean_bkg_val),str(std_bkg_val))
        
        stats.write(result+'\n'+result_tr+'\n'+result_val+'\n\n')
        if first_tagger:
            tagger_stats.write(result+'\n')
        tagger_stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Train cv '+cv_num,str(mean_tr),str(std_tr),str(mean_signal_tr),str(std_signal_tr),str(mean_bkg_tr),str(std_bkg_tr))+'\n')
        tagger_stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Valid cv '+cv_num,str(mean_val),str(std_val),str(mean_signal_val),str(std_signal_val),str(mean_bkg_val),str(std_bkg_val))+'\n')
        tagger_stats.close()

    stats.close()
    # normalise the data first? should have been standardised....
    # filename for plots:
    plt_fname = stats_fname.replace(weight_flag+'.txt','')

    # weights
    weight_sig = w_train[y_train==1] if weight_plots else None
    weight_bkg = w_train[y_train==0] if weight_plots else None
    print label_dict
    for i, t in enumerate(taggers):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.hist(X_train[y_train==1][:,i],normed=1, bins=50,color='red',label='signal',alpha=0.5,weights=weight_sig)
        ax.hist(X_train[y_train==0][:,i],normed=1, bins=50,color='blue',label='background',alpha=0.5,weights=weight_bkg)
        
        print t
        if t.strip() in label_dict.keys():
            plt.xlabel(label_dict[t])
            plt.title('Fold number ' + cv_num + ': ' + label_dict[t])
        else:
            print 'not in dict'
            plt.xlabel(t.replace('_','\_'))
            plt.title('Fold number ' + cv_num + ': '+ t.replace('_','\_'))
        # add the pt range and sqrt{s}=13 TeV onto the plot.
        # add the grooming algorithm?
        # TODO: again I'm stressed because of time and I can't afford to make this more general
        # and I'm hardcoding shit.  pt range is always 4-16 or 8-12, just one grooming alg.
        # some of these things need to get moved... After VISUALLY inspecting (what a cost!)
        # the output, these are the ones that need to have different coords
        # split12 - this one is tough, not sure what to do about this one
        # tauwta21 -  I think the legend can got top center - loc=9
        # thrustmaj - everything moved to x=0.2 or 0.3
        # yfilt - move text to x =0.2 or 0.3, move legend to loc=9
        # zcut12 - move legend to loc=9
        xval = 0.7
        locval = 7
        #if tagger.find('SPLIT12') != -1:
        #    notsure
        if t.find('TauWTA2TauWTA1') != -1 or t.find('ZCUT12') != -1:
            locval = 9
        elif t.find('ThrustMaj') != -1:
            xval = 0.02
            locval = 6
        elif t.find('YFilt') != -1:
            xval = 0.02
            locval = 9
        ax.text(xval,0.9, r'$\sqrt{s}=13$ TeV', transform = ax.transAxes )
        if plt_fname.find('4_16') != -1 or plt_fname.find('400_1600'):
            ax.text(xval,0.85, '$400<p_{T}<1600$ GeV', transform = ax.transAxes )
        else:
            ax.text(xval,0.85, '$800<p_{T}<1200$ GeV', transform = ax.transAxes )
        ax.text(xval,0.8, '68% mass window', transform = ax.transAxes )
        ax.text(xval,0.75, r'$\textrm{anti-k}_{t}~R=1.0~\textrm{jets}$', fontsize=10, transform = ax.transAxes )
        ax.text(xval,0.71, r'Trimmed', fontsize=10, transform = ax.transAxes )
        ax.text(xval,0.67, r'$f_{cut}=5\%,~R_{sub}=0.2$', fontsize=10, transform = ax.transAxes )

        # want to change the limits for the y value
        ymin, ymax = plt.ylim()
        plt.ylim(ymax=ymax*1.1)
        
        plt.ylabel('Number of events')
        plt.legend(loc=locval)
        plt.savefig('fold_plots/'+plt_fname+'_'+t+weight_flag+'.pdf')
        plt.clf()
        plt.close(fig)


def evaluateFull(model, model_eval_obj, file_full, transform_valid_weights, weight_validation, job_id, df_train, sig_tr_idx, bkg_tr_idx, bkg_rej_train):
    import os
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
    import modelEvaluation as me
    import sys

    import numpy as np
    #print file_full
    X_full, y_full, w_full, efficiencies = joblib.load(file_full, mmap_mode='c')
    #print X_full.shape
    print efficiencies
    if transform_valid_weights and weight_validation:
        for idx in xrange(0, w_full.shape[0]):
            if y_full[idx] == 1:
                w_full[idx] = 1.0
            else:
                w_full[idx] = np.arctan(1./w_full[idx])
    if not weight_validation:
        w_full_tmp = None
    else:
        w_full_tmp = w_full
    full_score = model.score(X_full, y_full, sample_weight=w_full_tmp)
    prob_predict_full = model.predict_proba(X_full)[:,1]
    fpr_full, tpr_full, thresh_full = roc_curve(y_full, prob_predict_full, sample_weight=w_full_tmp)
    # need to set the maximum efficiencies for signal and bkg
    m_full = me.modelEvaluation(fpr_full, tpr_full, thresh_full, model, model_eval_obj.params, job_id, model_eval_obj.taggers, model_eval_obj.Algorithm, full_score, file_full,feature_importances=model.feature_importances_, decision_function=df_train, decision_function_sig = sig_tr_idx, decision_function_bkg = bkg_tr_idx)
    m_full.setSigEff(efficiencies[0])
    m_full.setBkgEff(efficiencies[1])
    # get the indices in the full sample
    sig_full_idx = y_full == 1
    bkg_full_idx = y_full == 0
    # set the probabilities and the true indices of the signal and background
    m_full.setProbas(prob_predict_full, sig_full_idx, bkg_full_idx, w_full_tmp)
    # set the different scoresx
    y_pred_full = model.predict(X_full)
    m_full.setScores('test',accuracy=accuracy_score(y_pred_full, y_full, sample_weight=w_full_tmp), precision=precision_score(y_pred_full, y_full, sample_weight=w_full_tmp), recall=recall_score(y_pred_full, y_full, sample_weight=w_full_tmp), f1=f1_score(y_pred_full, y_full, sample_weight=w_full_tmp))
    m_full.setScores('train',accuracy=model_eval_obj.train_accuracy, precision=model_eval_obj.train_precision, recall=model_eval_obj.train_recall, f1=model_eval_obj.train_f1)
    # write this into a root file
    m_full.toROOT()
    # save the train score
    m_full.setTrainRejection(bkg_rej_train)
    m_full.plotDecisionFunction()


    # save the model to use later.

    import bz2
    import pickle
    f_name_full = 'evaluationObjects/'+job_id+'.pbz2'
    try:
        with bz2.BZ2File(f_name_full,'w') as d:
            pickle.dump(m_full, d)
            d.close()
    except:
        msg = 'unable to dump '+job_id+ ' object'
        with bz2.BZ2File(f_name_full,'w') as d:
            pickle.dump(msg, d)
        d.close()
        #print 'unable to dump '+job_id+ '_full object:', sys.exc_info()[0]
        
def compute_evaluation(cv_split_filename, model, params, job_id = '', taggers = [], weighted=True, algorithm='', full_dataset='',compress=True, transform_weights=True, transform_valid_weights=False, weight_validation=False):
    """Function executed by a worker to evaluate a model on a CV split

    Usage:
    cv_split_filename:
    model:
    params:
    
    """
    import os
    from sklearn.externals import joblib
    import numpy as np
    from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
    import modelEvaluation as me
    import sys

    import numpy as np
    #import cv_fold
    print cv_split_filename
    X_train, y_train, w_train, X_validation, y_validation, w_validation = joblib.load(
        cv_split_filename, mmap_mode='c')

    # get the indices in the validation sample
    sig_idx = y_validation == 1
    bkg_idx = y_validation == 0
    # get the indices in the training sample
    sig_tr_idx = y_train == 1
    bkg_tr_idx = y_train == 0

    # set up the model
    model.set_params(**params)
    # if we are weighting I think that we need to have both the MC weights (or weights we have from our physics knowledge)
    # and the normalisation weights.
    # think that the correct way to do it would be to apply your weights as you would normally. You want to apply the weights so that you can correctly (according to our physics knowledge) represent the data.  I think that after that you can apply the normalisation.  This keeps the shape of the distribution the same, it just gives the bdt a better way of classifying the signal.  I mean I think you could get the same performance more or less if you had enough signal anyway.
    # this means that we then have to adjust the w_train sample so that we have w_train[sig_tr_idx] *= 1/np.count_nonzero(sig_tr_idx) and w_train[bkg_tr_idx] *= 1/np.count_nonzero(bkg_tr_idx)
    # set up array with the scaling factors
    sig_count = (1/float(np.count_nonzero(sig_tr_idx)))
    bkg_count = (1/float(np.count_nonzero(bkg_tr_idx)))
    sig_scaling = sig_tr_idx*sig_count
    bkg_scaling = bkg_tr_idx*bkg_count
    tot_scaling = sig_scaling+bkg_scaling

    #w_train = w_train*tot_scaling
    # the weight transformation was very successful on the dnn, so I'm going to try it here too
    # implementation of the weight transform done on the 6th of Nov 2015.
    # copied from the create_folds.py file in the WTaggingNN setup
    if transform_weights:
        for idx in xrange(0, w_train.shape[0]):
            if y_train[idx] == 1:
                w_train[idx] = 1.0
            else:
                w_train[idx] = np.arctan(1./w_train[idx])
    if transform_valid_weights:# and weight_validation:
        for idx in xrange(0, w_validation.shape[0]):
            if y_validation[idx] == 1:
                w_validation[idx] = 1.0
            else:
                w_validation[idx] = np.arctan(1./w_validation[idx])

    if weighted:
        model.fit(X_train, y_train, w_train)
        #validation_score = model.score(X_validation, y_validation, w_validation)
    else:
        model.fit(X_train, y_train)

    # we want to do this for both the validation sample AND the full sample so that we
    # can compare it with the cut-based tagger.
    if weighted:
        w_test = w_validation
    else:
        w_test = None
    validation_score = model.score(X_validation, y_validation, sample_weight=w_test)
    prob_predict_valid = model.predict_proba(X_validation)[:,1]
    fpr, tpr, thresholds = roc_curve(y_validation, prob_predict_valid, sample_weight=w_test)
    

    m = me.modelEvaluation(fpr, tpr, thresholds, model, params, job_id, taggers, algorithm, validation_score, cv_split_filename, feature_importances=model.feature_importances_, decision_function=model.decision_function(X_train), decision_function_sig = sig_tr_idx, decision_function_bkg = bkg_tr_idx)
    m.setProbas(prob_predict_valid, sig_idx, bkg_idx, w_validation)
    # set all of the scores
    y_val_pred = model.predict(X_validation)
    m.setScores('test',accuracy=accuracy_score(y_val_pred, y_validation, sample_weight=w_test), precision=precision_score(y_val_pred, y_validation, sample_weight = w_test), recall=recall_score(y_val_pred, y_validation, sample_weight = w_test), f1=f1_score(y_val_pred, y_validation, sample_weight = w_test))
    y_train_pred = model.predict(X_train)
    m.setScores('train',accuracy=accuracy_score(y_train_pred, y_train, sample_weight = w_train), precision=precision_score(y_train_pred, y_train, sample_weight = w_train), recall=recall_score(y_train_pred, y_train, sample_weight = w_train), f1=f1_score(y_train_pred, y_train, sample_weight = w_train)
)
    # create the output root file for this.
    m.toROOT()
    # score to return
    roc_bkg_rej = m.getRejPower()
    # calculate the training score as well
    prob_predict_train = model.predict_proba(X_train)[:,1]
    bkg_rej_train = m.calculateBkgRej(prob_predict_train, sig_tr_idx, bkg_tr_idx, w_train)
    m.setTrainRejection(bkg_rej_train)

    # save the model for later
    import pickle
    import bz2
    f_name = 'evaluationObjects/'+job_id+'.pbz2'
    try:
        with bz2.BZ2File(f_name,'w') as d:
            pickle.dump(m, d)
            d.close()
    except:
        msg = 'unable to dump '+job_id+ ' object'
        with bz2.BZ2File(f_name,'w') as d:
            pickle.dump(msg, d)

    
    # do this for the full dataset
    # try reading in the memmap file
    # the easiest way to find the name of the file is to take the cv_split_filename
    # and then search backwards to find an underscore. The number between this underscore
    # and the file extension, pkl, should be 100 for this file. It is written this
    # way in the persist_cv_ method in this file.
    if full_dataset == '':
        underscore_idx = cv_split_filename.rfind('_')
        if underscore_idx == -1:
            print 'could not locate the full dataset'
            # return the rejection on the validation set anyway
            return roc_bkg_rej
        file_full = cv_split_filename[:underscore_idx+1]+'100.pkl'
    else:
        file_full = full_dataset
    # check that this file exists
    if not os.path.isfile(file_full):
        print 'could not locate the full dataset'
        return roc_bkg_rej
    print m.taggers
    evaluateFull(model,m, file_full, transform_valid_weights, weight_validation, job_id+'_full', model.decision_function(X_train), sig_tr_idx, bkg_tr_idx, bkg_rej_train)
        
    return roc_bkg_rej#bkgrej#validation_score


def grid_search(lb_view, model, cv_split_filenames, param_grid, variables, algo, id_tag = 'cv', weighted=True, full_dataset='', compress=True, transform_weights=False, transform_valid_weights=False, weight_validation=False):
    """Launch all grid search evaluation tasks."""
    from sklearn.grid_search import ParameterGrid
    all_tasks = []
    all_parameters = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(all_parameters):
        task_for_params = []
       
        for j, cv_split_filename in enumerate(cv_split_filenames):    
            t = lb_view.apply(
                compute_evaluation, cv_split_filename, model, params, job_id='paramID_'+str(i)+id_tag+'ID_'+str(j), taggers=variables, weighted=weighted,algorithm=algo, full_dataset=full_dataset, compress=compress, transform_weights=transform_weights, transform_valid_weights=transform_valid_weights, weight_validation=weight_validation)
            task_for_params.append(t) 
        
        all_tasks.append(task_for_params)
        
    return all_parameters, all_tasks


def progress(tasks):
    return np.mean([task.ready() for task_group in tasks
                                 for task in task_group])


def find_bests(all_parameters, all_tasks, n_top=5, save=False, bests_tag='cv'):
    """Compute the mean score of the completed tasks"""
    mean_scores = []
    param_id = 0
    for param, task_group in zip(all_parameters, all_tasks):
        scores = [t.get() for t in task_group if t.ready()]
        if len(scores) == 0:
            continue
        mean_scores.append((np.mean(scores), param, param_id))
        param_id+=1
    bests = sorted(mean_scores, reverse=True, key=lambda x: x[0])[:n_top]        
    if save:
        f = open('bests/bests'+bests_tag+'.txt','w')
        for b in bests:
            f.write('mean_score: ' + str(b[0]) + ' params: ' + str(b[1]) + ' param id: ' + str(b[2])+'\n')
        f.close()
    return bests


def cross_validation(data, model, params, iterations, variables, ovwrite=True, ovwrite_full=True,suffix_tag = 'cv', scale=True, onlyFull = False):
    X = data[variables].values
    y = data['label'].values
    w = data['weight'].values

    # find the efficiency for both signal and bkg
    signal_eff = data.loc[data['label']==1]['eff'].values[0]
    bkg_eff = data.loc[data['label']==0]['eff'].values[0]

    # create the cross validation splits and write them to disk
    filenames = persist_cv_splits(X, y, w, variables, n_cv_iter=iterations, name='data', suffix="_"+suffix_tag+"_%03d.pkl", test_size=0.25, scale=scale,random_state=None, overwrite=ovwrite, overwrite_full=ovwrite_full, signal_eff=signal_eff, bkg_eff=bkg_eff, onlyFull=onlyFull)

    return filenames
    
def printProgress(tasks):
    prog = progress(tasks)
    print("Tasks completed: {0}%".format(100 * prog))
    return prog




def runTest(cv_split_filename, model, trainvars, algo, label = 'test', full_dataset='', transform_weights=False, transform_valid_weights=False, weight_validation=False, rfc = False):
    
    from sklearn.ensemble import RandomForestClassifier
    base_estimators = [DecisionTreeClassifier(max_depth=4,min_weight_fraction_leaf=0.01,class_weight="auto",max_features="auto")]#min_weight_fraction_leaf=0.0
    
    

    params = OrderedDict([
            ('base_estimator', base_estimators),
            ('n_estimators', [50]),
            ('learning_rate', [0.3])
            ])
    if rfc:
        model = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
        params = []
    
    from sklearn.grid_search import ParameterGrid
    all_parameters = list(ParameterGrid(params))
    
    for i, params in enumerate(all_parameters):
        compute_evaluation(cv_split_filename, model, params, job_id = label, taggers = trainvars, weighted=True, algorithm=algo, full_dataset=full_dataset, transform_weights=transform_weights, transform_valid_weights=transform_valid_weights, weight_validation=weight_validation)
        #plotSamples(cv_split_filename, trainvars)
        return
'''        
def runTestForest():
    from sklearn.ensemble import RandomForestClassifier
    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f],importances[indices[f]]))

    # plot it
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')
    plt.xticks(range(X_train.shape[1]),
               feat_labels, rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()
        
def runTestRegression():
    from sklearn.linear_model import LogisticRegression
    LogisticRegression(penalty='l1')
    lr = LogisticRegression(penalty='l1', C=0.1)
    lr.fit(X_train_std, y_train)
    print('Training accuracy:', lr.score(X_train_std, y_train))

    print('Test accuracy:', lr.score(X_test_std, y_test))
    print lr.intercept_
    print lr.coef_
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan',
              'magenta', 'yellow', 'black',
              'pink', 'lightgreen', 'lightblue',
              'gray', 'indigo', 'orange']
    weights, params = [], []
    for c in np.arange(-4, 6):
        lr = LogisticRegression(penalty='l1',
                                C=10**c,
                                random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
        weights = np.array(weights)
        for column, color in zip(range(weights.shape[1]), colors):
            plt.plot(params, weights[:, column],
                     label=df_wine.columns[column+1],
                     color=color)
            plt.axhline(0, color='black', linestyle='--', linewidth=3)
            plt.xlim([10**(-5), 10**5])
            plt.ylabel('weight coefficient')
            plt.xlabel('C')
            plt.xscale('log')
            plt.legend(loc='upper left')
            ax.legend(loc='upper center',
                      bbox_to_anchor=(1.38, 1.03),
                      ncol=1, fancybox=True)
            plt.show()

'''    
    
def main(args):
    '''
    Main method which runs the classifier.  Takes in a number of arguments which are then used to control which methods get run -> creating plots, cv splits, correlation plots, grid_search, run test
    '''
    parser = argparse.ArgumentParser(description='Run parts of the MVA tagger.')
    parser.add_argument('-a','--algorithm', default='AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw', help = 'Set the Algorithm (default: AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw)')
    parser.add_argument('--fileid', default='corr_matrix', help = 'File id to use if drawing the correlation matrix.')
    parser.add_argument('--key', default='mc15_v1_2_10_default', help = 'Key to be used for finding the files to plot, or the key that gets used when creating the plots and cv splits. (Default: mc15_v1_2_10_default)')
    parser.add_argument('--fulldataset', default = 'persist/data_DEFAULT_100.pkl', help = 'The name of the full dataset pkl file. This is the one created with the persist_cv method (default: persist/data_mc15_nc_v1_2_10_v1_100.pkl)')
    parser.add_argument('--plotCorrMatrix', dest='plotCorrMatrix',action='store_true', help = 'Plot the correlation matrix. Usually set the fileid parameter at the same time as this.')
    parser.add_argument('--createFoldsOnly', dest='createFoldsOnly',action='store_true', help = 'Create the cv splits only, without running any computation.')
    parser.add_argument('--runTestCase', type=bool, default=False, help = 'Run a test of the BDT with the current set of variables. Useful for checking the setup is correct and for getting the feature_importances for a single run.')
    parser.add_argument('--test-id', dest='testid', default = 'test', help = 'The name of the output files from the test run (default: test)')
    parser.add_argument('--allVars', dest='allVars',action='store_true', help = 'Use all of the variables. This is useful at the beginning when it is not yet clear which variables should be used for training.')
    parser.add_argument('--plotCV', dest='plotCV', action='store_true', help = 'Plot the cv splits and get stats about the variables in the cv splits.')
    parser.add_argument('--testSample', default = 'persist/data_DEFAULT_000.pkl', help = 'The name of the file on which to run the test (see runTestCase option). Default is persist/data_(key)_000.pkl')
    parser.add_argument('--runMVA', type=bool, default=False, help = 'Whether or not to run the full grid search.')
    parser.add_argument('--folds', type=int, default=10, help = 'Number of cv folds to create.')
    # should be using a subparser here, but whatever.
    parser.add_argument('--onlyFull', dest='onlyFull',action='store_true', help = 'If creating folds should only the full dataset be done or not?')
    parser.add_argument('--transform-weights', dest='txweights',action='store_true', help = "if weights must be transformed")
    parser.add_argument('--transform-valid-weights', dest='txvalweights',action='store_true', help = "if validation weights must be transformed")
    parser.add_argument('--weight-validation', dest='weightval',action='store_true', help = "if weights must be applied during validation and testing")
    parser.add_argument('--no-weight-train', dest='weighted',action='store_false', help = "Turn off weighting for training.")
    parser.add_argument('--RFC', dest='rfc', action='store_true', help = 'If running RandomTreesClassifier for the testing.')
    parser.set_defaults(txweights=False)
    parser.set_defaults(txvalweights=False)
    parser.set_defaults(weightval=False)
    parser.set_defaults(weighted=True)
    parser.set_defaults(plotCV=False)
    parser.set_defaults(onlyFull=False)
    parser.set_defaults(plotCorrMatrix=True)
    parser.set_defaults(allVars = False)
    parser.set_defaults(createFoldsOnly=False)
    parser.set_defaults(rfc=False)
    args = parser.parse_args()
    print 'allVars: ' + str(args.allVars)
    model = AdaBoostClassifier()

    #base_estimators = [DecisionTreeClassifier(max_depth=3,min_weight_fraction_leaf=0.01,class_weight="auto",max_features="auto"), DecisionTreeClassifier(max_depth=4,min_weight_fraction_leaf=0.01,class_weight="auto",max_features="auto"), DecisionTreeClassifier(max_depth=5,min_weight_fraction_leaf=0.01,class_weight="auto",max_features="auto")]#, DecisionTreeClassifier(max_depth=6), DecisionTreeClassifier(max_depth=8), DecisionTreeClassifier(max_depth=10),DecisionTreeClassifier(max_depth=15)]
    #base_estimators = [DecisionTreeClassifier(max_depth=3,class_weight="auto"), DecisionTreeClassifier(max_depth=4,class_weight="auto"), DecisionTreeClassifier(max_depth=5,class_weight="auto")]#, DecisionTreeClassifier(max_depth=6), DecisionTreeClassifier(max_depth=8), DecisionTreeClassifier(max_depth=10),DecisionTreeClassifier(max_depth=15)]
    '''
    base_estimators = [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=6), DecisionTreeClassifier(max_depth=8), DecisionTreeClassifier(max_depth=10),DecisionTreeClassifier(max_depth=15)]
    
    params = OrderedDict([
        ('base_estimator', base_estimators),
        ('n_estimators', np.linspace(20, 80, 5, dtype=np.int)),
        ('learning_rate', np.linspace(0.1, 0.3, 3))
    ])
    '''
    # best performing parameteres for jz5_v2 WITHOUT weighting the validation samples
    '''
    params = [{'base_estimator':[DecisionTreeClassifier(max_depth=3)],'n_estimators':[71],'learning_rate':[0.3]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=5)],'n_estimators':[45],'learning_rate':[0.2]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=3)],'n_estimators':[80],'learning_rate':[0.1]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=3)],'n_estimators':[71],'learning_rate':[0.2]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=3)],'n_estimators':[80],'learning_rate':[0.3]}]
    '''
    # best performing parameteres for jz5_v2 WITH weighting the validation samples

    params = [{'base_estimator':[DecisionTreeClassifier(max_depth=5)],'n_estimators':[62],'learning_rate':[0.3]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=4)],'n_estimators':[80],'learning_rate':[0.2]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=5)],'n_estimators':[45],'learning_rate':[0.2]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=4)],'n_estimators':[71],'learning_rate':[0.2]},
              {'base_estimator':[DecisionTreeClassifier(max_depth=5)],'n_estimators':[62],'learning_rate':[0.1]}]

    #{'n_estimators': 20, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
    #            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
    #            min_samples_split=2, min_weight_fraction_leaf=0.0,
    #            random_state=None, splitter='best'), 'learning_rate': 0.70000000000000007} param id: 269

    #algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_matchedM_loose_v2_200_1000_mw'
    #algorithm = ''
    #algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_notcleaned_v1_200_1000_mw'

    # these were used for dc14
    #allvars = ['Aplanarity','ThrustMin','Sphericity','Tau21','ThrustMaj','EEC_C2_1','EEC_C2_2','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','ZCUT12','Tau2','EEC_D2_2','PlanarFlow']# features v1
    #trainvars = ['EEC_C2_1','EEC_C2_2','SPLIT12','Aplanarity','EEC_D2_1','TauWTA2'] # features_l_2_10_v2

    # these are for the mc15 samples
    #allvars = ['Aplanarity','ThrustMin','Sphericity','ThrustMaj','EEC_C2_1','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','ZCUT12','PlanarFlow']# features v1
    # now adding nTracks!
    allvars = ['Aplanarity','ThrustMin','Sphericity','ThrustMaj','EEC_C2_1','Dip12','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','ZCUT12','PlanarFlow']#, 'nTracks']# features v1
    # trainvars for mc15 200-1000 AK10
    #trainvars = ['EEC_C2_1','SPLIT12','Aplanarity','EEC_D2_1','TauWTA2']
    # trainvars for mc15 1000-1500 AK10
    #trainvars = ['EEC_C2_1','SPLIT12','EEC_D2_1','TauWTA2TauWTA1','PlanarFlow']
    # train vars for mc15_jz5_v1
    #trainvars = ['EEC_C2_1','SPLIT12','EEC_D2_1','TauWTA2TauWTA1','PlanarFlow','ZCUT12','Aplanarity']

    
    # if we are running the bdt (or other classifier) with all of the variables
    if args.allVars == True:
        trainvars = allvars
    else:
        # trainvars for mc15_jz5_v2
        #trainvars = ['EEC_C2_1','SPLIT12','EEC_D2_1','TauWTA2TauWTA1','PlanarFlow','Sphericity','Aplanarity']
        # now adding nTracks!
        trainvars = ['EEC_C2_1','SPLIT12','EEC_D2_1','TauWTA2TauWTA1','PlanarFlow','Sphericity','Aplanarity']#, 'nTracks']
        # seeing if removing sphericity and aplanarity will help.  Looking at the distributions there is probably too much overlap.
        # the peaks overlap, but bkg has a long tail.
        #trainvars = ['EEC_C2_1','SPLIT12','EEC_D2_1','TauWTA2TauWTA1','PlanarFlow', 'Sphericity', 'nTracks']
        
    print trainvars
    #key = 'mc15_v1_2_10_v6'
    #key = 'mc15_nc_v1_2_10_v1'

    compress = True
    file_type = 'pbz2' # .pickle or .pbz2

    trainvars_iterations = [trainvars]
    #full_dataset = 'persist/data_features_nc_2_10_v5_100.pkl'
    if args.fulldataset.find('DEFAULT') != -1:
        full_dataset = args.fulldataset.replace('DEFAULT',args.key)
    else:
        full_dataset = args.fulldataset#'persist/data_mc15_nc_v1_2_10_v1_100.pkl'
    print full_dataset
    weight_plots = True
    weight_flag = '_weighted' if weight_plots else ''
    
    
    if False:#plotCorrMatrix:
        filenames = [f for f in os.listdir('persist/') if f.find(args.key) != -1 and f.find('100.')==-1 and f.endswith('pkl')]
        plotCorrelation(filenames, full_dataset, trainvars, key=args.key)

    # perhaps this part should be moved so that it can be run at the same time as the other stuff? in a single call...
    if args.plotCV:
        filenames = [f for f in os.listdir('persist/') if f.find(args.key) != -1 and f.find('100.')==-1 and f.endswith('pkl')]
        for i,f in enumerate(filenames):
            plotSamples(f,full_dataset,trainvars, key = args.key, first_tagger = i == 0, weight_plots = True)
            
        # create the combined stats file for all taggers and all cv splits
        combined_stats = open('fold_stats/combined_stats_'+args.key+weight_flag+'.txt','w')
        combined_stats.write('{0:15}  {1:10} {2:14}{3:10}'.format('Sample','Signal','Background','Total')+'\n')
        with open('fold_stats/event_counts'+weight_flag+'.txt') as infile:
            combined_stats.write(infile.read())
        combined_stats.write('\n')
        combined_stats.write('{0:15}: {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}'.format('Variable','Mean','Std','Mean Sig','Std Sig','Mean Bkg','Std Bkg')+'\n\n')
        for t in trainvars:
            label = t
            if t in label_dict.keys():
                label = label_dict[t]
            combined_stats.write(label+'\n')
            with open('fold_stats/'+t+'_'+args.key+weight_flag+'.txt') as tfile:
                combined_stats.write(tfile.read())
            combined_stats.write('\n')


    if not (args.createFoldsOnly or args.plotCorrMatrix or args.runTestCase or args.runMVA):
        sys.exit()
        
    import pandas as pd
    sampletype = 'merged'
    #data = pd.read_csv('csv/'+args.algorithm+'_merged.csv')
    data = pd.read_csv('csv/'+args.algorithm+'_'+sampletype+'.csv')

    if args.plotCorrMatrix:
        X = data[allvars].values
        corr_matrix = np.corrcoef(X, rowvar=0)
        drawMatrix(args.key, corr_matrix, "Correlation Matrix for full dataset", allvars, len(allvars), file_id= args.fileid, sampletype = sampletype)#"full_allvars_mc15")

    # just create the folds

    if args.createFoldsOnly:
        # we need to add some extra variables that might not get used for training, but we want in there anyway!
        # ideally we don't always want to create 10 folds, esp if we're just looking for the full file only, or just doing some initial exploration of the data
        filenames = cross_validation(data, model, params, args.folds, trainvars, ovwrite=True, ovwrite_full=True,suffix_tag=args.key, scale=False, onlyFull=args.onlyFull)

    # run the test case
    if args.runTestCase:
        #testSample = 'persist/data_mc15_v1_2_10_v6_000.pkl'
        if args.testid == 'test' and args.fileid != '':
            args.testid = args.fileid
        if args.testSample.find('DEFAULT') != -1:
            args.testSample = args.testSample.replace('DEFAULT',args.key)
        print trainvars
        runTest(args.testSample, model, trainvars, args.algorithm, label=args.testid, full_dataset=full_dataset, transform_weights=args.txweights, transform_valid_weights = args.txvalweights, weight_validation=args.weightval, rfc = args.rfc)

    if not args.runMVA:
        sys.exit(0)
        

    # do the full grid search    
    from IPython.parallel import Client

    client = Client()
    #with client[:].sync_imports():
    client[:]['evaluateFull'] = evaluateFull
    lb_view = client.load_balanced_view()


    
    for t in trainvars_iterations:
        filenames = cross_validation(data, model, params, args.folds, t, ovwrite=False, ovwrite_full=False, suffix_tag=args.key, scale=False)
        allparms, alltasks = grid_search(
            lb_view, model, filenames, params, t, args.algorithm, id_tag=args.key, weighted=args.weighted, full_dataset=full_dataset,compress=compress, transform_weights=args.txweights, transform_valid_weights = args.txvalweights, weight_validation=args.weightval)


        prog = printProgress(alltasks)
        while prog < 1:
            time.sleep(10)
            prog = printProgress(alltasks)
            pprint(find_bests(allparms,alltasks))


        pprint(find_bests(allparms,alltasks,len(allparms), True, args.key))


if __name__=='__main__':
    main(sys.argv)
