
class modelEvaluation:
    def __init__(self, fpr, tpr, thresholds, model, params, job_id, taggers, Algorithm, score, train_file, feature_importances=[], decision_function=[], decision_function_sig = [], decision_function_bkg = []): # add in the decision_function

        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.model = model
        self.params = params
        self.feature_importances = feature_importances
        self.job_id = job_id
        self.taggers = taggers
        self.Algorithm = Algorithm
        self.score = score
        self.train_file = train_file
        self.max_eff = 1.0
        self.min_eff = 0.0
        self.ROC_rej_power_05 = -1
        self.sortedFeatures()
        self.sig_eff = 1.0
        self.bkg_eff = 1.0
        self.output_path = 'ROC'
        self.output_prefix = 'SK'
        self.decision_function = decision_function
        self.df_sig_idx = decision_function_sig
        self.df_bkg_idx = decision_function_bkg
        self.ROC_rej_power_05_train = -1
        self.signal_train_events = -1
        self.bkg_train_events = -1

    def setNumberTrainEvents(self, signal=-1, background=-1):
        self.signal_train_events = signal
        self.bkg_train_events = background
        
    def setOutputPath(self, path):
        self.output_path = path

    def setOutputPrefix(self, prefix):
        self.output_prefix = prefix

    def rejFromTPR(self):
        # this is here to calculate the 50% eff from the fpr and tpr
        # find 0.5 tpr
        import numpy as np
        idx = (self.np.abs(self.tpr-0.5)).argmin()
        fpr_05 = self.fpr[idx]
        rej = 1-fpr_05
        if rej != 1:
            bkgrej = 1/(1-rej)
        else:
            bkgrej = -1
        return bkgrej


    def sortedFeatures(self):
        import numpy as np
        feature_importance = self.feature_importances
        # make importances relative to max importance
        #feature_importance = 100.0 * (feature_importance / feature_importance.max())
        self.features_sorted_idx = np.argsort(feature_importance)[::-1]
        # now to access most important feature's name use
        # self.taggers[self.features_sorted_idx[0]]
        #for f in range(len(feature_importances)):
        #    print("%d. feature %s (%f)" % (f + 1, taggers[sorted_idx[f]], feature_importances[sorted_idx[f]]))

        
    def plot(self):
        '''
        Plot the true positive rate against 1- the false positive rate
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        colors = plt.get_cmap('jet')(np.linspace(0, 1.0,combos(len(self.taggers),2) ))
        labelstring = ' And '.join(t for t in self.taggers)

        plt.plot(self.tpr, (1-self.fpr), label=labelstring, color=color)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('1- Background Efficiency')
        plt.xlabel('Signal Efficiency')
        plt.title(self.Algorithm+' ROC Curve')
        plt.legend(loc="lower left",prop={'size':6})
        plt.savefig(str(self.job_id)+'rocmva.pdf')

    def setProbas(self, probas, signal_idx, bkg_idx, weights = None):
        '''
        Set the discriminant (here the probability scores) from the BDT and the actual signal
        and background indices.
        '''
        self.discriminant = probas
        self.sig_idx = signal_idx
        self.bkg_idx = bkg_idx
        self.weights = weights
        
    def plotDiscriminant(self, discriminant, signal_idx, bkg_idx, weights = None, save_disc = True, rejection_power=True):
        '''
        Plot the discriminants and the resulting ROC curve derived from them.

        Keyword args:
        discriminant --- The score of the BDT (set in the setProbas method)
        signal_idx --- The true indices of all signal events
        bkg_idx ---The true indices of all background events
        save_disc --- Flag indicating if the discriminant plots should be saved.
        rejection_power --- Whether or not to calculate bkg power: 1/eff in addtion to 1-eff
        '''
        import ROOT as root
        from ROOT import TH2D, TCanvas, TFile, TNamed, TH1F, TLegend
        import numpy as np
        from root_numpy import fill_hist
        import functions as fn
        import os
        

        # stop showing plots to screen
        root.gROOT.SetBatch(True)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        fo = TFile.Open(self.output_path+"/"+self.output_prefix+str(self.job_id)+'.root','RECREATE')

        bins = 100
        # when creating the plots do it over the range of all probas (scores)
        discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), bins)

        hist_bkg = TH1F("Background Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))
        hist_sig = TH1F("Signal Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))

        # fill the signal and background histograms
        if weights is not None:
            fill_hist(hist_bkg,discriminant[bkg_idx], weights[bkg_idx])
            fill_hist(hist_sig,discriminant[signal_idx], weights[signal_idx])
        else:
            fill_hist(hist_bkg,discriminant[bkg_idx])
            fill_hist(hist_sig,discriminant[signal_idx])
        if hist_bkg.Integral() != 0:
            hist_bkg.Scale(1/hist_bkg.Integral())
        if hist_sig.Integral() != 0:
            hist_sig.Scale(1/hist_sig.Integral())

        hist_sig.SetLineColor(4)
        hist_bkg.SetLineColor(2)
        #hist_sig.SetFillColorAlpha(4, 0.5);
        hist_sig.SetFillStyle(3004)
        #hist_bkg.SetFillColorAlpha(2, 0.5);
        hist_bkg.SetFillStyle(3005)
        hist_sig.Write()
        hist_bkg.Write()
        c = TCanvas()
        leg = TLegend(0.8,0.55,0.9,0.65);leg.SetFillColor(root.kWhite)
        leg.AddEntry(hist_sig, "Signal","l")
        leg.AddEntry(hist_bkg, "Background", "l")
        max_y = max(hist_sig.GetMaximum(), hist_bkg.GetMaximum())
        hist_sig.SetMaximum(max_y*1.2)
        hist_bkg.SetMaximum(max_y*1.2)
        hist_sig.Draw('hist')
        hist_bkg.Draw('histsame')
        c.Write()
        if save_disc == True:
            if not os.path.exists('disc_plots'):
                os.makedirs('disc_plots')
            c.SaveAs('disc_plots/discriminants_'+str(self.job_id)+'.png')

        # before deciding whether to do a left or right cut for the roc curve we have to find the median.
        sig_median = np.median(discriminant[signal_idx])
        bkg_median = np.median(discriminant[bkg_idx])
        if sig_median > bkg_median:
            roc_cut = 'R'
        else:
            roc_cut = 'L'

        # create the single sided roccurve with the code from Sam
        self.roc_graph = fn.RocCurve_SingleSided(hist_sig, hist_bkg, self.sig_eff,self.bkg_eff, roc_cut)
        self.roc_graph.SetName('BackgroundRejection')
        self.roc_graph.SetTitle('BackgroundRejection')
        self.roc_graph.Write()
        
        # get teh background rejection power at 50% signal efficiency
        # store the efficiencies first
        self.ROC_sig_efficiency, self.ROC_bkg_rejection = fn.getEfficiencies(self.roc_graph)
        self.bkgRejectionPower()
        # write the roc score as a string to the output file
        rej_string = 'rejection_power_'+str(self.ROC_rej_power_05)
        rej_n = TNamed(rej_string,rej_string)
        rej_n.Write()


        if rejection_power:
            c.SetLogy()
            self.roc_graph_power = fn.RocCurve_SingleSided(hist_sig, hist_bkg, self.sig_eff,self.bkg_eff, roc_cut, rejection=False)
            c.cd()
            self.roc_graph_power.SetName('BackgroundPower')
            self.roc_graph_power.SetTitle('BackgroundPower')
            self.roc_graph_power.Write()

        # write the decision function to the root file as well, if it is defined.
        if len(self.decision_function) > 0:
            self.decisionFunctionCanvas()
            # add the legends
            leg2 = TLegend(0.8,0.55,0.9,0.65);leg2.SetFillColor(root.kWhite)
            leg2.AddEntry(self.df_sig, "Signal","l")
            leg2.AddEntry(self.df_bkg, "Background", "l")
            # canvas to draw them on
            c2 = TCanvas('Decision Functions')
            self.df_sig.Draw('hist')
            self.df_bkg.Draw('histsame')
            leg2.Draw('same')
            c2.Write()
            # now write the df histograms as well
            self.df_sig.Write()
            self.df_bkg.Write()
            
        self.hist_sig = hist_sig.Clone(); self.hist_sig.SetDirectory(0)
        self.hist_bkg = hist_bkg.Clone(); self.hist_bkg.SetDirectory(0)
        
        fo.Close()

    def setTrainRejection(self, rej):
        self.ROC_rej_power_05_train = rej

    def calculateBkgRej(self, discriminant, signal_idx, bkg_idx, weights=None):
        '''
        This does essentially the same thing as the plotDiscriminant method, except that it does it for
        an arbritrary discriminant and doesn't save the histograms. It just calculates the score.
        '''
        import ROOT as root
        from ROOT import TH2D, TCanvas, TFile, TNamed, TH1F, TLegend
        import numpy as np
        from root_numpy import fill_hist
        import functions as fn
        import os
        

        # stop showing plots to screen
        root.gROOT.SetBatch(True)

        bins = 100
        # when creating the plots do it over the range of all probas (scores)
        discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), bins)

        hist_bkg = TH1F("Background Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))
        hist_sig = TH1F("Signal Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))

        # fill the signal and background histograms
        if weights is not None:
            fill_hist(hist_bkg,discriminant[bkg_idx], weights[bkg_idx])
            fill_hist(hist_sig,discriminant[signal_idx], weights[signal_idx])
        else:
            fill_hist(hist_bkg,discriminant[bkg_idx])
            fill_hist(hist_sig,discriminant[signal_idx])
        if hist_bkg.Integral() != 0:
            hist_bkg.Scale(1/hist_bkg.Integral())
        if hist_sig.Integral() != 0:
            hist_sig.Scale(1/hist_sig.Integral())

        # before deciding whether to do a left or right cut for the roc curve we have to find the median.
        sig_median = np.median(discriminant[signal_idx])
        bkg_median = np.median(discriminant[bkg_idx])
        if sig_median > bkg_median:
            roc_cut = 'R'
        else:
            roc_cut = 'L'
        roc_graph = fn.RocCurve_SingleSided(hist_sig, hist_bkg, self.sig_eff,self.bkg_eff, roc_cut)
        fpr_05 = fn.GetBGRej50(roc_graph)

        if fpr_05 != 1:
            return float(1/(1-fpr_05))

        return -1.0


    def setMaxEff(self, eff):
        '''
        Set the maximum efficiency when calculating the background rejection
        '''
        self.max_eff = eff

    def setMinEff(self, eff):
        '''
        Set the minimum efficiency when calculating the background rejection
        '''
        self.min_eff = eff

    def setSigEff(self, eff):
        '''
        Set the signal efficiency
        '''
        self.sig_eff = eff

    def setBkgEff(self, eff):
        '''
        Set the background efficiency
        '''
        self.bkg_eff = eff

    def bkgRejectionPower(self):
        '''
        Calculate the background rejection power at 50% signal efficiency.  This uses
        the ROC curve calculated in plotDiscriminant from RocCurve_SingleSided call.
        '''        
        import numpy as np
        import functions as fn
        # first check that all of these have been created
        if not hasattr(self, 'ROC_sig_efficiency'):
            # check if the roc_graph has been calculated
            if not hasattr(self, 'roc_graph'):
                self.toROOT()
            else:
                self.ROC_sig_efficiency, self.ROC_bkg_rejection = fn.getEfficiencies(self.roc_graph)

        # only use events that have an efficiency in a given window
        sel = (self.ROC_sig_efficiency >= self.min_eff) & (self.ROC_sig_efficiency <= self.max_eff)
        # find the entry in rejection matrix that corresponds to 50% efficiency
        idx = (np.abs(self.ROC_sig_efficiency[sel]-0.5)).argmin()

        # in order to have consistency between this, Tagger and AGILEPack
        # we will now use the GetBGRej50() method from Sam.
        fpr_05 = fn.GetBGRej50(self.roc_graph) #self.ROC_bkg_rejection[sel][idx]
        self.ROC_rejection_05 = fpr_05

        if fpr_05 != 1:
            self.ROC_rej_power_05 = 1/(1-fpr_05)
        else:
            self.ROC_rej_power_05 = -1

        return self.ROC_rej_power_05

    def decisionFunctionCanvas(self):
        '''
        Create two histograms which are then drawn onto the same canvas.

        This is only really defined for the BDT, not AGILE NN since that doesn't
        give a "score".
        '''
        import ROOT as root
        from ROOT import TH2D, TCanvas, TFile, TNamed, TH1F, TLegend
        import numpy as np
        from root_numpy import fill_hist
        import functions as fn
        import os
        # check that the decision function output was set
        if len(self.decision_function) == 0:
            return False
        df_sig = TH1F("Signal Decision Function", "Score", 100, -1.0, 1.0)
        df_bkg = TH1F("Background Decision Function", "Score", 100, -1.0, 1.0)
        # fill the histograms with the df
        fill_hist(df_sig,self.decision_function[self.df_sig_idx])
        fill_hist(df_bkg,self.decision_function[self.df_bkg_idx])
        # normalise
        if df_sig.Integral() != 0:
            df_sig.Scale(1./df_sig.Integral())
        if df_bkg.Integral() != 0:
            df_bkg.Scale(1./df_bkg.Integral())
        
        # set up drawing options and colours
        df_sig.SetLineColor(4); df_sig.SetFillStyle(3004)
        df_bkg.SetLineColor(2); df_bkg.SetFillStyle(3005)
        # set the y axis
        max_y = max(df_sig.GetMaximum(), df_bkg.GetMaximum())
        df_sig.SetMaximum(max_y*1.2)
        df_bkg.SetMaximum(max_y*1.2)
        # clone these things
        self.df_sig = df_sig.Clone(); self.df_sig.SetDirectory(0)
        self.df_bkg = df_bkg.Clone(); self.df_bkg.SetDirectory(0)
        return True

            
    def plotDecisionFunction(self):
        import numpy as np
        import matplotlib.pyplot as plt
        # check that the decision function output was set
        if len(self.decision_function) == 0:
            return False
        # Plot the two-class decision scores
        plot_colors = "br"
        plot_step = 0.02
        class_names = ["Signal","Background"]

        plt.figure(figsize=(10, 5))
        # get the range
        plot_range = (self.decision_function.min(), self.decision_function.max())
        # fill in the histogram
        for i, n, c in zip([self.df_bkg_idx, self.df_sig_idx], class_names, plot_colors):
                plt.hist(self.decision_function[i],
                                      bins=10,
                                      range=plot_range,
                                      facecolor=c,
                                      label='Class %s' % n,
                                      alpha=.5)
        x1, x2, y1, y2 = plt.axis()
        # set the axis ranges
        plt.axis((x1, x2, y1, y2 * 1.2))
        plt.legend(loc='upper right')
        plt.ylabel('Samples')
        plt.xlabel('Score')
        plt.title('Decision Scores')
        plt.savefig('disc_plots/'+str(self.job_id)+'decision_function.pdf')
        #plt.show()
        return True

    def setScores(self, sample ,accuracy=-1.0, recall = -1.0, precision = -1.0, f1 = -1.0, cut=-1.0):
        '''
        All different metric evaluations.
        results relative to the true category.
        \[ {\rm accuracy} \equiv \frac{\rm correct~labels}{\rm total~samples} \]
        \[ {\rm precision} \equiv \frac{\rm true~positives}{\rm true~positives + false~positives} \]
        \[ {\rm recall} \equiv \frac{\rm true~positives}{\rm true~positives + false~negatives} \]
        \[ F_1 \equiv 2 \frac{\rm precision \cdot recall}{\rm precision + recall} \]
        The accuracy, precision, recall, and f1-score all range from 0 to 1, with 1 being optimal. Here we've used the following definitions:
        True Positives are those which are labeled 1 which are actually 1
        False Positives are those which are labeled 1 which are actually 0
        True Negatives are those which are labeled 0 which are actually 0
        False Negatives are those which are labeled 0 which are actually 1
        '''
        if sample == 'test':
            self.test_accuracy = accuracy
            self.test_recall = recall
            self.test_precision = precision
            self.test_f1 = f1
            self.test_cut = cut
        elif sample == 'train':
            self.train_accuracy = accuracy
            self.train_recall = recall
            self.train_precision = precision
            self.train_f1 = f1
            self.train_cut = cut


    def getRejPower(self):
        return self.ROC_rej_power_05

    def toROOT(self):
        if self.discriminant is not None:
            #self.plotDiscriminant(self.sig_idx, self.bkg_idx, self.discriminant)
            self.plotDiscriminant(self.discriminant, self.sig_idx, self.bkg_idx, self.weights)
        else:
            return
