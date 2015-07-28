
class modelEvaluation:
    def __init__(self, fpr, tpr, thresholds, model, params, rejection, feature_importances, job_id, taggers, Algorithm, score, train_file):
        import numpy as np
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.model = model
        self.params = params
        self.rejection = rejection
        self.feature_importances = feature_importances
        self.job_id = job_id
        self.taggers = taggers
        self.Algorithm = Algorithm
        self.score = score
        self.train_file = train_file
        self.max_eff = 1.0
        self.min_eff = 0.0
        
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

    def setProbas(self, probas, signal_idx, bkg_idx):
        '''
        Set the discriminant (here the probability scores) from the BDT and the actual signal
        and background indices.
        '''
        self.discriminant = probas
        self.sig_idx = signal_idx
        self.bkg_idx = bkg_idx
        
    def plotDiscriminant(self, discriminant, signal_idx, bkg_idx, save_disc = True):
        '''
        Plot the discriminants and the resulting ROC curve derived from them.

        Keyword args:
        discriminant --- The score of the BDT (set in the setProbas method)
        signal_idx --- The true indices of all signal events
        bkg_idx ---The true indices of all background events
        save_disc --- Flag indicating if the discriminant plots should be saved.
        '''
        import ROOT as root
        from ROOT import TH2D, TCanvas, TFile, TNamed, TH1F
        import numpy as np
        from root_numpy import fill_hist
        import functions as fn
        import os

        # stop showing plots to screen
        root.gROOT.SetBatch(True)

        if not os.path.exists('ROC'):
            os.makedirs('ROC')
        fo = TFile.Open("ROC/SK"+str(self.job_id)+'.root','RECREATE')

        '''
        info = 'Rejection_power_'+str(self.rejection)
        tn = TNamed(info,info)
        tn.Write()
        '''
        bins = 100
        # when creating the plots do it over the range of all probas (scores)
        discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), bins)

        hist_bkg = TH1F("Background Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))
        hist_sig = TH1F("Signal Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))

        # fill the signal and background histograms
        fill_hist(hist_bkg,discriminant[bkg_idx])
        if hist_bkg.Integral() != 0:
            hist_bkg.Scale(1/hist_bkg.Integral())
        fill_hist(hist_sig,discriminant[signal_idx])
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
        
        self.roc_graph = fn.RocCurve_SingleSided(hist_sig, hist_bkg, 1,1, roc_cut)
        self.roc_graph.Write()
        
        # get teh background rejection power at 50% signal efficiency
        # store the efficiencies first
        self.ROC_sig_efficiency, self.ROC_bkg_rejection = fn.getEfficiencies(self.roc_graph)
        self.bkgRejectionPower()
        # write the roc score as a string to the output file
        rej_string = 'rejection_power_'+str(self.ROC_rej_power_05)
        rej_n = TNamed(rej_string,rej_string)
        rej_n.Write()

        fo.Close()

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

    def bkgRejectionPower(self):
        '''
        Calculate the background rejection power at 50% signal efficiency.  This uses
        the ROC curve calculated in plotDiscriminant from RocCurve_SingleSided call.
        '''        
        import numpy as np
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

        fpr_05 = self.ROC_bkg_rejection[sel][idx]

        self.ROC_rejection_05 = fpr_05#1-fpr_05

        if fpr_05 != 1:
            self.ROC_rej_power_05 = 1/(1-fpr_05)
        else:
            self.ROC_rej_power_05 = -1

        return self.ROC_rej_power_05

    def getRejPower(self):
        return self.ROC_rej_power_05

    def toROOT(self):
        if self.discriminant is not None:
            #self.plotDiscriminant(self.sig_idx, self.bkg_idx, self.discriminant)
            self.plotDiscriminant(self.discriminant, self.sig_idx, self.bkg_idx)
        else:
            return
