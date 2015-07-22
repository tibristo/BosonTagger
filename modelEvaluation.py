
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
        
    def plot(self):
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
        self.discriminant = probas
        self.sig_idx = signal_idx
        self.bkg_idx = bkg_idx
        
    def plotDiscriminant(self, discriminant, signal_idx, bkg_idx):
        import ROOT as root
        from ROOT import TH2D, TCanvas, TFile, TNamed, TH1F
        import numpy as np
        from root_numpy import fill_hist
        import functions as fn

        root.gROOT.SetBatch(True)
        matrix = np.vstack((self.tpr, 1-self.fpr)).T.astype(float)

        labelstring = ' And '.join(t for t in self.taggers)
        hist = TH2D(self.Algorithm,labelstring, 100, 0, 1, 100, 0, 1)

        fill_hist(hist, matrix)

        fo = TFile.Open("ROC/SK"+str(self.job_id)+'.root','RECREATE')
        hist.Write()

        info = 'Rejection_power_'+str(self.rejection)
        tn = TNamed(info,info)
        tn.Write()
        bins = 100
        discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), bins)

        hist_bkg = TH1F("Background Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))
        hist_sig = TH1F("Signal Discriminant","Discriminant",bins, np.min(discriminant), np.max(discriminant))
        discr_rej = 1-discriminant[bkg_idx]
        # TODO: 27/06/2015 right now this is not working!!!!
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

        # before deciding whether to do a left or right cut for the roc curve we have to find the median.
        sig_median = np.median(discriminant[signal_idx])
        bkg_median = np.median(discriminant[bkg_idx])

        if sig_median > bkg_median:
            roc_cut = 'R'
        else:
            roc_cut = 'L'
        
        roc_graph = fn.RocCurve_SingleSided(hist_sig, hist_bkg, 1,1, roc_cut)
        roc_graph.Write()
        
        fo.Close()


    def toROOT(self):
        if self.discriminant is not None:
            #self.plotDiscriminant(self.sig_idx, self.bkg_idx, self.discriminant)
            self.plotDiscriminant(self.discriminant, self.sig_idx, self.bkg_idx)
        else:
            return
