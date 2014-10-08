from ROOT import * 
from functions import setweights
#from numpy import *
#import root_numpy
#import pandas as pd

from AtlasStyle import *

#InputDir = "/Users/wbhimji/Data/BosonTaggingMVA_ForJoeGregory/Data/v4/"
InputDir = "/Users/wbhimji/Data/BoostedBosonNtuples/CamKt12LCTopoSplitFilteredMu67SmallR0YCut9/"
treename = 'physics'
SetAtlasStyle()
ROOT.gROOT.LoadMacro("MakeROCBen.C")
ROOT.gROOT.LoadMacro("SignalPtWeight.C")

trees,files,weights = ( {} for i in range(3) ) 

setweights(weights)

writecsv= False
Algorithm = 'CamKt12LCTopoSplitFilteredMu67SmallR0YCut9'
#Algorithm = 'AKT10LCTRIM530'
cutstring = "(leadGroomedIndex != -99) * (jet_CamKt12Truth_pt[leadTruthIndex] > 200000) * (jet_CamKt12Truth_pt[leadTruthIndex] < 350000) * (jet_CamKt12Truth_eta[leadTruthIndex] > -1.2) * (jet_CamKt12Truth_eta[leadTruthIndex] < 1.2)"
branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']
AlgBranchStubs = ['_pt','_eta', '_phi', '_m', '_Tau2','_Tau1', '_WIDTH', '_SPLIT12', '_PlanarFlow']
#'_ZCUT12', '_Dip12', '_DipExcl12', '_ActiveArea', '_VoronoiArea', '_Angularity', '_QW', '_PullMag', '_PullPhi', '_Pull_C00', '_Pull_C01', '_Pull_C10', '_Pull_C11', '_QJetMAvg', '_QJetMVol', '_TJetMAvg', '_TJetMVol', '_Aplanarity', '_Sphericity', '_ThrustMin', '_ThrustMaj', '_FoxWolf20', '_CBeta2', '_JetCharge', '_MassDropSplit', '_MassDropRecl', '_MassRatio', '_ys12']
branches.extend(['jet_' + Algorithm + branch for branch in AlgBranchStubs]) 
plotbranchstubs = ['_m', '_Tau2','_Tau1', '_SPLIT12', '_PlanarFlow']
plotbranches = ['jet_' + Algorithm + branch for branch in plotbranchstubs]

print "Reading in files"
for typename in ['sig','bkg']:
    filename = InputDir + Algorithm + "_1_inclusive_" + typename  + ".root"
    files[typename] = TFile(filename)
    trees[typename] = files[typename].Get(treename)
    
    #ROOT numpy not working with The Rest for some reason
    if writecsv == True:
        numpydata = root_numpy.root2array(filename,treename,branches,cutstring)
        numpydata = pd.DataFrame(numpydata)
        if typename == 'sig': 
            numpydata['label']=1 
        else: 
            numpydata['label']=0 
        numpydata.to_csv(typename+'.csv')
       

#cuts=  [ '(CA12Truth_eta > -1.2)', '(CA12Truth_eta < 1.2)', '(CA12Truth_pt > 200)', '(CA12Truth_pt < 350)']
#cutstring = ' * '.join(cuts)

hist = {}

for typename in ['sig','bkg']:
    histnamestub = typename + '_' + Algorithm 
    histname = histnamestub + '_m'
    hist[histname] = TH1D(histname,'Mass',200,0.,300.*1000)
    histname = histnamestub + '_SPLIT12'
    hist[histname] = TH1D(histname,'Split12',100,0.,100.*1000)
    histname = histnamestub + '_Tau1'
   # hist[histname] = TH1D(histname,'Mass Drop',200,0.1,1.)
    hist[histname] = TH1D(histname,'Tau1',200,0.2,0.5)
    histname = histnamestub + '_Tau2'
    hist[histname] = TH1D(histname,'Tau2',200,0.2,0.5)
#    hist[histname] = TH1D(histname,'Momentum Balance',200,0.2,1.)
    histname = histnamestub + '_PlanarFlow'
    hist[histname] = TH1D(histname,'Planar Flow',200,0.2,1.)
  
canv1 = TCanvas("canv1")
canv1.Divide(3,3)
canv2 = TCanvas("canv2")
canv2.cd()

leg1 = TLegend(0.6,0.7,0.9,0.9);leg1.SetFillColor(kWhite)
leg2 = TLegend(0.2,0.2,0.5,0.4);leg2.SetFillColor(kWhite)

roc={}
for index, branchname in enumerate(Algorithm + branch for branch in plotbranchstubs):
    roc[branchname] = TGraph()
    canv1.cd(index+1)
    for indexin, datatype in enumerate(trees):
        print "plotting " + datatype + branchname
        histname =  datatype + "_" + branchname
        varexp = 'jet_' + branchname + '[leadGroomedIndex] >>' + histname
        cutstringandweight = cutstring
            # +  ' * mc_event_weight ' put back (SignalPtWeight(jet_CamKt12Truth_pt[leadTruthIndex]))
            # eed to put back the weights here fron runnumer
        trees[datatype].Draw(varexp,cutstringandweight)
        hist[histname].Sumw2(); hist[histname].Scale(1.0/hist[histname].Integral());
        hist[histname].SetLineStyle(1); hist[histname].SetFillStyle(3005); hist[histname].SetMarkerSize(0);
        hist[histname].SetXTitle(branchname.replace(Algorithm,""))

#Make ROC Curves before rebinning 
    MakeROCBen(1, hist["sig_" +branchname], hist["bkg_" +branchname], roc[branchname])

    hist['sig_'+branchname].SetFillColor(4); hist['sig_'+branchname].SetLineColor(4); 
    hist['bkg_'+branchname].SetFillColor(2); hist['bkg_'+branchname].SetLineColor(2);  
    leg1.Clear()
    leg1.AddEntry(hist["sig_" + branchname],"W jets","l");    leg1.AddEntry(hist["bkg_" + branchname],"QCD jets","l");
    hist['sig_' + branchname].Rebin(10)
    hist['bkg_' + branchname].Rebin(10)
    hist['sig_' + branchname].Draw("e")
    hist['bkg_' + branchname].Draw("esame")
    leg1.Draw()
    
    canv2.cd()
    if index==0:
        roc[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
        roc[branchname].GetYaxis().SetTitle("1 - Efficiency_{QCD jets}")
        roc[branchname].Draw("al")        
    else:
        roc[branchname].SetLineColor(index+2)
        roc[branchname].Draw("same")
        leg2.AddEntry(roc[branchname],branchname,"l");
        leg2.Draw()

canv1.SaveAs(Algorithm + '-Tim2-VariablePlot.pdf')
canv2.SaveAs(Algorithm + '-Tim2-ROCPlot.pdf')
