from ROOT import * 
from functions import setweights
#from numpy import *
#import root_numpy
from AtlasStyle import *

#InputDir = "/Users/wbhimji/Data/BosonTaggingMVA_ForJoeGregory/Data/v4/"
InputDir = "/home/tim/boosted_samples/BoostedBosonMerging/TopoTrimmedPtFrac5SmallR30trimmed_8TeV/"
InputDir = "/home/tim/boosted_samples/BoostedBosonMerging/TopoSplitFilteredMu67SmallR0YCut9filtered_8TeV/"
#treename = 'NTupLepW'
treename = 'physics'
SetAtlasStyle()
ROOT.gROOT.LoadMacro("MakeROCBen.C")
ROOT.gROOT.LoadMacro("SignalPtWeight.C")

trees,files,weights = ( {} for i in range(3) ) 

setweights(weights)

for typename in weights:
    if typename == 'signal':
        filename = InputDir + typename + ".root"
    else:
        filename = InputDir + "Sherpa_CT10_WmunuMassiveCBPt" + typename + ".root"
    files[typename] = TFile(filename)
    trees[typename] = files[typename].Get(treename)
###
#Algorithm = 'CA12LCSF67'
Algorithm = 'AKT10LCTRIM530'
branches = ['weight_mc', 'CA12Truth_pt', 'CA12Truth_eta']
AlgBranchStubs = ['_pt','_eta', '_phi', '_m', '_Tau2Tau1', '_Width', '_SPLIT12', '_ZCUT12', '_Dip12', '_DipExcl12', '_ActiveArea', '_VoronoiArea', '_PlanarFlow', '_Angularity', '_QW', '_PullMag', '_PullPhi', '_Pull_C00', '_Pull_C01', '_Pull_C10', '_Pull_C11', '_QJetMAvg', '_QJetMVol', '_TJetMAvg', '_TJetMVol', '_Aplanarity', '_Sphericity', '_ThrustMin', '_ThrustMaj', '_FoxWolf20', '_CBeta2', '_JetCharge', '_MassDropSplit', '_MassDropRecl', '_MassRatio', '_ys12']
branches.extend([Algorithm + branch for branch in AlgBranchStubs]) 

for typename in trees:
    trees[typename].SetBranchStatus("*",0)
    for branchname in branches:
        trees[typename].SetBranchStatus(branchname, 1)

plotbranchstubs = ['_m','_SPLIT12','_MassDropSplit','_Tau2Tau1','_ys12','_PlanarFlow']
plotbranches = [Algorithm + branch for branch in plotbranchstubs]

#cuts=  [ '_eta > -1.2)', '_eta < 1.2)', '_pt > 200)', '_pt < 350)']
cuts=  [ '(CA12Truth_eta > -1.2)', '(CA12Truth_eta < 1.2)', '(CA12Truth_pt > 200)', '(CA12Truth_pt < 350)']
#cutstring = ' && '.join(['(' + Algorithm + cut for cut in cuts])
cutstring = ' * '.join(['(' + Algorithm + cut for cut in cuts])
cutstring = ' * '.join(cuts)
print 'Applying cuts: ' + cutstring

h_signal = {}
h_background = {}
#make this use algorithm

h_signal[Algorithm + '_m'] = TH1D('h_signal_' + Algorithm + '_m','Mass',200,0.,300.)
h_background[Algorithm + '_m'] = TH1D('h_background_' + Algorithm + '_m','Mass',200,0.,300.)
h_signal[Algorithm + '_SPLIT12'] = TH1D('h_signal_' + Algorithm + '_SPLIT12','Split12',100,0.,100.)
h_background[Algorithm + '_SPLIT12'] = TH1D('h_background_' + Algorithm + '_SPLIT12','Split12',200,0.,100.)
h_signal[Algorithm + '_MassDropSplit'] = TH1D('h_signal_' + Algorithm + '_MassDropSplit','Mass Drop',200,0.1,1.)
h_background[Algorithm + '_MassDropSplit'] = TH1D('h_background_' + Algorithm + '_MassDropSplit','Mass Drop',200,0.1,1.)
h_signal[Algorithm + '_Tau2Tau1'] = TH1D('h_signal_' + Algorithm + '_Tau2Tau1','Tau2/Tau1',200,0.2,1.)
h_background[Algorithm + '_Tau2Tau1'] = TH1D('h_background_' + Algorithm + '_Tau2Tau1','Tau2/Tau1',200,0.2,1.)
h_signal[Algorithm + '_ys12'] = TH1D('h_signal_' + Algorithm + '_ys12','Momentum Balance',200,0.2,1.)
h_background[Algorithm + '_ys12'] = TH1D('h_background_' + Algorithm + '_ys12','Momentum Balance',200,0.2,1.)
h_signal[Algorithm + '_PlanarFlow'] = TH1D('h_signal_' + Algorithm + '_PlanarFlow','Planar Flow',200,0.2,1.)
h_background[Algorithm + '_PlanarFlow'] = TH1D('h_background_' + Algorithm + '_PlanarFlow','Planar Flow',200,0.2,1.)

canv1 = TCanvas("canv1")
canv1.Divide(3,3)
canv2 = TCanvas("canv2")
canv2.cd()

leg1 = TLegend(0.6,0.7,0.9,0.9);leg1.SetFillColor(kWhite)
leg2 = TLegend(0.2,0.2,0.5,0.4);leg2.SetFillColor(kWhite)

roc={}
for index, branchname in enumerate(plotbranches):
    roc[branchname] = TGraph()
    for indexin, datatype in enumerate(trees):
        canv1.cd(index+1)
        if datatype == 'signal':
            histname = 'h_signal_' + branchname
            varexp = branchname + '>>' + histname
            cutstringandweight = cutstring +  '* (SignalPtWeight(CA12Truth_pt)) * weight_mc * ' + str(weights[datatype])
        else:
            histname = 'h_background_' + branchname
            varexp = branchname + '>>+' + histname
            cutstringandweight = cutstring +  ' * weight_mc * ' + str(weights[datatype])

        trees[datatype].Draw(varexp,cutstringandweight)
    h_signal[branchname].Sumw2(); h_signal[branchname].Scale(1.0/h_signal[branchname].Integral());
    h_signal[branchname].SetFillColor(4); h_signal[branchname].SetLineColor(4); h_signal[branchname].SetLineStyle(1); h_signal[branchname].SetFillStyle(3005); h_signal[branchname].SetMarkerSize(0);
    h_signal[branchname].SetXTitle(branchname)
    h_background[branchname].Sumw2(); h_background[branchname].Scale(1.0/h_background[branchname].Integral());
    h_background[branchname].SetFillColor(2); h_background[branchname].SetLineColor(2);h_background[branchname].SetLineStyle(1); h_background[branchname].SetFillStyle(3005);h_background[branchname].SetMarkerSize(0);
    leg1.Clear()
    leg1.AddEntry(h_signal[branchname],"W jets","l");    leg1.AddEntry(h_background[branchname],"QCD jets","l");
    #    h_background[branchname].Draw("histsame")
    MakeROCBen(1, h_signal[branchname], h_background[branchname], roc[branchname])

    h_signal[branchname].Rebin(10)
    h_background[branchname].Rebin(10)
    h_signal[branchname].Draw("e")
    h_background[branchname].Draw("esame")
    leg1.Draw()
    canv2.cd()
    if index==0:
        roc[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
        roc[branchname].GetYaxis().SetTitle("Efficiency_{QCD jets}")
        roc[branchname].Draw("al")        
    else:
        roc[branchname].SetLineColor(index+2)
        roc[branchname].Draw("same")
    leg2.AddEntry(roc[branchname],branchname,"l");
    leg2.Draw()

canv1.SaveAs(Algorithm + 'VariablePlot.pdf')
canv2.SaveAs(Algorithm + 'ROCPlot.pdf')
