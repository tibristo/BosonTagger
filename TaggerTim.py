from ROOT import * 
from functions import setweights, pTReweight, getFileIDNumber, setrunnumbers
#from numpy import *
#import root_numpy
#import pandas as pd
import sys
import argparse
import subprocess
from array import array

from AtlasStyle import *

parser = argparse.ArgumentParser(description='Plot some variables.')
parser.add_argument('inputfile',help = 'the input file')
parser.add_argument('algorithm',help='The algorithm: filter100, filter67 or trimmed')
parser.add_argument('-f','--fileid', help = 'An optional fileid to append to the name of the output file')
parser.add_argument('-e','--extendedVars', help = 'An optional argument specifying if TauWTA and ZCUT12 should be plotted',type=bool)
parser.add_argument('-s','--subjets', help = 'specify if the subjet variables massdrop and momentum balance should be plotted',type=bool)
args = parser.parse_args()

#InputDir = "/Users/wbhimji/Data/BosonTaggingMVA_ForJoeGregory/Data/v4/"
#InputDir = "/Users/wbhimji/Data/BoostedBosonNtuples/TopoSplitFilteredMu67SmallR0YCut9/"
#algName = 'TopoSplitFilteredMu67SmallR0YCut9'
#algName = 'TopoTrimmedPtFrac5SmallR30'
#InputDir = "/home/tim/boosted_samples/BoostedBosonMerging/TopoSplitFilteredMu67SmallR0YCut9filtered_8TeV/"
if not args.inputfile or not args.algorithm:
    print 'Need more args! usage: python TaggerTim.py filename algorithm [fileid]'
    sys.exit(0)

InputDir = args.inputfile #"/home/tim/boosted_samples/BoostedBosonMerging/TopoTrimmedPtFrac5SmallR30trimmed_8TeV/"
treename = 'physics'
SetAtlasStyle()
ROOT.gROOT.LoadMacro("MakeROCBen.C")
ROOT.gROOT.LoadMacro("SignalPtWeight.C")
ROOT.gROOT.LoadMacro("SignalPtWeight2.C")
ROOT.gROOT.LoadMacro("NEvents.C")




trees,files,weights,runs = ( {} for i in range(4) ) 

setweights(weights)
setrunnumbers(runs)

writecsv= False
Algorithm = ''
alg_lower = args.algorithm.lower()
if alg_lower.find('mu67') != -1:
    Algorithm = 'CamKt12LCTopoSplitFilteredMu67SmallR0YCut9'
elif alg_lower.find('mu100') != -1:
    Algorithm = 'CamKt12LCTopoSplitFilteredMu100SmallR30YCut4'
elif alg_lower.find('trimmed') != -1:
    Algorithm = 'AntiKt10LCTopoTrimmedPtFrac5SmallR30'
else:
    print 'unrecognised algorithm: please choose from filteredmu67, filteredmu100, trimmed'
    sys.exit(0)


AlgorithmTruth = 'CamKt12Truth'
if alg_lower.find('truth') != -1:
    plotTruth = True
else:
    plotTruth = False


fileid = ''
if args.fileid:
    fileid = args.fileid
extendedVars = False
if args.extendedVars:
    extendedVars = args.extendedVars
subjets = False
if args.subjets:
    subjets = args.subjets

#Algorithm = 'AKT10LCTRIM530'
#cutstring = "(leadGroomedIndex != -99) * (jet_CamKt12Truth_pt[leadTruthIndex] > 200000) * (jet_CamKt12Truth_pt[leadTruthIndex] < 350000) * (jet_CamKt12Truth_eta[leadTruthIndex] > -1.2) * (jet_CamKt12Truth_eta[leadTruthIndex] < 1.2)"
#cutstring = "(leadGroomedIndex != -99) * (jet_CamKt12Truth_pt[leadTruthIndex] > 1000000) * (jet_CamKt12Truth_pt[leadTruthIndex] < 1500000) * (jet_CamKt12Truth_eta[leadTruthIndex] > -1.2) * (jet_CamKt12Truth_eta[leadTruthIndex] < 1.2)"
cutstring = "(leadGroomedIndex != -99) * (jet_CamKt12Truth_pt[leadTruthIndex] > 0) * (jet_CamKt12Truth_pt[leadTruthIndex] < 3500000) * (jet_CamKt12Truth_eta[leadTruthIndex] > -1.2) * (jet_CamKt12Truth_eta[leadTruthIndex] < 1.2)"
branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']
AlgBranchStubs = ['_pt','_eta', '_phi', '_m', '_Tau2','_Tau1', '_WIDTH', '_SPLIT12', '_PlanarFlow', '_Angularity']
#'_ZCUT12', '_Dip12', '_DipExcl12', '_ActiveArea', '_VoronoiArea', '_Angularity', '_QW', '_PullMag', '_PullPhi', '_Pull_C00', '_Pull_C01', '_Pull_C10', '_Pull_C11', '_QJetMAvg', '_QJetMVol', '_TJetMAvg', '_TJetMVol', '_Aplanarity', '_Sphericity', '_ThrustMin', '_ThrustMaj', '_FoxWolf20', '_CBeta2', '_JetCharge', '_MassDropSplit', '_MassDropRecl', '_MassRatio', '_ys12']
if not plotTruth:
    branches.extend(['jet_' + Algorithm + branch for branch in AlgBranchStubs]) 
else:
    branches.extend(['jet_' + AlgorithmTruth + branch for branch in AlgBranchStubs]) 
plotbranchstubs = ['_m','_Tau1', '_SPLIT12', '_PlanarFlow','_Tau21','_pt']
if subjets and not plotTruth:
    plotbranchstubs.append('_massdrop')
    plotbranchstubs.append('_yt')#, '_Angularity']
if extendedVars:
    plotbranchstubs.append('_TauWTA2TauWTA1')
    plotbranchstubs.append('_ZCUT12')
    AlgBranchStubs.append('_TauWTA1')
    AlgBranchStubs.append('_TauWTA2')
    AlgBranchStubs.append('_TauWTA1TauWTA2')
    AlgBranchStubs.append('_ZCUT12')
plotbranches = ['jet_' + Algorithm + branch for branch in plotbranchstubs]



print "Reading in files"
loadedweights = False
for typename in ['sig','bkg']:
    if Algorithm.find('Mu67') != -1:
        filename = InputDir + Algorithm.replace('CamKt12LC','') +  "_1_inclusive_" + typename  + ".root"
    elif Algorithm.find('Mu100') != -1:
        filename = InputDir + Algorithm.replace('CamKt12LC','') +  "_2_inclusive_" + typename  + ".root"
    elif Algorithm.find('Trim') != -1:
        filename = InputDir + Algorithm.replace('AntiKt10LC','') + "_3_inclusive_" + typename  + ".root"
        
    if not loadedweights:
        loadweights(filename[:filename.find("inclusive")+9]+".ptweights",20)
        loadedweights = True

    loadEvents(filename[:-5]+".nevents")
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
xbins = [100*1000,225*1000,250*1000,275*1000,300*1000,335*1000,350*1000,400*1000,450*1000,500*1000,600*1000,700*1000,800*1000,900*1000,1000*1000]

varBinPt = False

if plotTruth:
    Algorithm = AlgorithmTruth


for typename in ['sig','bkg']:
    histnamestub = typename + '_' + Algorithm 
    histname = histnamestub + '_m'
    hist[histname] = TH1D(histname,'Mass',200,0.,300.*1000)
    histname = histnamestub + '_SPLIT12'
    hist[histname] = TH1D(histname,'Split12',200,0.,100.*1000)
    histname = histnamestub + '_Tau1'
    hist[histname] = TH1D(histname,'Tau1',200,0.2,0.5)
    histname = histnamestub + '_Tau2'
    hist[histname] = TH1D(histname,'Tau2',200,0.2,0.4)
    histname = histnamestub + '_Tau21'
    hist[histname] = TH1D(histname,'Tau21',200,0.2,1.5)
    histname = histnamestub + '_PlanarFlow'
    hist[histname] = TH1D(histname,'Planar Flow',200,0.2,1.)
    if subjets and not plotTruth:
        histname = histnamestub + '_massdrop'
        hist[histname] = TH1D(histname,'Mass Drop',200,0.1,1.)
        histname = histnamestub + '_yt'
        hist[histname] = TH1D(histname,'Momentum Balance',200,0.2,1.)

    #histname = histnamestub + '_Angularity'
    #hist[histname] = TH1D(histname,'Angularity',200,0.2,1.)
    if extendedVars:
        histname = histnamestub + '_TauWTA2TauWTA1'
        hist[histname] = TH1D(histname,'TauWTA2TauWTA1',200,0.2,1.5)
        histname = histnamestub + '_ZCUT12'
        hist[histname] = TH1D(histname,'ZCUT12',200,0.,0.6)
    if varBinPt:
        histname = histnamestub + '_pt'
        hist[histname] = TH1D(histname,'pt', len(xbins)-1, array('d',xbins))    
    else:
        histname = histnamestub + '_pt'
        hist[histname] = TH1D(histname,'pt',200,0,3500*1000)
  
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
        cutstringandweight = cutstring +' * mc_event_weight * 1/NEvents(mc_channel_number) '
            # +  ' * mc_event_weight ' put back (SignalPtWeight(jet_CamKt12Truth_pt[leadTruthIndex]))
            # eed to put back the weights here fron runnumer
        if datatype == 'bkg': # does nevents only get applied to bkg or bkg and signal?
            cutstringandweight+= '* filter_eff * xs  '#* k_factor 
        elif datatype == 'sig':
            cutstringandweight+= '* filter_eff * SignalPtWeight2(jet_CamKt12Truth_pt) '
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
    if branchname.find('pt') == -1 or not varBinPt:
        hist['sig_' + branchname].Rebin(10)
        hist['bkg_' + branchname].Rebin(10)
    hist['sig_' + branchname].Draw("e")
    hist['bkg_' + branchname].Draw("esame")
    leg1.Draw()
    if branchname.find('pt') != -1:
        pTReweight(hist['sig_'+branchname], hist['bkg_'+branchname], Algorithm+fileid, varBinPt, xbins)
    
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

canv1.SaveAs(Algorithm + fileid + '-Tim2-VariablePlot.pdf')
# this gives much higher quality .pngs than if you do canv.SaveAs(xyz.png)
cmd = 'convert -verbose -density 150 -trim ' + Algorithm + fileid + '-Tim2-VariablePlot.pdf -quality 100 -sharpen 0x1.0 '+ Algorithm + fileid +'-Tim2-VariablePlot.png'
p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.wait()

canv2.SaveAs(Algorithm + fileid + '-Tim2-ROCPlot.pdf')
cmd = 'convert -verbose -density 150 -trim ' + Algorithm + fileid + '-Tim2-ROCPlot.pdf -quality 100 -sharpen 0x1.0 '+ Algorithm + fileid +'-Tim2-ROCPlot.png'
p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.wait()


