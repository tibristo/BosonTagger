from ROOT import * 
import functions as fn
from numpy import *
import root_numpy
import pandas as pd
import sys
import argparse
import subprocess
import os
from array import array



from AtlasStyle import *
gROOT.SetBatch(True)
STUB = 0
MINX = 1
MAXX = 2

def writePlots(Algorithm, fileid, canv1, canv2):
    '''
    Write plots to file - png/ pdf
    Keyword args:
    Algorithm -- name of algorithm being used.
    fileid -- file identifier from config file
    canv1 -- TCanvas of variables to be saved
    canv2 -- TCanvas of ROC curve
    '''

    canv1.SaveAs('plots/' + Algorithm + fileid + '-Tim2-VariablePlot.pdf')
    # this gives much higher quality .pngs than if you do canv.SaveAs(xyz.png)
    cmd = 'convert -verbose -density 150 -trim plots/' + Algorithm + fileid + '-Tim2-VariablePlot.pdf -quality 100 -sharpen 0x1.0 plots/'+ Algorithm + fileid +'-Tim2-VariablePlot.png'
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

    canv2.SaveAs('plots/' + Algorithm + fileid + '-Tim2-ROCPlot.pdf')
    cmd = 'convert -verbose -density 150 -trim plots/' +  Algorithm + fileid + '-Tim2-ROCPlot.pdf -quality 100 -sharpen 0x1.0 plots/' +  Algorithm + fileid +'-Tim2-ROCPlot.png'
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()


def analyse(Algorithm, plotbranchstubs, canv1, canv2, trees, cutstring, hist, leg1, leg2, fileid, varpath = "", savePlots = True, mass_min = "", mass_max = ""):
    '''
    Run through the Algorithm for a given mass range.  Returns the bkg rej at 50% signal eff.
    Keyword args:
    Algorithm -- Name of the algorithm.  Set in main, comes from config file.
    plotbranchstubs -- the variables to be plotted.  Set in main.
    trees -- contains the actual data.
    cutstring -- basic selection cuts to be applied.
    hist -- Histograms that will be filled/ saved.
    leg1 -- TLegend for histograms
    leg2 -- TLegend for ROC curves
    fileid -- File identifier that gets used in the output file names
    varpath -- output folder to save single variable plots to
    savePlots -- whether or not to write out plots to file
    mass_min -- Mass window minimum
    mass_max -- Mass window maximum

    Returns:
    Background rejection at 50% signal efficiency using the ROC curve and variable used to achieve maximum rejection.
    '''

    # reset hists
    for h in hist.keys():
        hist[h].Reset()
    
    roc={}
    maxrej = 0
    maxrejvar = ''
    cutstring_mass = cutstring+ " * (jet_" +Algorithm + "_m < " +mass_max+ ")" + " * (jet_" +Algorithm + "_m > " +mass_min+ ") " 
    for index, branchname in enumerate(Algorithm + branch for branch in plotbranchstubs):
        roc[branchname] = TGraph()
        canv1.cd(index+1)
        for indexin, datatype in enumerate(trees):
            print "plotting " + datatype + branchname
            histname =  datatype + "_" + branchname
            varexp = 'jet_' + branchname + ' >>' + histname
            cutstringandweight = cutstring_mass +' * mc_event_weight * 1/NEvents(mc_channel_number) '

            if datatype == 'bkg': 
                cutstringandweight += '* filter_eff * xs  '#* k_factor 
                hist[histname].SetMarkerStyle(21)
            elif datatype == 'sig':
                cutstringandweight += '* filter_eff * SignalPtWeight2(jet_CamKt12Truth_pt) '
            hist[histname].Sumw2();
            trees[datatype].Draw(varexp,cutstringandweight)
            if hist[histname].Integral() > 0.0:
                hist[histname].Scale(1.0/hist[histname].Integral());
            
            hist[histname].SetLineStyle(1); hist[histname].SetFillStyle(0); hist[histname].SetMarkerSize(1);
            hist[histname].SetXTitle(plotreverselookup[branchname.replace(Algorithm,"")])
            hist[histname].SetYTitle("Normalised Entries")

        #Make ROC Curves before rebinning 
        MakeROCBen(1, hist["sig_" +branchname], hist["bkg_" +branchname], roc[branchname])


        #eff_sig_bin = roc[branchname].GetXaxis().FindBin(0.5)
        pX = Double(0.5)
        pY = Double(0.0)
        
        eff_sig_bin,pY = fn.findYValue(roc[branchname], pX, pY)
        sigeff = Double(0.5)
        bkgrej = Double(0.0)
        if (eff_sig_bin < 0):
            bkgrej = pY
        else:
            eff_sig_point = roc[branchname].GetPoint(eff_sig_bin, sigeff, bkgrej)

        if bkgrej > maxrej:
            maxrej = bkgrej
            maxrejvar = branchname

        hist['sig_'+branchname].SetFillColor(4); hist['sig_'+branchname].SetLineColor(4); hist['sig_'+branchname].SetMarkerColor(4); 
        hist['bkg_'+branchname].SetFillColor(2); hist['bkg_'+branchname].SetLineColor(2);  hist['bkg_'+branchname].SetMarkerColor(2);  
        leg1.Clear()
        leg1.AddEntry(hist["sig_" + branchname],"W jets","l");    leg1.AddEntry(hist["bkg_" + branchname],"QCD jets","l");
        hist['sig_' + branchname].Rebin(10)
        hist['bkg_' + branchname].Rebin(10)
        if (hist['sig_'+branchname].GetMaximum() > hist['bkg_'+branchname].GetMaximum()):
            fn.drawHists(hist['sig_' + branchname], hist['bkg_' + branchname])
        else:
            fn.drawHists(hist['bkg_' + branchname], hist['sig_' + branchname])
        leg1.Draw()

        fn.addLatex(fn.getAlgorithmString(),fn.getAlgorithmSettings(),ptrange, fn.getE(), [nvtxlow, nvtx])
        if savePlots:
            p = canv1.cd(index+1).Clone()
            tc = TCanvas(branchname)
            p.SetPad(0,0,1,1) # resize
            p.Draw()
            tc.SaveAs(varpath+branchname+".png")

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

    if savePlots:
        writePlots(Algorithm, fileid, canv1, canv2)

    return maxrej, maxrejvar



def main(args):
    parser = argparse.ArgumentParser(description='Plot some variables.')
    parser.add_argument('config', help = 'required config file')
    parser.add_argument('-i','--inputfile',help = 'the input file')
    parser.add_argument('-a', '--algorithm',help='The algorithm: filter100, filter67 or trimmed')
    parser.add_argument('-f','--fileid', help = 'An optional fileid to append to the name of the output file')
    parser.add_argument('--pthigh', help = 'Optional high pT cut in GeV')
    parser.add_argument('--ptlow', help = 'Optional low pT cut in GeV')
    parser.add_argument('--nvtx', help = 'Upper cut on number of primary vertices')
    parser.add_argument('--nvtxlow', help = 'Lower cut on number of primary vertices')

    args = parser.parse_args()

    config_f = ''
    if not args.config:
        print 'Need more args! usage: python TaggerTim.py config [inputfile] [algorithm] [fileid]'
        sys.exit(0)
    else:
        config_f = args.config

    InputDir = ''
    if args.inputfile:
        InputDir = args.inputfile #"/home/tim/boosted_samples/BoostedBosonMerging/TopoTrimmedPtFrac5SmallR30trimmed_8TeV/"
    treename = 'physics'

    SetAtlasStyle()
    ROOT.gROOT.LoadMacro("MakeROCBen.C")
    ROOT.gROOT.LoadMacro("SignalPtWeight.C")
    ROOT.gROOT.LoadMacro("SignalPtWeight2.C")
    ROOT.gROOT.LoadMacro("NEvents.C")

    trees,files,weights,runs = ( {} for i in range(4) ) 

    fn.setweights(weights)
    fn.setrunnumbers(runs)
    # read in config file
    fn.readXML(config_f)

    writecsv= True
    Algorithm = ''
    setTruth = False
    plotTruth = False

    if not args.algorithm:
        Algorithm = fn.getAlgorithm()
    else:
        alg_lower = args.algorithm.lower()
        if alg_lower.find('truth') != -1:
            plotTruth = True
            setTruth = True

    if Algorithm == '':
        print "No algorithm given in command line or config file!"
        sys.exit(0)

    # save each variable plot for a given groomer into a folder
    varpath = 'plots/'+Algorithm

    AlgorithmTruth = 'CamKt12Truth'
    if not setTruth:
        plotTruth = fn.getTruth()
    if plotTruth:
        varpath += 'Truth'

    fileid = ''
    if args.fileid:
        fileid = args.fileid
    else:
        fileid = fn.getFileID()

    varpath += fileid +'/'
    if not os.path.exists(varpath):
        os.makedirs(varpath)

    # define cuts on the pt
    if args.pthigh and args.ptlow:
        ptrange = [float(args.ptlow)*1000., float(args.pthigh)*1000.]
    else:
        ptrange = fn.getPtRange()

    # set cut on the number of primary vertices
    nvtx = 999 # default is no cut, also set to 999 in functions.py as default
    if args.nvtx:
        nvtx = int(args.nvtx)
    else:
        nvtx = int(fn.getNvtx())
    nvtxlow = 0
    if args.nvtxlow:
        nvtxlow = int(args.nvtxlow)
    else:
        nvtxlow = int(fn.getNvtxLow())

    # default
    cutstring = "(jet_CamKt12Truth_pt > "+str(ptrange[0])+") * (jet_CamKt12Truth_pt < "+str(ptrange[1])+") * (jet_CamKt12Truth_eta > -1.2) * (jet_CamKt12Truth_eta < 1.2) * (vxp_n < " +str(nvtx)+ ") * (vxp_n > "+str(nvtxlow)+")"

    # default
    branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']
    AlgBranchStubs = fn.getBranches() 

    if not plotTruth:
        branches.extend(['jet_' + Algorithm + branch for branch in AlgBranchStubs]) 
    else:
        branches.extend(['jet_' + AlgorithmTruth + branch for branch in AlgBranchStubs]) 

    plotconfig = fn.getPlotBranches()

    plotbranchstubs = [item[0] for item in plotconfig.values()]#['_m','_Tau1', '_SPLIT12', '_PlanarFlow','_Tau21','_pt']

    plotconfig['pt'][MINX] = ptrange[0]
    plotconfig['pt'][MAXX] = ptrange[1]

    #create reverse lookup as well, since they shouldn't have duplicate entries this should be okay
    plotreverselookup = {v[0]: k for k, v in plotconfig.items()}

    if plotTruth:
        if '_massdrop' in plotbranchstubs:
            plotbranchstubs.remove('_massdrop')
        if '_yt' in plotbranchstubs:
            plotbranchstubs.remove('_yt')#, '_Angularity']
        if '_TauWTA2TauWTA1' in plotbranchstubs:
            plotbranchstubs.remove('_TauWTA2TauWTA1')
        if '_ZCUT12' in plotbranchstubs:
            plotbranchstubs.remove('_ZCUT12')

    plotbranches = ['jet_' + Algorithm + branch for branch in plotbranchstubs]

    signalFile = fn.getSignalFile()
    backgroundFile = fn.getBackgroundFile()
    ptweightFile = fn.getPtWeightsFile()
    ptweightBins = fn.getBins()
    eventsFileSig = ''
    eventsFileBkg = ''
    numbins = 20 #default

    if len(ptweightBins) == 1:
        numbins = int(ptweightBins[0])

    fileslist = os.listdir(InputDir)
    sigtmp = ''

    for f in fileslist:
        if signalFile == '' and f.endswith("sig.root"):
            signalFile = InputDir+'/'+f
        elif backgroundFile == '' and f.endswith("bkg.root"):
            backgroundFile = InputDir+'/'+f
        if f.endswith("sig.nevents"):
            eventsFileSig = InputDir+'/'+f
        if f.endswith("bkg.nevents"):
            eventsFileBkg = InputDir+'/'+f
        if ptweightFile == '' and f.endswith("ptweights"):
            ptweightFile = InputDir+'/'+f


    for typename in ['sig','bkg']:
        if typename == 'sig':
            filename = signalFile
        else:
            filename = backgroundFile

        files[typename] = TFile(filename)
        trees[typename] = files[typename].Get(treename)
    
        #ROOT numpy not working with The Rest for some reason
        if writecsv == True:
            numpydata = root_numpy.root2array(filename,'physics',branches,cutstring)
            numpydata = pd.DataFrame(numpydata)

            numpydata.rename(columns=lambda x: x.replace('jet_' + Algorithm,''), inplace=True)
            if typename == 'sig': 
                numpydata['label']=1 
                numpydata.to_csv('csv/' + Algorithm + fileid + '-merged.csv')

            else: 
                numpydata['label']=0 
                numpydata.to_csv('csv/' + Algorithm + fileid + '-merged.csv',mode='a',header=False)
                #numpydata.hist(bins=20,grid=False,histtype='step',label=typename)

    loadEvents(eventsFileSig)
    loadEvents(eventsFileBkg)
       
    if len(ptweightBins) <= 1:
        loadweights(ptweightFile,numbins)
    else:
        loadweights(ptweightFile,ptweightBins)

    hist = {}
    #xbins = [100*1000,225*1000,250*1000,275*1000,300*1000,335*1000,350*1000,400*1000,450*1000,500*1000,600*1000,700*1000,800*1000,900*1000,1000*1000]

    varBinPt = False

    if plotTruth:
        Algorithm = AlgorithmTruth


    for typename in ['sig','bkg']:
        histnamestub = typename + '_' + Algorithm
        for br in plotconfig.keys():
            histname = histnamestub+plotconfig[br][STUB]
            hist_title = br
            hist[histname] = TH1D(histname, hist_title, 200, plotconfig[br][MINX], plotconfig[br][MAXX])
  
    canv1 = TCanvas("canv1")
    canv1.Divide(3,3)
    canv2 = TCanvas("canv2")
    canv2.cd()

    leg1 = TLegend(0.8,0.55,0.9,0.65);leg1.SetFillColor(kWhite)
    leg2 = TLegend(0.2,0.2,0.5,0.4);leg2.SetFillColor(kWhite)

    # need to think of a clever way to define the masses... look for best signal efficiency window?
    masses = [[0,1000*1000,]]
    if not os.path.exists('optimisation'):
        os.makedirs('optimisation')
    records = open('TaggerOpt'+Algorithm+'_'+fileid+'.out','w')
    max_rej = 0
    maxrejvar = ''
    maxrejm_min = 0
    maxrejm_max = 0
    for m in masses:
        m_min = m[0]
        m_max = m[1]
        rej,rejvar = analyse(Algorithm, plotbranchstubs, canv1, canv2, trees, cutstring, hist, leg1, leg2, fileid, varpath, False, str(m_min), str(m_max))
        records.write(str(rej) + ' ' + rejvar + ' ' + str(m_min) + ' ' + str(m_max)+'/n')
        if rej > max_rej:
            max_rej = rej
            maxrejvar = rejvar
            maxrejm_min = m_min
            maxrejm_max = m_max
    return max_rej, maxrejvar, maxrejm_min, maxrejm_max

if __name__ == '__main__':
    max_rej, maxrejvar, maxrejm_min, maxrejm_max=main(sys.argv)
    return max_rej, maxrejvar, maxrejm_min, maxrejm_max

