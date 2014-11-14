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
MINX = 2
MAXX = 3

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


def analyse(Algorithm, plotbranches, plotreverselookup, canv1, canv2, trees, cutstring, hist, leg1, leg2, fileid, ptreweight = True, varpath = "", savePlots = True, mass_min = "", mass_max = ""):
    '''
    Run through the Algorithm for a given mass range.  Returns the bkg rej at 50% signal eff.
    Keyword args:
    Algorithm -- Name of the algorithm.  Set in main, comes from config file.
    plotbranches -- the variables to be plotted.  Set in main.
    plotreverselookup -- Lookup for an algorithm from a variable stub
    trees -- contains the actual data.
    cutstring -- basic selection cuts to be applied.
    hist -- Histograms that will be filled/ saved.
    leg1 -- TLegend for histograms
    leg2 -- TLegend for ROC curves
    fileid -- File identifier that gets used in the output file names
    ptreweight -- Reweight signal according to pT
    varpath -- output folder to save single variable plots to
    saveplots -- whether or not to write out plots to file
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
    #set up the cutstring/ selection to cut on the correct jet masses
    cutstring_mass = cutstring+ " * (jet_" +Algorithm + "_m < " +mass_max+ ")" + " * (jet_" +Algorithm + "_m > " +mass_min+ ") " 
    # loop through the indices and branchnames
    for index, branchname in enumerate(plotbranches):
        # set up ROC dictionary entry
        roc[branchname] = TGraph()
        # new canvas
        canv1.cd(index+1)
        # loop through the datatypes: signal and background
        for indexin, datatype in enumerate(trees):
            print "plotting " + datatype + branchname
            histname =  datatype + "_" + branchname
            # set up the tree.Draw() variable expression for the histogram
            #varexp = 'jet_' + branchname + ' >>' + histname
            varexp = branchname + ' >>' + histname
            # add the mc_weight and weighted number of events to the selection string
            cutstringandweight = cutstring_mass +' * mc_event_weight * 1/NEvents(mc_channel_number) '

            # add the cross section and filter efficiency for the background
            if datatype == 'bkg': 
                cutstringandweight += '* filter_eff * xs  '#* k_factor 
                hist[histname].SetMarkerStyle(21)
            # filter efficiency for signal
            elif datatype == 'sig':
                cutstringandweight += '* filter_eff '
                # apply pt reweighting to the signal
                if ptreweight:
                    cutstringandweight +=' * SignalPtWeight2(jet_CamKt12Truth_pt) '
                # if we don't apply pt reweighting then we can reweight by cross section
                else:
                    cutstringandweight += ' * xs '
            hist[histname].Sumw2();
            # apply the selection to the tree and store the output in the histogram
            trees[datatype].Draw(varexp,cutstringandweight)
            # if the histogram is not empty then normalise it 
            if hist[histname].Integral() > 0.0:
                hist[histname].Scale(1.0/hist[histname].Integral());
            
            # set up the axes titles and colours/ styles
            hist[histname].SetLineStyle(1); hist[histname].SetFillStyle(0); hist[histname].SetMarkerSize(1);
            if (branchname.find('jet_')!=-1):
                hist[histname].SetXTitle(plotreverselookup[branchname.replace("jet_"+Algorithm,"")])
            else:
                hist[histname].SetXTitle(plotreverselookup[branchname])
            hist[histname].SetYTitle("Normalised Entries")

        #Make ROC Curves before rebinning 
        MakeROCBen(1, hist["sig_" +branchname], hist["bkg_" +branchname], roc[branchname])



        pX = Double(0.5)
        pY = Double(0.0)

        # find the corresponding bkg rejection for the 50% signal efficiency point from ROC curve
        eff_sig_bin,pY = fn.findYValue(roc[branchname], pX, pY)
        sigeff = Double(0.5)
        bkgrej = Double(0.0)
        if (eff_sig_bin < 0):
            bkgrej = pY
        else:
            # Get the point in the ROC
            eff_sig_point = roc[branchname].GetPoint(eff_sig_bin, sigeff, bkgrej)

        if bkgrej > maxrej:
            maxrej = bkgrej
            maxrejvar = branchname

        hist['sig_'+branchname].SetFillColor(4); hist['sig_'+branchname].SetLineColor(4); hist['sig_'+branchname].SetMarkerColor(4); 
        hist['bkg_'+branchname].SetFillColor(2); hist['bkg_'+branchname].SetLineColor(2);  hist['bkg_'+branchname].SetMarkerColor(2);  

        leg1.Clear()
        # add legend entries for bkg and signal histograms
        leg1.AddEntry(hist["sig_" + branchname],"W jets","l");    leg1.AddEntry(hist["bkg_" + branchname],"QCD jets","l");
        #hist['sig_' + branchname].Rebin(10)
        #hist['bkg_' + branchname].Rebin(10)
        # plot the maximum histogram
        if (hist['sig_'+branchname].GetMaximum() > hist['bkg_'+branchname].GetMaximum()):
            fn.drawHists(hist['sig_' + branchname], hist['bkg_' + branchname])
        else:
            fn.drawHists(hist['bkg_' + branchname], hist['sig_' + branchname])
        leg1.Draw()

        # add correctly formatted text to the plot for the ATLAS collab text, energy, etc.
        fn.addLatex(fn.getAlgorithmString(),fn.getAlgorithmSettings(),fn.getPtRange(), fn.getE(), [fn.getNvtxLow(), fn.getNvtx()])
        # save individual plots
        if savePlots:
            p = canv1.cd(index+1).Clone()
            tc = TCanvas(branchname)
            p.SetPad(0,0,1,1) # resize
            p.Draw()
            tc.SaveAs(varpath+branchname+".png")
        # plot the ROC curves
        canv2.cd()
        if index==0:
            roc[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
            roc[branchname].GetYaxis().SetTitle("1 - Efficiency_{QCD jets}")
            roc[branchname].Draw("al")        
        else:
            roc[branchname].SetLineColor(index+2)
            roc[branchname].Draw("same")
        # legend for the roc curve
        leg2.AddEntry(roc[branchname],branchname,"l");
        leg2.Draw()
    # write out canv1 and roc curves on one page/ png each
    if savePlots:
        writePlots(Algorithm, fileid, canv1, canv2)

    # return the variable with the maximum background rejection
    return maxrej, maxrejvar



def main(args):
    # read in and parse all of the command line arguments
    parser = argparse.ArgumentParser(description='Plot some variables.')
    parser.add_argument('config', help = 'required config file')
    parser.add_argument('-i','--inputfile',help = 'the input file')
    parser.add_argument('-a', '--algorithm',help='The algorithm: filter100, filter67 or trimmed')
    parser.add_argument('-f','--fileid', help = 'An optional fileid to append to the name of the output file')
    parser.add_argument('--pthigh', help = 'Optional high pT cut in GeV')
    parser.add_argument('--ptlow', help = 'Optional low pT cut in GeV')
    parser.add_argument('--nvtx', help = 'Upper cut on number of primary vertices')
    parser.add_argument('--nvtxlow', help = 'Lower cut on number of primary vertices')
    parser.add_argument('--ptreweighting', help = 'Apply pT reweighting')
    parser.add_argument('--saveplots', help = 'Apply pT reweighting')
    parser.add_argument('--tree', help = 'Name of tree in input file')

    args = parser.parse_args()

    config_f = ''
    # if no config file is specified the program exits
    if not args.config:
        print 'Need more args! usage: python TaggerTim.py config [-i inputfile] [-a algorithm] [-f fileid] [--pthigh=x] [--ptlow=y] [--nvtx=n] [--nvtxlow=l] [--ptreweighting=true/false] [--saveplots=true/false] [--tree=name]'
        sys.exit(0)
    else:
        config_f = args.config
    # get the input file
    InputDir = ''
    if args.inputfile:
        InputDir = args.inputfile #"/home/tim/boosted_samples/BoostedBosonMerging/TopoTrimmedPtFrac5SmallR30trimmed_8TeV/"

    # load ROOT macros for pt reweighting and event weighting
    SetAtlasStyle()
    ROOT.gROOT.LoadMacro("MakeROCBen.C")
    ROOT.gROOT.LoadMacro("SignalPtWeight.C")
    ROOT.gROOT.LoadMacro("SignalPtWeight2.C")
    ROOT.gROOT.LoadMacro("NEvents.C")

    # declare the dictionarys for trees, input files, weights and run numbers
    trees,files,weights,runs = ( {} for i in range(4) ) 

    # set event weights and run numbers
    fn.setweights(weights)
    fn.setrunnumbers(runs)
    # read in config file
    fn.readXML(config_f)

    # set the treename for the input file
    treename = ''
    if not args.tree:
        treename = fn.getTree()
    else:
        treename = args.tree
    if treename == '':
        print "specify a tree name in command line args or config file"
        sys.exit()

    # flag to write out trees int csv format
    writecsv= True
    # string for algorithm
    Algorithm = ''
    # flags for plotting truth variables
    setTruth = False
    plotTruth = False

    # if the algorithm name is not given in the command line args look for it in the config file
    if not args.algorithm:
        Algorithm = fn.getAlgorithm() # from config file
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

    # truth algorithm
    AlgorithmTruth = 'CamKt12Truth'
    if not setTruth:
        plotTruth = fn.getTruth() # get truth flag from config file if not set in command line args
    if plotTruth:
        varpath += 'Truth'

    # file identifier for the output files
    fileid = ''
    if args.fileid:
        fileid = args.fileid # set from command line
    else:
        fileid = fn.getFileID() # set from config file

    # flag indicating if pt reweighting should be done on signal
    ptreweight = True
    if not args.ptreweighting:
        ptreweight = fn.getPtReweightFlag()
    else:
        if args.ptreweighting == 'false' or args.reweighting == 'False':
            ptreweight = False
    # output path for storing individual variable plots - there are a lot of these so it is 
    # useful to be able to store these in a separate folder
    varpath += fileid +'/'
    if not os.path.exists(varpath):
        os.makedirs(varpath)

    # define cuts on the pt
    if args.pthigh and args.ptlow:
        ptrange = [float(args.ptlow)*1000., float(args.pthigh)*1000.]
        fn.pt_high = float(args.pthigh)*1000
        fn.pt_low = float(args.ptlow)*1000
    else:
        ptrange = fn.getPtRange()

    # set cut on the number of primary vertices
    nvtx = 999 # default is no cut, also set to 999 in functions.py as default
    if args.nvtx:
        nvtx = int(args.nvtx)
    else:
        nvtx = int(fn.getNvtx())
    fn.nvtx = nvtx
    nvtxlow = 0
    if args.nvtxlow:
        nvtxlow = int(args.nvtxlow)
    else:
        nvtxlow = int(fn.getNvtxLow())
    fn.nvtxlow = nvtxlow
    # set the saveplots option
    saveplots = True
    if args.saveplots:
        if args.saveplots == 'false' or args.saveplots == 'False' or args.saveplots == 'off':
            saveplots = False

    # default selection string
    cutstring = "(jet_CamKt12Truth_pt > "+str(ptrange[0])+") * (jet_CamKt12Truth_pt < "+str(ptrange[1])+") * (jet_CamKt12Truth_eta > -1.2) * (jet_CamKt12Truth_eta < 1.2) * (vxp_n < " +str(nvtx)+ ") * (vxp_n > "+str(nvtxlow)+")"


    # configuration for each variable to plot - axis ranges, titles
    plotconfig = fn.getPlotBranches()

    # just the variables like m, pt, etc
    plotbranchstubs = [item[0] for item in plotconfig.values()]

    # set the plotting range for the pt
    plotconfig['pt'][MINX] = ptrange[0]
    plotconfig['pt'][MAXX] = ptrange[1]

    # get extra variables to be used for selection from tree
    # this is a dictionary with [variable]= [stub, jetvariable flag]
    AlgBranchStubs = fn.getBranches() 
    #create reverse lookup as well, since they shouldn't have duplicate entries this should be okay. 
    # this allows looking up the branch name given just the variable pt, m, etc
    plotreverselookup = {v[0]: k for k, v in AlgBranchStubs.items()}
    # keep track of whether or not the plot is a jet variable and needs the algorithm name appended
    plotjetlookup = {v[0]: v[1] for k, v in AlgBranchStubs.items()}


    # default branches to be plotted
    branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']

    # set up the full branch names for each variable
    if not plotTruth:
        branches.extend(['jet_' + Algorithm + vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == True]) 
        branches.extend([vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == False]) 
    else:
        branches.extend([vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == False]) 
        branches.extend(['jet_' + AlgorithmTruth + vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == True]) 


    # remove variables that will not be present for truth
    if plotTruth:
        if '_massdrop' in plotbranchstubs:
            plotbranchstubs.remove('_massdrop')
        if '_yt' in plotbranchstubs:
            plotbranchstubs.remove('_yt')#, '_Angularity']
        if '_TauWTA2TauWTA1' in plotbranchstubs:
            plotbranchstubs.remove('_TauWTA2TauWTA1')
        if '_ZCUT12' in plotbranchstubs:
            plotbranchstubs.remove('_ZCUT12')

    # add algorithm names to branches
    plotbranches = ['jet_' + Algorithm + branch for branch in plotbranchstubs if plotjetlookup[branch] == True]
    plotbranches += [branch for branch in plotbranchstubs if plotjetlookup[branch] == False]

    # set up the input signal file
    signalFile = fn.getSignalFile()
    # set up background file
    backgroundFile = fn.getBackgroundFile()
    # file to use for pt reweighting inputs
    ptweightFile = fn.getPtWeightsFile()
    # get the number of bins to use for pt reweighting from config file
    ptweightBins = fn.getBins()
    eventsFileSig = ''
    eventsFileBkg = ''
    numbins = 20 #default

    # if the number of pt bins is not variable just use the one entry
    if len(ptweightBins) == 1:
        numbins = int(ptweightBins[0])

    # get list of all files in the input directory and filter out different input files
    fileslist = os.listdir(InputDir)
    sigtmp = ''

    for f in fileslist:
        # if teh signal file and background file were not specified in the config file find them in the input directory
        if signalFile == '' and f.endswith("sig.root"):
            signalFile = InputDir+'/'+f
        elif backgroundFile == '' and f.endswith("bkg.root"):
            backgroundFile = InputDir+'/'+f
        # files for event reweighting
        if f.endswith("sig.nevents"):
            eventsFileSig = InputDir+'/'+f
        if f.endswith("bkg.nevents"):
            eventsFileBkg = InputDir+'/'+f
        # if pt reweight file hasn't been set find it in the input folder
        if ptweightFile == '' and f.endswith("ptweights"):
            ptweightFile = InputDir+'/'+f

    # read the signal and background files
    for typename in ['sig','bkg']:
        if typename == 'sig':
            filename = signalFile
        else:
            filename = backgroundFile

        files[typename] = TFile(filename)
        # read in the trees
        trees[typename] = files[typename].Get(treename)
    
        # write the tree out in csv format to use again later
        if writecsv == True:
            numpydata = root_numpy.root2array(filename,treename,branches,cutstring)
            numpydata = pd.DataFrame(numpydata)

            numpydata.rename(columns=lambda x: x.replace('jet_' + Algorithm,''), inplace=True)
            if typename == 'sig': 
                numpydata['label']=1 
                numpydata.to_csv('csv/' + Algorithm + fileid + '-merged.csv')

            else: 
                numpydata['label']=0 
                numpydata.to_csv('csv/' + Algorithm + fileid + '-merged.csv',mode='a',header=False)
                #numpydata.hist(bins=20,grid=False,histtype='step',label=typename)

    # load all of the event weights from the event weighting file
    loadEvents(eventsFileSig)
    loadEvents(eventsFileBkg)
       
    # load all of the pt reweighting from the pt reweighting file
    if len(ptweightBins) <= 1:
        loadweights(ptweightFile,numbins)
    else:
        loadweights(ptweightFile,ptweightBins)

    # dictionary to hold all histograms
    hist = {}

    # flag if variable bin widths are being used - right now not being used anymore, will re-implement
    varBinPt = False

    if plotTruth:
        Algorithm = AlgorithmTruth

    # set up all of the histograms and names
    for typename in ['sig','bkg']:
        histnamestub = typename + '_jet_' + Algorithm
        print plotconfig.items()
        for br in plotconfig.keys():
            #print br
            if plotconfig[br][1] == True: # if it is a jet variable
                histname = histnamestub+plotconfig[br][STUB]
                print histname
            else:
                histname = typename+"_"+plotconfig[br][STUB]
            hist_title = br
            hist[histname] = TH1D(histname, hist_title, 200, plotconfig[br][MINX], plotconfig[br][MAXX])
  
    # canvas for histogram plots
    canv1 = TCanvas("canv1")
    canv1.Divide(3,3)
    # canvas for ROC curves
    canv2 = TCanvas("canv2")
    canv2.cd()

    # legends for histograms and roc curves
    leg1 = TLegend(0.8,0.55,0.9,0.65);leg1.SetFillColor(kWhite)
    leg2 = TLegend(0.2,0.2,0.5,0.4);leg2.SetFillColor(kWhite)

    # need to think of a clever way to define the masses... look for best signal efficiency window?
    masses = [[0,1000*1000,]]
    # make sure out optimisation folder exists
    if not os.path.exists('optimisation'):
        os.makedirs('optimisation')
    # log the output
    records = open('TaggerOpt'+Algorithm+'_'+fileid+'.out','w')
    # store teh maximum background rejection
    max_rej = 0
    maxrejvar = ''
    maxrejm_min = 0
    maxrejm_max = 0
    for m in masses:
        m_min = m[0]
        m_max = m[1]
        # run the analysis for mass range
        rej,rejvar = analyse(Algorithm, plotbranches, plotreverselookup, canv1, canv2, trees, cutstring, hist, leg1, leg2, fileid, ptreweight, varpath, saveplots, str(m_min), str(m_max))
        records.write(str(rej) + ' ' + rejvar + ' ' + str(m_min) + ' ' + str(m_max)+'/n')
        if rej > max_rej:
            max_rej = rej
            maxrejvar = rejvar
            maxrejm_min = m_min
            maxrejm_max = m_max
    records.close()
    return max_rej, maxrejvar, maxrejm_min, maxrejm_max

if __name__ == '__main__':
    max_rej, maxrejvar, maxrejm_min, maxrejm_max=main(sys.argv)
    sys.exit()
    #return max_rej, maxrejvar, maxrejm_min, maxrejm_max

