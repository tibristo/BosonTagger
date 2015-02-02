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
import cPickle as pickle
import massPtFunctions


from AtlasStyle import *
gROOT.SetBatch(True)
STUB = 0
MINX = 2
MAXX = 3
BINS = 4
FN = 5
# store all of the rejection results
totalrejection = []

# these are used for the maximum rejection results
max_rej = 0
maxrejvar = ''
maxrejm_min = 0
maxrejm_max = 0


def getMassWindow(massfile):
    '''
    Get the mass window limits from the input file. Input file will have the lines 'top edge: MAX' and 'bottom edge: MIN' in it somewhere.
    Keyword args:
    massfile -- input file
    
    returns:
    max_mass, min_mass
    '''
    f = open(massfile)
    print massfile
    m_max = 0.0
    m_min = 0.0
    for l in f:
        # find the correct lines
        if l.startswith('top edge'):
            # convert the last element after splitting to a float.
            m_max = float(l.strip().split()[-1])
        elif l.startswith('bottom edge'):
            # convert the last element after splitting to a float.
            m_min = float(l.strip().split()[-1])
    f.close()
    return m_max, m_min

def writePlots(Algorithm, fileid, canv1, canv2, writeROC):
    '''
    Write plots of variables and ROCs to file - png/ pdf
    Keyword args:
    Algorithm -- name of algorithm being used.
    fileid -- file identifier from config file
    canv1 -- TCanvas of variables to be saved
    canv2 -- TCanvas of ROC curve
    writeROC -- Flag indicating if ROC plots should be drawn in addition to variable plots
    '''
    # plot the variables
    canv1.SaveAs('plots/' + Algorithm + fileid + '-Tim2-VariablePlot.pdf')
    # using linux tool 'convert' gives much higher quality .pngs than if you do canv.SaveAs(xyz.png)
    cmd = 'convert -verbose -density 150 -trim plots/' + Algorithm + fileid + '-Tim2-VariablePlot.pdf -quality 100 -sharpen 0x1.0 plots/'+ Algorithm + fileid +'-Tim2-VariablePlot.png'
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    
    if not writeROC:
        return
    #plot the rocs
    canv2.SaveAs('plots/' + Algorithm + fileid + '-Tim2-ROCPlot.pdf')
    cmd = 'convert -verbose -density 150 -trim plots/' +  Algorithm + fileid + '-Tim2-ROCPlot.pdf -quality 100 -sharpen 0x1.0 plots/' +  Algorithm + fileid +'-Tim2-ROCPlot.png'
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()


def writePlotsToROOT(Algorithm, fileid, hist, rocs, rocs_rejpow):
    '''
    Write plots to a ROOT file instead of png/pdf
    Keyword args:
    Algorithm -- Algorithm name
    fileid -- Identifier for output file
    hist -- dictionary of histograms
    rocs -- dictionary of ROCs
    rocs_rejpow -- dictionary of ROCs rejection power
    '''
    fo = TFile.Open('plots/'+Algorithm+fileid+'.root','RECREATE')
    for h in hist.keys():
        if hist[h].Integral() != 0:
            hist[h].Write()
    for r in rocs.keys():
        rocs[r].Write('roc_'+r)
    for r in rocs_rejpow.keys():
        rocs_rejpow[r].Write('bkgrejroc_'+r)
    fo.Close()

def analyse(Algorithm, plotbranches, plotreverselookup,  trees, cutstring, hist, leg1, leg2, fileid, records, ptreweight = True, varpath = "", savePlots = True, mass_min = "", mass_max = "", scaleLumi = 1):
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
    scaleLumi -- luminosity scale factor

    Returns:
    Background rejection at 50% signal efficiency using the ROC curve and variable used to achieve maximum rejection.
    '''

    # open log file
    logfile = open(records,'w')

    # canvas for histogram plots
    canv1 = TCanvas("canv1")
    canv1.Divide(5,5)
    # canvas for ROC curves
    canv2 = TCanvas("canv2")

    tempCanv = TCanvas("temp")
    # reset hists
    for h in hist.keys():
        hist[h].Reset()
    
    global totalrejection
    # dict containing all of the ROC curves
    roc={}
    roc_errUp = {}
    roc_errDo = {}
    # dict containing the bkgRejPower curves
    bkgRejROC = {}
    # bool that is set to false if no ROC curves are drawn - this will happen if any 
    # hist added to the roc is empty
    writeROC = False

    # dictionary holding all of the histograms without a mass cut
    hist_nomw = {}
    saveNoMassWindowPlots = savePlots

    maxrej = 0
    maxrejvar = ''
    #set up the cutstring/ selection to cut on the correct jet masses
    cutstring_mass = cutstring+ " * (jet_" +Algorithm + "_m < " +mass_max+ ")" + " * (jet_" +Algorithm + "_m > " +mass_min+ ") " 
    # loop through the indices and branchnames
    for index, branchname in enumerate(plotbranches):
        # add ROC dictionary entry
        roc[branchname] = TGraph()
        roc[branchname].SetTitle(branchname)
        roc[branchname].SetName('roc_'+branchname)
        # add error rocs as well
        roc_errUp[branchname] = TGraph()
        roc_errUp[branchname].SetTitle(branchname+'_err_up')
        roc_errUp[branchname].SetName('roc_'+branchname+'_err_up')
        roc_errDo[branchname] = TGraph()
        roc_errDo[branchname].SetTitle(branchname+'_err_do')
        roc_errDo[branchname].SetName('roc_'+branchname+'_err_do')
        # add bkg rej power dictionary entry
        bkgRejROC[branchname] = TGraph()
        bkgRejROC[branchname].SetTitle(branchname)
        bkgRejROC[branchname].SetName('bkgrejroc_'+branchname)
        # new canvas
        canv1.cd(index+1)

        # keep the integral when not applying mass window cuts, this
        # allows us to calculate the efficiency of the mass window cut
        signal_eff = 1.0
        bkg_eff = 1.0


        logfile.write('blah')

        # loop through the datatypes: signal and background
        for indexin, datatype in enumerate(trees):
            histname =  datatype + "_" + branchname

            print "plotting " + datatype + branchname
            logfile.write("plotting " + datatype + branchname+"\n")

            # set up the tree.Draw() variable expression for the histogram
            varexp = branchname + '>>' + histname
            minxaxis = hist[histname].GetXaxis().GetXmin()
            maxxaxis = hist[histname].GetXaxis().GetXmax()
            # add the mc_weight and weighted number of events to the selection string
            # also make sure that the variable being plotted is within the bounds specified 
            # in the config file (the limits on the histogram)
            cutstringandweight = '*mc_event_weight*1./NEvents(mc_channel_number)'#*('+ branchname +'>0)'#+str(minxaxis)+')'#*('+ branchname +'<'+str(maxxaxis)+')' 

            # add the cross section and filter efficiency for the background
            
            if datatype == 'bkg': 
                cutstringandweight += '*filter_eff*xs*k_factor'
                hist[histname].SetMarkerStyle(21)
            # filter efficiency for signal
            elif datatype == 'sig':
                cutstringandweight += '*filter_eff'
                # apply pt reweighting to the signal
                if ptreweight:
                    cutstringandweight +='*SignalPtWeight2(jet_CamKt12Truth_pt)'
                # if we don't apply pt reweighting then we can reweight by cross section
                else:
                    cutstringandweight += '*xs'
            
            hist[histname].Sumw2();

            # apply the selection to the tree and store the output in the histogram
            #print cutstringandweight
            trees[datatype].Draw(varexp,cutstring_mass+cutstringandweight)

            # if the histogram is not empty then normalise it 
            mw_int = hist[histname].Integral()

            if hist[histname].Integral() > 0.0:
                if scaleLumi != 1:
                    hist[histname].Scale(scaleLumi);
                else:
                    hist[histname].Scale(1.0/hist[histname].Integral());


            # set up the axes titles and colours/ styles
            hist[histname].SetLineStyle(1); hist[histname].SetFillStyle(0); hist[histname].SetMarkerSize(1);
            if (branchname.find('jet_')!=-1):
                hist[histname].SetXTitle(plotreverselookup[branchname.replace("jet_"+Algorithm,"")])
            else:
                hist[histname].SetXTitle(plotreverselookup[branchname])
            hist[histname].SetYTitle("Normalised Entries")

            #now get the same plot for no mass window cut to get the eff
            hist_full = hist[histname].Clone()
            hist_full.Reset()
            hist_full.SetName(histname+'_full')
            # need to store the variable in this histogram
            varexpfull = branchname + ' >>' + histname+'_full'

            trees[datatype].Draw(varexpfull,cutstring+cutstringandweight+"*(jet_" +Algorithm + "_m < 300*1000)" + " * (jet_" +Algorithm + "_m > 0)")
            # get the integral and normalise
            full_int = hist_full.Integral()
            #save this histogram to the no mass window histo dictionary
            hist_nomw[histname+'_full'] = hist_full
            logfile.write('DEBUG mw_int: ' +str(mw_int)+'\n')
            logfile.write('DEBUG full_int: ' +str(full_int)+'\n')

            if datatype == 'sig':
                if full_int !=0:
                    signal_eff = mw_int/full_int
                else:
                    signal_eff = 0
            else:
                if full_int != 0:
                    bkg_eff = mw_int/full_int
                else:
                    bkg_eff = 0


        #Make ROC Curves before rebinning, but only if neither of the samples are zero
        if (hist["sig_" +branchname].Integral() != 0 and hist["bkg_" +branchname].Integral() != 0):
            MakeROCBen(1, hist["sig_" +branchname], hist["bkg_" +branchname], roc[branchname], bkgRejROC[branchname],signal_eff,bkg_eff, roc_errUp[branchname], roc_errDo[branchname])
            print signal_eff
            print bkg_eff
            writeROC = True



        pX = Double(0.5)
        pY = Double(0.0)

        # find the corresponding bkg rejection for the 50% signal efficiency point from bkg rejection power ROC curve
        # Howeever, if we want the background rejection power for the mass variable we do not want to take 50% as we already have made
        # a cut on the mass to get it to 68%.
        # calculate error on this as well -> deltaX = X*(deltaY/Y)
        eval_roc = 1
        err_up = 1
        err_do = 1
        if not branchname.endswith("_m"):
            eval_roc = roc[branchname].Eval(0.5)
            # get the bin for this point so that we can find the associated error in roc error tgraphs
            #rocBin = fn.findYValue(roc[branchname],Double(0.5), pY, 0.01, True, True)
            bkgrej = 1/(1-eval_roc)
            eval_rocup = roc_errUp[branchname].Eval(0.5)
            eval_rocdo = roc_errDo[branchname].Eval(0.5)
            err_up = bkgrej-1/(1-eval_rocup)
            err_do = bkgrej-1/(1-eval_rocdo)
        else:
            eval_roc = roc[branchname].Eval(0.68)
            bkgrej = 1/(1-roc[branchname].Eval(0.68))
            #rocBin = fn.findYValue(roc[branchname],Double(0.68), pY, 0.01, True, True)
            eval_rocup = roc_errUp[branchname].Eval(0.68)
            eval_rocdo = roc_errDo[branchname].Eval(0.68)
            #err_up = roc_errUp[branchname].GetPoint(rocBin)
            #err_do = roc_errDo[branchname].GetPoint(rocBin)

        if (eval_rocup != 0):
            bkgrej_errUp = bkgrej-1/(1-eval_rocup)
        else:
            bkgrej_errUp = -1
        if (eval_rocdo!= 0 ):
            bkgrej_errDo = bkgrej-1/(1-eval_rocdo)
            #bkgrej_errUp = err_up#bkgrej*(err_up/eval_roc)
            #bkgrej_errDo = err_do#bkgrej*(err_do/eval_roc)
        else:
            bkgrej_errDo = -1

        #print 'DEBUG 1-bkgeff: '+ str(bkgrej)
        #print 'DEBUG roc bkg: ' + str(1/(1-roc[branchname].Eval(0.5)))

        # store a record of all background rejection values
        # want to store only the variable name, not the algorithm name, so string manipulation.  here it is stored as sig_jet_ALGO_variable.
        groups = branchname.split('_')
        j = '_'.join(groups[:2]), '_'.join(groups[2:])
        #totalrejection.append([branchname[branchname.rfind("_")+1:], float(bkgrej)])
        if not j[1] == 'pt':
            totalrejection.append([j[1], float(bkgrej), float(bkgrej_errUp), float(bkgrej_errDo)])

        if bkgrej > maxrej:
            maxrej = bkgrej
            maxrejvar = branchname

        hist['sig_'+branchname].SetFillColor(4); hist['sig_'+branchname].SetLineColor(4); hist['sig_'+branchname].SetMarkerColor(4); 
        hist['bkg_'+branchname].SetFillColor(2); hist['bkg_'+branchname].SetLineColor(2);  hist['bkg_'+branchname].SetMarkerColor(2);  
        if saveNoMassWindowPlots:
            hist_nomw['sig_'+branchname+'_full'].SetFillColor(4); hist_nomw['sig_'+branchname+'_full'].SetLineColor(4); hist_nomw['sig_'+branchname+'_full'].SetMarkerColor(4); 
            hist_nomw['bkg_'+branchname+'_full'].SetFillColor(2); hist_nomw['bkg_'+branchname+'_full'].SetLineColor(2);  hist_nomw['bkg_'+branchname+'_full'].SetMarkerColor(2);  

        leg1.Clear()
        # add legend entries for bkg and signal histograms
        leg1.AddEntry(hist["sig_" + branchname],"W jets","l");    leg1.AddEntry(hist["bkg_" + branchname],"QCD jets","l");
        #hist['sig_' + branchname].Rebin(4)
        #hist['bkg_' + branchname].Rebin(4)
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
            tempCanv.cd()
            p.SetPad(0,0,1,1) # resize
            p.Draw()
            tempCanv.SaveAs(varpath+branchname+".png")
            del p

        # now save "no mass window" plots
        if saveNoMassWindowPlots:
            p = canv1.cd(index+1).Clone() 
            tempCanv.cd()

            p.SetPad(0,0,1,1) # resize
            fn.addLatex(fn.getAlgorithmString(),fn.getAlgorithmSettings(),fn.getPtRange(), fn.getE(), [fn.getNvtxLow(), fn.getNvtx()])
            if (hist_nomw['sig_'+branchname+'_full'].GetMaximum() > hist_nomw['bkg_'+branchname+'_full'].GetMaximum()):
                fn.drawHists(hist_nomw['sig_' + branchname+'_full'], hist_nomw['bkg_' + branchname+'_full'])
            else:
                fn.drawHists(hist_nomw['bkg_' + branchname+'_full'], hist_nomw['sig_' + branchname+'_full'])
            leg1.Draw()

            p.Draw()
            tempCanv.SaveAs(varpath+branchname+"_noMW.png")
            del p

        # plot the ROC curves
        canv2.cd()
        roc[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
        roc[branchname].GetYaxis().SetTitle("1 - Efficiency_{QCD jets}")

        if index==0 and roc[branchname].Integral() != 0:
            roc[branchname].Draw("al")        
        elif roc[branchname].Integral() != 0:
            roc[branchname].SetLineColor(index+2)
            roc[branchname].Draw("same")
        # legend for the roc curve
        leg2.AddEntry(roc[branchname],branchname,"l");
        leg2.Draw()

    # write out canv1 and roc curves on one page/ png each
    if savePlots:
        #write out the plots after cuts
        writePlots(Algorithm, fileid, canv1, canv2, writeROC)
        writePlotsToROOT(Algorithm, fileid, hist, roc, bkgRejROC)
    # close logfile
    logfile.close()

    # return the variable with the maximum background rejection
    return maxrej, maxrejvar



def main(args):
    '''
    Main method which takes in all of the parameters for the tagger and sets up the
    configuration.  All of the histograms are set up here, ready to be filled.
    Main method launches "analyse" which runs over a tagger configuration.
    '''
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
    parser.add_argument('--channelnumber', help = 'RunNumber/ mc_channel_number to use for selection')
    parser.add_argument('--lumi', help = 'Luminosity scale factor')
    parser.add_argument('--massWindowCut', help = 'Whether a mass window cut should be applied')
    parser.add_argument('-v','--version',help = 'Version number')

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
        InputDir = args.inputfile 

    # load ROOT macros for pt reweighting and event weighting - would it make sense
    # to do this in the functions file and then call everything with fn.?
    SetAtlasStyle()
    ROOT.gROOT.LoadMacro("MakeROCBen.C")
    #ROOT.gROOT.LoadMacro("SignalPtWeight.C")
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
        fn.tree = treename
    if treename == '':
        print "specify a tree name in command line args or config file"
        sys.exit()

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
        if args.ptreweighting == 'false' or args.ptreweighting == 'False':
            ptreweight = False

    massWindowCut = False
    if args.massWindowCut.lower() == 'true':
        massWindowCut = True

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

    # set the saveplots option - whether we want to save individual plots for each var
    saveplots = True
    if args.saveplots:
        if args.saveplots == 'false' or args.saveplots == 'False' or args.saveplots == 'off':
            saveplots = False

    # this is if we are making selection on only one channel number
    if not args.channelnumber:
        channelcut = ''
    else:
        channelcut = ' * (mc_channel_number == '+str(args.channelnumber)+')'

    # lumi scaling
    lumi = 1.0
    if not args.lumi:
        lumi = fn.getLumi()

    # default selection string
    cutstring = "(jet_CamKt12Truth_pt > "+str(ptrange[0])+") * (jet_CamKt12Truth_pt < "+str(ptrange[1])+") * (jet_CamKt12Truth_eta >= -1.2) * (jet_CamKt12Truth_eta <= 1.2) " + channelcut

    # set up the input signal file
    signalFile = fn.getSignalFile()
    # set up background file
    backgroundFile = fn.getBackgroundFile()
    # file to use for pt reweighting inputs
    ptweightFile = fn.getPtWeightsFile()
    # get the number of bins to use for pt reweighting from config file
    ptweightBins = fn.getBins()
    ptweightBins = [200,250,300,350,400,450,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1800,2000,2200,2400,2600,2800,3000]
    eventsFileSig = ''
    eventsFileBkg = ''
    massWinFile = ''
    numbins = 50 #default

    # get all of the filenames. Note that signal and background file are not changed
    # if they have been set already.
    signalFile, backgroundFile, eventsFileSig, eventsFileBkg, ptweightFile, massWinFile = fn.getFiles(InputDir, signalFile, backgroundFile, ptweightFile, massWinFile, ptrange)

    # if the number of pt bins is not variable just use the one entry
    #if len(ptweightBins) == 1:
    #    numbins = int(ptweightBins[0])
    #znumbins = 100
    print "***************************************signalFile: " + signalFile

    if ptreweight and ptweightFile == '':
        print 'calculating pt reweighting since no existing file present.'
        #def run(fname, algorithm, treename, ptfile, version='v6'):
        massPtFunctions.run(signalFile, Algorithm, treename, '', 200, 3000, 'v6')
        # check that the pt reweighting calculation was a success, if not, quit.
        if massPtFunctions.success_pt:
            ptweightFile = massPtFunctions.filename_pt
        else:
            print 'pt rweighting file creation failed'
            sys.exit()



    if massWindowCut and massWinFile == '':
        print 'calculating mass window since no existing file present'
        #def run(fname, algorithm, ptlow, pthigh,treename,ptfile):
        massPtFunctions.run(signalFile, Algorithm, treename, ptweightFile, float(ptrange[0])/1000.,float(ptrange[1])/1000., 'v6')
        # check that the mass window calculation was a success, if not, quit.
        if massPtFunctions.success_m:
            massWinFile = massPtFunctions.filename_m
        else:
            print 'mass window calculation was a failure'
            sys.exit()

    # load all of the event weights from the event weighting file
    loadEvents(eventsFileSig)
    loadEvents(eventsFileBkg)
       
    # load all of the pt reweighting from the pt reweighting file
    if len(ptweightBins) <= 1:
        loadweights(ptweightFile,numbins)
    else:
        loadweights(ptweightFile,-1,array('f',ptweightBins))


    for typename in ['sig','bkg']:
        if typename == 'sig':
            filename = signalFile
        else:
            filename = backgroundFile
        # open the files
        files[typename] = TFile(filename)
        # read in the trees
        trees[typename] = files[typename].Get(treename)


    # remove any branches from the config file that are not in the actual input files
    file_branches = fn.getFileBranches(signalFile, fn.getTree())
    fn.pruneBranches(file_branches)

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
    branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']#, 'avgIntPerXing']

    # set up the full branch names for each variable
    if not plotTruth:
        branches.extend(['jet_' + Algorithm + vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == True]) 
        branches.extend([vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == False]) 
    else:
        branches.extend([vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == False]) 
        branches.extend(['jet_' + AlgorithmTruth + vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == True]) 


    # add algorithm names to branches
    plotbranches = ['jet_' + Algorithm + branch for branch in plotbranchstubs if plotjetlookup[branch] == True]
    for branch in plotbranchstubs:
        if not plotjetlookup[branch]:
            plotbranches.append(str(branch))
    #plotbranches += [branch for branch in plotbranchstubs if plotjetlookup[branch] == False]
    #plotbranches.append('avgIntPerXing')
    # flag if variable bin widths are being used - right now not being used anymore, will re-implement
    varBinPt = False

    # dictionary to hold all histograms
    hist = {}

    # at this point change the Algorithm to AlgorithmTruth, since we needed the normal
    # Algorithm name up until now to get pathnames correct.
    if plotTruth:
        Algorithm = AlgorithmTruth

    # set up all of the histograms and names
    for typename in ['sig','bkg']:
        histnamestub = typename + '_jet_' + Algorithm
        #print plotconfig.items()
        for br in plotconfig.keys():
            #print br
            if plotconfig[br][1] == True: # if it is a jet variable
                if plotconfig[br][FN]!='':
                    histname = typename+'_'+plotconfig[br][FN]+'(jet_'+Algorithm+plotconfig[br][STUB]+')'
                    
                else:
                    histname = histnamestub+plotconfig[br][STUB]
                print histname
            else:
                histname = typename+"_"+plotconfig[br][STUB]
                print plotconfig[br][STUB]
            hist_title = br
            hist[histname] = TH1D(histname, hist_title, plotconfig[br][BINS], plotconfig[br][MINX], plotconfig[br][MAXX])
  
    # legends for histograms and roc curves
    leg1 = TLegend(0.8,0.55,0.9,0.65);leg1.SetFillColor(kWhite)
    leg2 = TLegend(0.2,0.2,0.5,0.4);leg2.SetFillColor(kWhite)

    # set up the mass window cuts
    mass_max = 300*1000.;
    mass_min = 0.0;

    if massWindowCut:
        mass_max, mass_min = getMassWindow(massWinFile)
        print 'calc mass window'

    masses = [[mass_min,mass_max]]
    # make sure out optimisation folder exists
    if not os.path.exists('optimisation'):
        os.makedirs('optimisation')
    
    # flag to write out trees into csv format
    writecsv= False

    # store teh maximum background rejection
    global max_rej, maxrejvar, maxrejm_min, maxrejm_max
    max_rej = 0
    maxrejvar = ''
    maxrejm_min = 0
    maxrejm_max = 0
    print plotbranches

    for m in masses:

        m_min = m[0]
        m_max = m[1]

        # log the output
        records = 'TaggerOpt'+Algorithm+'_'+fileid+'_'+str(m_max)+'_'+str(m_min)+'_'+'.out'

        # run the analysis for mass range
        masses = " * (jet_" +Algorithm + "_m < " +str(m_max)+ ")" + " * (jet_" +Algorithm + "_m > " +str(m_min)+ ") " 
        rej,rejvar = analyse(Algorithm, plotbranches, plotreverselookup, trees, cutstring, hist, leg1, leg2, fileid, records, ptreweight, varpath, saveplots, str(m_min), str(m_max), lumi)
        if writecsv:
            fn.writeCSV(signalFile, backgroundFile, branches, cutstring+masses, treename, Algorithm, fileid, [eventsFileSig, eventsFileBkg], ptweightFile, ptweightBins)
        #records.write(str(rej) + ' ' + rejvar + ' ' + str(m_min) + ' ' + str(m_max)+'/n')

        if rej > max_rej:
            max_rej = rej
            maxrejvar = rejvar
            maxrejm_min = m_min
            maxrejm_max = m_max
    #records.close()
    # dump totalrejection in pickle to be read in by the scanBkgrej module which runs this module
    print totalrejection

    if not args.version:
        version = 'v1'
    else:
        version = args.version
    with open("tot_rej_"+version+".p","wb") as f:
        pickle.dump(totalrejection, f)
    # print out all of the info, which is read from stdout by the scanner script.
    # this is not really needed anymore as it was a hack to get around ROOT's annoying
    # global memory management.  This is fixed now by using pickle files.  However,
    # this hasn't been fully implemented everywhere yet, so it will take a little while to remove it.
    print "MAXREJSTART:" +str(max_rej)+","+maxrejvar+","+str(maxrejm_min)+","+str(maxrejm_max)+ "MAXREJEND"
    output = "MAXREJSTART:" +str(max_rej)+","+maxrejvar+","+str(maxrejm_min)+","+str(maxrejm_max)+ "MAXREJEND"
    # dump the output to a pickle file
    with open("TaggerOutput_"+version+".p","wb") as f:
        pickle.dump(output,f)

if __name__ == '__main__':
    #max_rej, maxrejvar, maxrejm_min, maxrejm_max=main(sys.argv)
    main(sys.argv)
    sys.exit()

def runMain(args):
    '''
    Use a list for the arguments that would be used from command line.
    '''
    sys.argv = args
    main(args)


    
