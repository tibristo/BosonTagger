from ROOT import *
import os
from numpy import *
import root_numpy
import pandas as pd
#from AtlasStyle import *

def pTReweight(hist_sig, hist_bkg, algorithm, varBinPt, xbins):
    import ROOT
    from array import array
    bins = hist_bkg.GetNbinsX()
    name = 'ptreweight'+algorithm
    if not varBinPt:
        hist_reweight = ROOT.TH1D(name,name, bins, 0, hist_bkg.GetXaxis().GetBinUpEdge(bins))
    else:
        hist_reweight = ROOT.TH1D(name,name, len(xbins)-1, array('d',xbins))
    for x in range(1,bins):
        if hist_sig.GetBinContent(x) == 0:
            weight = -1
        else:
            weight = hist_bkg.GetBinContent(x)/hist_sig.GetBinContent(x)
        hist_reweight.SetBinContent(x,weight)
        #print str(hist_sig.GetXaxis().GetBinLowEdge(x)/1000) + ' - ' + str(hist_sig.GetXaxis().GetBinUpEdge(x)/1000) + ' weight: ' + str(weight)
    tc = ROOT.TCanvas("ptr")
    hist_reweight.Draw('e')
    tc.SaveAs('pt_reweight'+algorithm+'.png')

def getFileBranches(inputfile, treename='physics'):
    file_branches = []
    file_branches_stubs = []
    file_in = TFile(inputfile)
    print inputfile
    print treename
    physics = file_in.Get(treename)
    ob = physics.GetListOfBranches()
    for tb in ob:
        name = tb.GetName()
        namespl = name.split('_')
        # normally it will be jet_ALGORITHM_variable, so 3 parts
        if len(namespl) == 3:
        #file_branches.append(name)
        # get the variable name
            file_branches.append(namespl[2])#name[name.rfind('_')+1:])
        # sometimes it is jet_ALGORITHM_VAR_IABLE
        elif len(namespl) > 3:
            app_str = ''
            # add almost all
            for i in range(2,len(namespl)-1):
                app_str+=namespl[i]+'_'
            # last one doesn't have _ at the end
            app_str+=namespl[len(namespl)-1]
            file_branches.append(app_str)
        elif len(namespl) == 1: # stuff like averageIntPerXing, for example
            file_branches.append(namespl[0])
    file_in.Close()
    return file_branches



tree = ''
plotbranches = {}
branches = {}
pt_high = 3500000
pt_low = 0
algorithm = ''
truth = False
bins = []
signal_file = ''
background_file = ''
ptweights_file = ''
fileid = ''
algorithmString = ''
algorithmSettings = ''
energy = ''
nvtx = 999
nvtxlow = 0
ptreweightflag = True
lumi = 1.0


def getPlotBranches():
    return plotbranches
def getBranches():
    return branches
def getPtRange():
    return [pt_low, pt_high]
def getAlgorithm():
    return algorithm
def getAlgorithmString():
    return algorithmString
def getAlgorithmSettings():
    return algorithmSettings
def getTruth():
    return truth
def getBins():
    return bins
def getSignalFile():
    return signal_file
def getBackgroundFile():
    return background_file
def getPtWeightsFile():
    return ptweights_file
def getFileID():
    return fileid
def getE():
    return energy
def getNvtx():
    return nvtx
def getNvtxLow():
    return nvtxlow
def getTree():
    return tree
def getPtReweightFlag():
    return ptreweightflag
def getLumi():
    return lumi

def pruneBranches(file_branches):
    global plotbranches
    global branches
    todelete = []
    print file_branches
    # if we have split12 and jet mass we can calculate YFilt
    # so look for these two variables (mass is always there), and if they are there, YFilt can be added
    if 'SPLIT12' in file_branches:
        file_branches.append('YFilt')
    # loop through all of the variables in the branches dict
    for k in branches.keys():
        # if the stub value (take away the leading underscore) is not in the file
        # mark it to be deleted
        if not branches[k][0][1:] in file_branches and not branches[k][0] in file_branches:

            todelete.append(k)

    for d in todelete:
        del branches[d]

    todeleteplot = []
    for k in plotbranches.keys():
        if not plotbranches[k][0][1:] in file_branches and not plotbranches[k][0] in file_branches:
            todeleteplot.append(k)
    for d in todeleteplot:
        del plotbranches[d]


def readXML(configfile):
    """Read in the variable names and histogram limits froms the config xml file."""
    # TODO: add try catch statements incase values are not well defined (for example floats being ill-defined as a string)
 
    import xml.etree.ElementTree as ET
    xmlTree = ET.parse(configfile)
    root = xmlTree.getroot()


    global tree
    for t in root.findall('treename'):
        tree = t.get('name')

    varName = ''
    global plotbranches
    global branches
    global file_branches

    for child in root.findall('varName'):
        varName = child.get('name')
        stub = child.find('stub').text
        jetVariable = True
        function = ''
        if child.find('jetVariable') is not None:
            if child.find('jetVariable').text == "False":
                jetVariable = False
                print varName
        if child.find('function') is not None:
            function = child.find('function').text
        #if function!='':
        #    varName = function+'('+varName+')'
        branches[varName] = [stub, jetVariable, function]
        if child.find('plot').text == "True":
            maxV = float(child.find('maxValue').text)
            minV = float(child.find('minValue').text)
            nbins = 100
            if child.find('bins') is not None:
                nbins = int(child.find('bins').text)
            plotbranches[varName] = [stub,jetVariable,minV,maxV,nbins,function]

    global pt_high
    global pt_low
    for cstring in root.findall('cutstring'):
        for x in list(cstring):
            if x.tag == 'pt_high':
                pt_high = float(x.get('name'))
            elif x.tag == 'pt_low':
                pt_low = float(x.get('name'))
    
    global algorithm 
    global algorithmString
    global algorithmSettings
    global energy
    for algo in root.findall('Algorithm'):
        algorithm = algo.get('name')
        aset = algo.find('AlgorithmSettings')
        astr = algo.find('AlgorithmString')
        aE = algo.find('Energy')
        if not astr is None:
            algorithmString = algo.find('AlgorithmString').text
        if not aset is None:
            algorithmSettings = algo.find('AlgorithmSettings').text
        if not aE is None:
            energy = algo.find('Energy').text
        else:
            energy = '13'

    global truth
    for tr in root.findall('plotTruth'):
        if tr.get('name') == 'True':
            truth = True
        
        
    global bins
    for ptw in root.findall('ptBins'):
        for val in list(ptw):
            if val.tag == 'bin':
                bins.append(float(val.get('name')))
            
    global signal_file
    for folder in root.findall('signal'):
        signal = folder.get('name')

    global bkg_file
    for folder in root.findall('background'):
        bkg_file = folder.get('name')

    global ptweights_file
    for folder in root.findall('ptweights'):
        ptweights_file = folder.get('name')

    global fileid
    for f in root.findall('fileid'):
        fileid = f.get('name')

    global nvtx
    global nvtxlow
    for vxp in root.findall('nvtx'):
        for v in list(vxp):
            if v.tag == 'high':
                nvtx = int(v.text)
            elif v.tag == 'low':
                nvtxlow = int(v.text)

    global lumi
    for l in root.findall('lumi'):
        lumi = float(l.get('name'))

def addLatex(algo, algosettings, ptrange, E, nvtxrange):
    '''
    Method to add Latex text to a plot.  The canvas is already set before this method
    is called, so that when this is run it is already on that canvas.
    
    Key args:
    algo --- Algorithm name
    algosettings --- Any additional settings that were used.
    ptrange --- The pT range that this is plotted in.
    E --- Energy of the simulations
    nvtxrange --- The cuts on the number of vertices.
    '''
    from ROOT import TLatex 
    texw = TLatex();
    texw.SetNDC();
    texw.SetTextSize(0.035);
    texw.SetTextFont(72);
    texw.DrawLatex(0.58,0.88,"ATLAS");
    
    p = TLatex();
    p.SetNDC();
    p.SetTextFont(42);
    p.SetTextSize(0.035);
    p.SetTextColor(ROOT.kBlack);
    p.DrawLatex(0.66,0.88,"Simulation Work in Progress");#"Internal Simulation");

    p = TLatex();
    p.SetNDC();
    p.SetTextFont(42);
    p.SetTextSize(0.032);
    p.SetTextColor(ROOT.kBlack);
    p.DrawLatex(0.65,0.64,"#sqrt{s} = "+str(E)+" TeV");
    
    p2 = TLatex();
    p2.SetNDC();
    p2.SetTextFont(42);
    p2.SetTextSize(0.032);
    p2.SetTextColor(ROOT.kBlack);
    p2.DrawLatex(0.65,0.82,algo);
    
    p2 = TLatex();
    p2.SetNDC();
    p2.SetTextFont(42);
    p2.SetTextSize(0.032);
    p2.SetTextColor(ROOT.kBlack);
    p2.DrawLatex(0.65,0.76,algosettings);
    
    p3 = TLatex();
    p3.SetNDC();
    p3.SetTextFont(42);
    p3.SetTextSize(0.032);
    p3.SetTextColor(ROOT.kBlack);
    p3.DrawLatex(0.65,0.70,str(ptrange[0]/1000.0)+' < p_{T} (GeV) < ' + str(ptrange[1]/1000))#, '+str(nvtxrange[0])+'<nvtx<'+str(nvtxrange[1]));

def drawHists(hist1, hist2):
    '''
    Draw two histograms on the same canvas (with errors).  The canvas must be created before this method is called.
    '''

    hist1.SetMaximum(hist1.GetMaximum()*1.2)
    hist1.Draw("e")
    hist1.Draw("hist same")
    hist2.Draw("e same")
    hist2.Draw("hist same")




def findYValue(pGraph, pX, pY, Epsilon=0.01, pInterpolate=True, pWarn=True):
    # adapted from here: http://www.desy.de/~hperrey/root/GraphRoutines.h.html
    import ROOT
    import copy
    # finds the Y value in a graph corresponding to a given X value as closely as possible
    # returns negative point number, if the point is not within a certain percentage of x
    # (percentage determined by Epsilon parameter)
    # uses the Eval() method of the tgraph to interpolate (linearly) between data points
    # (after the existance of data points in the region has been veryfied)
    PointNumber =0
    x = ROOT.Double(0)
    y = ROOT.Double(0)
    pGraph.GetPoint(0,x,y);
    delta = ROOT.TMath.Abs(pX-x);
    for i in range (1,pGraph.GetN()):
        # loop over points
        x1 = ROOT.Double()
        y1 = ROOT.Double()
        pGraph.GetPoint(i,x1,y1);
        #print 'entry: ' + str(i)
        #print x1
        #print y1
        # check if this points is closer to the x value we are looking for
        if (ROOT.TMath.Abs(pX-x1)<delta):
            # remember point
            delta = ROOT.TMath.Abs(pX-x1)
            x = copy.deepcopy(x1);y=copy.deepcopy(y1);PointNumber=copy.deepcopy(i)
    # warn if delta exceeds a certain percentage of x
    if ((ROOT.TMath.Abs(pX - Epsilon)) < ROOT.TMath.Abs(delta)):
            if (pWarn):
                print " Warning: Requested Y Value for point " +str(pX)
                print "          but only found point at " + str(x)
                print " (difference of " + str( ROOT.TMath.Abs(delta/ pX)) + "%)"
            # return the closest point
            #PointNumber = -1 * PointNumber;
            #Py = 0.0
   # ok, we have data points in the region of interest (or warned the user otherwise), now interpolate
    if (pInterpolate):
            # we need to sort the graph in order to use the Eval() method.
            # better use a copy before messing around with the graph...
            Copy = copy.deepcopy(pGraph)#.Clone();
            Copy.Sort();
            y = Copy.Eval(pX);
            Copy.Delete()
    pY = copy.deepcopy(y)
    return PointNumber, pY


def getFiles(InputDir, signalFile, backgroundFile, massWinFile, ptrange):
    '''
    This method traverses the input directory searching for the signal and background files, mass window file and the events files for signal and background.
    If any of the variables have already been set before running this method they will not
    be reset here again.
    Keyword args:
    InputDir --- The input directory of the algorithm being run.
    signal/backgroundFile --- The input sig/bkg root files.
    massWinFile --- File with mass window cuts.
    ptrange --- The low and high pt cuts.
    '''
    # get a list of the files in the input directory and search for the needed files
    fileslist = os.listdir(InputDir)
    sigtmp = ''
    eventsFileSig = ''
    eventsFileBkg = ''

    for f in fileslist:
        # if teh signal file and background file were not specified in the config file find them in the input directory
        if signalFile == '' and f.endswith("sig.root"):#"sig2.root"):
            signalFile = InputDir+'/'+f
        elif backgroundFile == '' and f.endswith("bkg.root"):#("bkg2.root"):
            backgroundFile = InputDir+'/'+f
        # files for event reweighting
        if f.endswith("sig.nevents"):
            eventsFileSig = InputDir+'/'+f
        if f.endswith("bkg.nevents"):
            eventsFileBkg = InputDir+'/'+f
        # the mass windows have been calculated. saved as
        # Algorithm_masswindow.out
        if massWinFile == '' and f.endswith('masswindow.out'):
            if f.find('pt') == -1:
                # rather than continue, should rather just run the calculation!!
                continue
            # check that the pt range for this mass window is correct
            pt_rng = f[f.find('pt')+3:-len('masswindow.out')-1]
            # the pt range is always split by an underscore
            spl = pt_rng.split('_')
            pt_l = float(spl[0])
            pt_h = float(spl[1])
            # make sure we have the correct pt range mass window file
            if pt_l*1000 == float(ptrange[0]) and pt_h*1000 == float(ptrange[1]):
                print 'mass window file: ' +f 
                massWinFile = InputDir+'/'+f

    return signalFile, backgroundFile, eventsFileSig, eventsFileBkg, massWinFile


def writeCSV(signalFile, backgroundFile, branches, cutstring, treename, Algorithm, fileid, ptweights, plotranges):
    '''
    Create csv files of the signal and background ntuples.
    Keyword args:
    signal/backgroundFile -- The filenames for the signal and bkg.
    branches -- the variables to be kept in the csv.
    cutstring -- basic selection cuts to be applied.
    treename -- name of tree in ntuples
    Algorithm -- algorithm being run
    fileid -- File identifier that gets used in the output file names
    ptweights -- pt weight file (bkg/signal pt)
    plotranges --- Dictionary of the ranges for different variables. This is the stub variable from Tagger, so it has a leading _
    '''

    import copy    
    from array import array

    branches_pruned = copy.deepcopy(branches)

    # add entries for weights
    to_append = ['mc_event_weight', 'evt_xsec', 'evt_filtereff', 'evt_nEvts', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta', 'jet_CamKt12Truth_phi', 'jet_CamKt12Truth_m', 'jet_CamKt12LCTopo_pt', 'jet_CamKt12LCTopo_eta', 'jet_CamKt12LCTopo_phi', 'jet_CamKt12LCTopo_m']
    for a in to_append:
        if a not in branches_pruned:
            branches_pruned.append(a)

    # list any variables that you do not want to cut on/ add to the cutstring
    donotcut = ['averageIntPerXing']
    plotranges = {key: value for key, value in plotranges.items() if key not in donotcut}

    # update cutstring to keep all variables within range specified in plotconfig
    for p in plotranges.keys():
        prefix = "jet_"+Algorithm
        if not p.startswith("_"):
            prefix += "_"
        cutstring += "*(" +prefix+p+">="+str(plotranges[p][0])+ ")*(" +prefix+p+"<="+str(plotranges[p][1]) + ")"

    # read the signal and background files
    for typename in ['sig','bkg']:
        if typename == 'sig':
            filename = signalFile
        else:
            filename = backgroundFile
        # open the files
        file_in = TFile(filename)
        # read in the trees
        tree = file_in.Get(treename)
    
        # write the tree out in csv format to use again later
        # need to add some entries to branches to store the event weights
        numpydata = root_numpy.root2array(filename,treename,branches_pruned,cutstring)

        # need to add single entry per event with full weight -> mc*pt*rest
        # see https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas
        numpydata = pd.DataFrame(numpydata)
        numpydata.rename(columns=lambda x: x.replace('jet_' + Algorithm+'_',''), inplace=True)

        if typename == 'bkg':
            numpydata['weight'] = [numpydata['evt_filtereff'][i]*numpydata['evt_xsec'][i]*numpydata['mc_event_weight'][i]*(1./numpydata['evt_nEvts'][i]) for i in xrange(0,len(numpydata['evt_xsec']))]
        else:
            numpydata['weight'] = [ptweights.GetBinContent(ptweights.GetXaxis().FindBin(numpydata['jet_CamKt12Truth_pt'][i]/1000.)) for i in xrange(0,len(numpydata['evt_xsec']))]
        #print list(numpydata)
        

        if typename == 'sig': 
            numpydata['label']=1 
            numpydata.to_csv('csv/' + Algorithm + fileid + '_' + typename + '.csv')
            
        else: 
            numpydata['label']=0 
            numpydata.to_csv('csv/' + Algorithm + fileid + '_' + typename + '.csv')#,mode='a',header=False)

        file_in.Close()
            


def RocCurve_SingleSided_WithUncer(sig, bkg, sigeff, bkgeff, cutside=''):
    '''
    Method taken from Sam's code:
    svn+ssh://svn.cern.ch/reps/atlasperf/CombPerf/JetETMiss/JetSubstructure2012/BoostedBosonTagging/code/meehan/PostAnalysis
    1-sided ROC Curve:
    PostAnalysis/MyPackages/MyLocalFunctions.py

    Different method of creating the ROC curve with a right/ left cut instead of 
    just going from left.


    Key args:
    sig -- Signal histogram
    bkg -- Bkg histogram
    sigeff -- Signal efficiency
    bkgeff -- Bkg efficiency
    curve -- The roc curve we are filling
    bkgRejPower -- The background rejection power curve
    cutside -- L or R

    Returns:
    gr -- the TGraph with the ROC curve
    hsigreg50 -- 50% signal
    hcutval50 -- 50% signal cut
    hsigreg25 -- 25% signal
    hcutval25 -- 25% signal cut
    '''
    print "\n\nMake ROC curve using right/left cut",cutside

    n = bkg.GetNbinsX()
    #print "NBins",n

    # normalise hists
    if sig.Integral()!=0:
        sig.Scale(1.0/sig.Integral());
    if(bkg.Integral()!=0):
        bkg.Scale(1.0/bkg.Integral());

    totalBerr=Double()
    totalSerr=Double()
    totalB = bkg.IntegralAndError(0,n,totalBerr)
    totalS = sig.IntegralAndError(0,n,totalSerr)
    
    siglow  = sig.GetXaxis().GetXmin()
    sighigh = sig.GetXaxis().GetXmax()
    hsigreg50 = TH1F("hsigreg50","hsigreg50",n,siglow,sighigh)
    hsigreg50.SetDirectory(0)
    hcutval50 = TH1F("hcutval50","hcutval50",5,0,5)
    hcutval50.SetDirectory(0)
    hcutval50.GetXaxis().SetBinLabel(1,"Left(0) , Right(1)")
    hcutval50.GetXaxis().SetBinLabel(2,"LowerCut")
    hcutval50.GetXaxis().SetBinLabel(3,"UpperCut")
    hsigreg25 = TH1F("hsigreg25","hsigreg25",n,siglow,sighigh)
    hsigreg25.SetDirectory(0)
    hcutval25 = TH1F("hcutval25","hcutval25",5,0,5)
    hcutval25.SetDirectory(0)
    hcutval25.GetXaxis().SetBinLabel(1,"Left(0) , Right(1)")
    hcutval25.GetXaxis().SetBinLabel(2,"LowerCut")
    hcutval25.GetXaxis().SetBinLabel(3,"UpperCut")
    if cutside=="R":
        hcutval50.SetBinContent(1,1)
        hcutval50.SetBinContent(3,sig.GetXaxis().GetBinLowEdge(n)+sig.GetXaxis().GetBinWidth(n))
        extrema50 = 100000
        hcutval25.SetBinContent(1,1)
        hcutval25.SetBinContent(3,sig.GetXaxis().GetBinLowEdge(n)+sig.GetXaxis().GetBinWidth(n))
        extrema25 = 100000
    elif cutside=="L":
        hcutval50.SetBinContent(1,0)
        hcutval50.SetBinContent(2,sig.GetXaxis().GetBinLowEdge(1))
        extrema50 = -100000
        hcutval25.SetBinContent(1,0)
        hcutval25.SetBinContent(2,sig.GetXaxis().GetBinLowEdge(1))
        extrema25 = -100000

    gr = TGraphErrors(n)
    for i in range(1,n+1):
        myS = 0.
        myB = 0.

        if cutside=="R":
            #loop from i to end
            myBerr=Double()
            mySerr=Double()
            myB = bkg.IntegralAndError(i,n,myBerr)
            myS = sig.IntegralAndError(i,n,mySerr)
            #print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS*sigeff, (1-myB*bkgeff))
            gr.SetPointError(i, mySerr*sigeff, myBerr*bkgeff)
            if myS<=0.73:
                hsigreg50.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)
                #print tempex,extrema50
                if tempex<extrema50:
                    extrema50 = tempex
                    #print "found extrema R: ",extrema50
            if myS<=0.36:
                hsigreg25.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)
                #print tempex,extrema25
                if tempex<extrema25:
                    extrema25 = tempex
                    #print "found extrema R: ",extrema50
        elif cutside=="L":
            #loop from 0 to i
            myBerr=Double()
            mySerr=Double()
            myB = bkg.IntegralAndError(1,i,myBerr)
            myS = sig.IntegralAndError(1,i,mySerr)
            #print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS*sigeff, (1-myB*bkgeff))
            gr.SetPointError(i, mySerr*sigeff, myBerr*bkgeff)
            if myS<=0.73:
                hsigreg50.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)+sig.GetXaxis().GetBinWidth(i)
                #print tempex,extrema50
                if tempex>extrema50:
                    extrema50 = tempex
                    #print "found extrema L: ",extrema50
            if myS<=0.36:
                hsigreg25.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)+sig.GetXaxis().GetBinWidth(i)
                #print tempex,extrema25
                if tempex>extrema25:
                    extrema25 = tempex
                    #print "found extrema L: ",extrema50
            
            
        else:
            print "You did not choose a left or right handed cut - EXITING ..."
            sys.exit()
            
    #artificially set the first point to (1,1) to avoid overflow issues
    gr.SetPoint(0, 0.0, 1.0)
    gr.SetPointError(0, 0.0, 0.0)
            
    ctest = TCanvas("ctest","ctest",400,400)
    gr.SetMinimum(0.0)
    gr.SetMaximum(1.0)
    gr.GetXaxis().SetRangeUser(0.0,1.0)
    gr.Draw("AE3")
    
    if cutside=="R":
        hcutval50.SetBinContent(2,extrema50)
        hcutval25.SetBinContent(2,extrema25)
    elif cutside=="L":
        hcutval50.SetBinContent(3,extrema50)
        hcutval25.SetBinContent(3,extrema25)

    curve = gr
    bkgRejPower = gr
    print "RETURNING from Single sided ROC calculation"
    return gr,hsigreg50,hcutval50,hsigreg25,hcutval25


def RocCurve_SingleSided(sig, bkg, sig_eff, bkg_eff, cutside='L'):

    
    n = bkg.GetNbinsX()
    #print "NBins",n

    # normalise hists
    #if sig.Integral()!=0:
    #    sig.Scale(1.0/sig.Integral());
    #if(bkg.Integral()!=0):
    #    bkg.Scale(1.0/bkg.Integral());

    totalBerr=Double()
    totalSerr=Double()
    totalB = bkg.Integral(0,n)
    totalS = sig.Integral(0,n)


    gr = TGraph(n)
    for i in range(1,n+1):
        myS = 0.
        myB = 0.

        myBerr=Double()
        mySerr=Double()
        #myB = bkg.Integral(1,i)
        #myS = sig.Integral(1,i)
        #print 'myS: ' + str(myS)
        #print 'myB: ' + str(myB)
        #gr.SetPoint(i, myS*sig_eff, (1-myB*bkg_eff))
        #gr.SetPointError(i, mySerr*sig_eff, myBerr*bkg_eff)
        if cutside=="R":
            #loop from i to end
            myB = bkg.Integral(i,n)
            myS = sig.Integral(i,n)
            #print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS*sig_eff, (1-myB*bkg_eff))
            #gr.SetPointError(i, mySerr*sigeff, myBerr*bkgeff)
        elif cutside=="L":
            #loop from 0 to i
            myB = bkg.Integral(1,i)
            myS = sig.Integral(1,i)
            #print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS*sig_eff, (1-myB*bkg_eff))
            #gr.SetPointError(i, mySerr*sigeff, myBerr*bkgeff)
            
            
    #artificially set the first point to (1,1) to avoid overflow issues
    #gr.SetPoint(0, 1.0, 1.0)
    
            
    ctest = TCanvas("ctest","ctest",400,400)
    gr.SetMinimum(0.0)
    gr.SetMaximum(1.0)
    gr.GetXaxis().SetRangeUser(0.0,1.0)
    gr.Draw("AE3")
    
    curve = gr
    bkgRejPower = gr
    print "RETURNING from Single sided ROC calculation"
    return gr


def median(hist):
    '''
    Calculate the median of a histogram.  Use GetQuantiles() with a probability of 0.5.
    '''
    
    import array as arr
    # The GetQuantiles method takes a Double_t * as the 2nd and 3rd args
    median = arr.array('d',[-1])
    prob = arr.array('d',[0.5])
    # median gets filled with the actual median
    hist.GetQuantiles(1, median, prob)
    
    return median[0]

def getEfficiencies(roc):
    '''
    Method to return a numpy array of the signal and background efficiency from a ROC curve.
    
    Key args:
    roc -- The input ROC curve in the form of a TGraph

    Returns:
    sig_eff --- the signal efficiency
    bkg_rej --- the background rejection (1 - eff)
    '''
    import ROOT as rt
    import numpy as np

    # number of points in the TGraph
    n_points = roc.GetN()
    sig_eff = np.zeros(n_points)
    bkg_rej = np.zeros(n_points)
    
    for i in range(n_points):
        sig_i = rt.Double(0)
        bkg_i = rt.Double(0)
        # the signal and background values are set by ref.
        roc.GetPoint(i, sig_i, bkg_i)
        sig_eff[i] = float(sig_i)
        bkg_rej[i] = float(bkg_i)

    return sig_eff, bkg_rej
