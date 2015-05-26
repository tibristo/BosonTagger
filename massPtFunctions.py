from ROOT import *
import sys
import os
import copy
import numpy as n
import math
from array import array

ptweightBins = [200,250,300,350,400,450,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1800,2000,2200,2400,2600,2800,3000]
ptweights = TH1F()

nevents = {}
success_pt = False
filename_pt = ''
success_m = False
filename_m = ''
pt_high = 3000
pt_low = 200
weightedxAOD = False

def setPtWeightFile(sigtmp, bkgtmp):
    '''
    Set up the histogram used for pt rw.  The signal and background pt TH1F histograms are 
    used to create this.
    sigtmp --- signal pT TH1F 
    bkgtmp --- background pT TH1F
    '''
    global ptweights

    sig = sigtmp.Clone()
    bkg = bkgtmp.Clone()

    if sig.Integral() != 0:
        sig.Scale(1./sig.Integral())
    if bkg.Integral() != 0:
        bkg.Scale(1./bkg.Integral())

    ptweights = copy.deepcopy(bkg)
    ptweights.SetDirectory(0)
    ptweights.Divide(sig)

    t = TCanvas("ptrw")
    gPad.SetLogy()
    print 'draw'
    ptweights.Draw()
    print 'drawn'
    t.SaveAs("ptrw.png")

    f = open('ptrw_weights','w')
    for x in range(1, int(ptweights.GetNbinsX())+1):
        f.write(str(ptweights.GetXaxis().GetBinLowEdge(x))+': ' + str(ptweights.GetBinContent(x)) + '\n')
    f.close()


def getPtWeightsFile():
    '''
    Return a copy of the ptweights histogram.
    '''

    return ptweights


def dR(eta1, phi1, eta2, phi2):
    '''
    Calculate the deltaR between two particles given their eta and phi values.
    '''
    dphi = abs(phi1-phi2)
    deta = eta1-eta2
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    return math.sqrt(dphi*dphi + deta*deta)


def NEvents(runNumber):
    '''
    Method for getting the number of events given a specific RunNumber.
    args 
    runNumber --- string or long of run number

    returns number of events, or 1 if not found
    '''
    # load global dictionary which is set in loadEvents()
    global nevents
    if long(runNumber) in nevents.keys():
        return nevents[runNumber];
    else:
        return 1;

def loadEvents(filename):
    '''
    Method for reading in a text file that contains the number of events per Run Number, comma separated (runnumber,events).  It stores these
    in a global dictionary called nevents.
    filename --- input text file.
    '''
    global nevents
    f = open(filename);
    for line in f:
        # comma separated file
        spl = line.strip().split(',')
        run = spl[0]
        ev = spl[1]
        nevents[long(run)] = float(ev)

    f.close()


def drawPtWeight():
    #global ptweights
    t = TCanvas("blah")
    ptweights.Draw()
    t.SaveAs("blahblahblah.png")

def ptWeight(pt):
    '''
    Method to return the weight for a given pT.
    pt --- pT in MeV.
    
    returns scale factor/ weight.
    '''
    ptw = ptweights.GetBinContent(ptweights.GetXaxis().FindBin(pt/1000));
    return ptw


def setupHistogram(fname, algorithm, treename, ptweightsHist, signal=False):
    '''
    Read in the jet mass from the input file and fill it into a histogram.  The histogram gets used to calculate the mass window.
    Keyword args:
    fname --- input file name.
    algorithm --- Algorithm being used.
    treename --- treename for input root file
    '''

    global pt_high
    global pt_low
    global weightedxAOD

    # load the NEvents file to weight by number of events
    InputDir = fname[:fname.rfind('/')]
    fileslist = os.listdir(InputDir)
    for f in fileslist:
        if f.find('nevents')!=-1:
            eventsfile = InputDir+'/'+f
    loadEvents(eventsfile)

    sampleString = 'SIG'
    if signal:
        sampleString = 'BKG'

    # set up a struct so that the SetBranchAddress method can be used.
    # normally one would use tree.variable to access the variable, but since
    # this variable name changes for each algorithm this is a better way to do it.
    gROOT.ProcessLine("struct jet_t { Float_t mass; Float_t pt; Float_t eta; Float_t phi; Float_t eta_topo; Float_t phi_topo;} ")
    # create an object of this struct
    jet = jet_t()
    
    # open file and ttree
    f = TFile.Open(fname,"READ")
    tree = f.Get(treename)
    
    # set branch address for the groomed jet mass
    tree.SetBranchAddress("jet_"+algorithm+"_m", AddressOf(jet,'mass'))
    tree.SetBranchAddress("jet_CamKt12Truth_pt", AddressOf(jet,'pt'))
    tree.SetBranchAddress("jet_CamKt12Truth_eta", AddressOf(jet,'eta'))
    tree.SetBranchAddress("jet_CamKt12Truth_phi", AddressOf(jet,'phi'))
    tree.SetBranchAddress("jet_CamKt12LCTopo_eta", AddressOf(jet,'eta_topo'))
    tree.SetBranchAddress("jet_CamKt12LCTopo_phi", AddressOf(jet,'phi_topo'))

    # histogram with 300 bins and 0 - 3 TeV range
    # histogram storing pt without being reweighted
    hist_pt = TH1F("pt"+sampleString+algorithm,"pt",100,200*1000,3000*1000)    
    hist_pt.SetDirectory(0)
    # jet mass
    hist_m = TH1F("mass"+sampleString+algorithm,"mass",600,0,1200*1000)    
    hist_m.SetDirectory(0)
    # pt 
    hist_rw = TH1F("ptrw"+sampleString+algorithm,"ptrw",100,200*1000,3000*1000)    
    hist_rw.SetDirectory(0)

    # maximum number of entries in the tree
    entries = tree.GetEntries()
    
    # loop through
    for e in xrange(entries):
        tree.GetEntry(e)
        if e%1000 == 0:
            print 'Progress: ' + str(e) + '/' + str(entries)
        # According to the email from Chris (25/04/2015) we should not be applying any weights on the signal except for ptrw. 
        # in addition we should be matching the ca12truth jet to the ca12lctopo jet as well. We are also not doing k_factor rw.
        # weight is used for the mass window calculation
        # weight2 is used for the pt reweighting
        if not weightedxAOD and not signal:
            weight = 1*tree.mc_event_weight*tree.filter_eff*(1/NEvents(tree.mc_channel_number))#*tree.k_factor
            weight2 = 1*tree.mc_event_weight*tree.filter_eff*(1/NEvents(tree.mc_channel_number))#*tree.k_factor
        elif not signal:
            weight = 1*tree.mc_event_weight*tree.evt_filtereff*(1/tree.evt_nEvts)#*tree.k_factor
            weight2 = 1*tree.mc_event_weight*tree.evt_filtereff*(1/tree.evt_nEvts)#*tree.k_factor
        else: # if it is signal
            weight = 1.0
            weight2 = 1.0

        # apply basic selection criteria - slightly different for pt and mass window
        #if createptfile:
        #    if (jet.pt < 200*1000 or abs(jet.eta) >= 1.2) or abs(dR(jet.eta,jet.phi,jet.eta_topo,jet.phi_topo)) >= 0.75*1.2: # need to match the lctopo and truth ca12 jets. 0.75*1.2 = 
        #        continue
        #else:
        if abs(jet.eta) > 1.2 or jet.pt > pt_high*1000 or jet.pt <= pt_low*1000 or jet.mass <= 0: #or jet.mass >= 300*1000 
            continue

        if signal:
            #ptw = ptweightsHist.GetBinContent(ptweightsHist.GetXaxis().FindBin(jet.pt/1000));
            weight*=ptWeight(jet.pt)
        elif not signal:
            weight*=tree.evt_xsec
            weight2*=tree.evt_xsec
        #print weight
        hist_m.Fill(jet.mass,weight)
        hist_pt.Fill(jet.pt,weight2)

        hist_rw.Fill(jet.pt,weight)

    return hist_m, hist_pt, hist_rw

def Qw(histo, frac=0.68):
    '''
    Method for calculating the mass window from a histogram of masses.
    Keyword args:
    histo --- input TH1F histogram of masses
    frac --- the mass window fraction. Normally use 68%
    '''
    # set up the variables that store the best found window
    minWidth = 1000000.0;
    topEdge = 0.0;
    botEdge = 0.0
    maxidx = 99
    minidx = 0
    maxfrac = 0

    # info on histogram - number of bins and the integral
    Nbins = histo.GetNbinsX()
    integral = histo.Integral();
    if integral == 0:# no div/0
        integral = 1
    closestWindow = 100
    closestFrac = 0.0
    closestTop = 0.0
    closestBottom = 0.0
    closestMinIdx = 0.0
    closestMaxIdx = 99.0
    numberWindows = 0

    # loop through each bin of the histogram
    for i in xrange(0,Nbins):

        tempFrac = 0.0
        # want to make sure we don't change i when changing imax
        imax = copy.deepcopy(i)

        # loop through until the tempFrac is above the frac (68%) criteria,
        # but making sure not to go out of range.                                                                                                                                             
        prevFrac = 0.0
        while(tempFrac<frac and imax != Nbins):
            tempFrac+=histo.GetBinContent(imax)/integral;
            prevFrac = histo.GetBinContent(imax)/integral
            imax+=1;

        width = histo.GetBinCenter(imax) - histo.GetBinCenter(i);
        # by applying this we say that the window we have just calculate MUST have at least 68%.
        #if tempFrac >= frac and width<minWidth:
        if tempFrac >= frac and imax != Nbins and width<minWidth:
            # set up the best found mass window variables
            minWidth = width;
            topEdge = histo.GetBinCenter(imax);
            botEdge = histo.GetBinCenter(i)
            minidx = copy.deepcopy(i)
            maxidx = copy.deepcopy(imax)
            maxfrac = copy.deepcopy(tempFrac)
            
        if tempFrac > frac:
            numberWindows += 1

        # find the closest window. It may be slightly lower than the 68%.
        if abs(tempFrac-prevFrac-frac) < closestWindow and imax!=Nbins and width < minWidth:
            closestFrac = tempFrac-prevFrac
            closestTop = histo.GetBinCenter(imax-1)
            closestBottom = histo.GetBinCenter(i)
            closestMinIdx = copy.deepcopy(i)
            closestMaxIdx = copy.deepcopy(imax-1)
        if abs(tempFrac-frac) < closestWindow and width < minWidth and imax!=Nbins:
            closestFrac = tempFrac
            closestTop = histo.GetBinCenter(imax)
            closestBottom = histo.GetBinCenter(i)
            closestMinIdx = copy.deepcopy(i)
            closestMaxIdx = copy.deepcopy(imax)
                

    return minWidth, topEdge, botEdge, minidx, maxidx, closestFrac, closestTop, closestBottom, closestMinIdx, closestMaxIdx, numberWindows


def run(fname, algorithm, treename, ptweightsHist, ptlow=200, pthigh=3000):
    '''
    Method for running over a single algorithm and calculating the mass window.
    Keyword args:
    fname --- the input file name
    algorithm --- the name of the algorithm
    treename --- Name of tree in input root files
    ptlow/high --- pt range. default values are those used for pt reweighting file creation
    '''
    global filename_m
    filename_m = ''
    global success_m
    success_m = False
    global pt_high
    pt_high = pthigh
    global pt_low
    pt_low = ptlow
    # setup the histogram - read in the mass entries from the input file and put them into a histogram

    hist_sig_m, hist_sig_pt,hist_rw = setupHistogram(fname, algorithm, treename, ptweightsHist, True)
    
    # folder where input file is
    folder = fname[:fname.rfind('/')+1]

    # calculate the width, top and bottom edges and the indices for the 68% mass window
    wid, topedge, botedge, minidx, maxidx,closestFrac, closestTop, closestBottom, closestMinIdx, closestMaxIdx, numWin = Qw(hist_sig_m, 0.68)
    # folder where input file is
    folder = fname[:fname.rfind('/')+1]
    # write this information out to a text file
    filename_m = folder+algorithm+"_pt_"+str(ptlow)+"_"+str(pthigh)+"_masswindow.out"
    fout = open(filename_m,'w')
    fout.write("width: " + str(wid) + '\n')
    fout.write("top edge: " + str(topedge) + '\n')
    fout.write("bottom edge: " + str(botedge) + '\n')
    fout.write("minidx: " + str(minidx) + '\n')
    fout.write("maxidx: " + str(maxidx) + '\n')
    fout.write('num entries' + str(hist_sig_m.GetEntries())+'\n')
    fout.write('closest frac: ' + str(closestFrac) + '\n')
    fout.write('closest top: ' + str(closestTop) + '\n')
    fout.write('closest bottom: ' + str(closestBottom) + '\n')
    fout.write('closest minidx: ' + str(closestMinIdx) + '\n')
    fout.write('closest maxidx: ' + str(closestMaxIdx) + '\n')
    fout.write('number windows found: ' + str(numWin) + '\n')
    fout.close()
    success_m = True


def runAll(input_file, file_id, treename):
    '''
    Method for running over a collection of algorithms and checking truth pt
    The algorithms are stored in a text file which is read in.  Each line gives the input folder, with the folder of the form ALGORITHM_fileID/.
    input_file --- the text file with the folders
    file_id --- the folder suffix
    treename --- Name of tree in input root files
    '''
    # open input file
    f = open(input_file,"read")
    # list that will store all of the filenames and corresponding algorithms
    fnames = []
    # read in file
    for l in f:
        # remove whitespace
        l = l.strip()+'/'
        # The input folder is given on each line
        folder = l
        # now get the files in this folder and iterate through, looking for the signal file
        fileslist = os.listdir(folder)
        sigfile = ""
        for fil in fileslist:
            # found the signal file!
            if fil.endswith("sig.root"):#"sig2.root"):
                sigfile = folder+fil

        # the algorithm can be obtained from the folder by removing the leading
        # directory information and removing the file identifier
        alg = l.split('/')[-2][:-len(file_id)]
        # add these to the list to run over
        fnames.append([sigfile,alg])

    # loop through all of the algorithms and find the mass windows
    for a in fnames:
        print "running " + a[1]
        run(a[0],a[1],treename)
