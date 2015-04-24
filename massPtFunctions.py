from ROOT import *
import sys
import os
import copy
import numpy as n
from array import array

ptweightBins = [200,250,300,350,400,450,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1800,2000,2200,2400,2600,2800,3000]
#ptweights = TH1F("ptreweight","ptreweight",100,0,3000);
ptweights = TH1F("ptreweight","ptreweight",len(ptweightBins)-1,array('d',ptweightBins));

nevents = {}
success_pt = False
filename_pt = ''
success_m = False
filename_m = ''
pt_high = 3000
pt_low = 200
weightedxAOD = False

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


def ptWeight(pt):
    '''
    Method to return the weight for a given pT.
    pt --- pT in MeV.
    
    returns scale factor/ weight.
    '''
    # load global pt weights histogram
    global ptweights
    return ptweights.GetBinContent(ptweights.GetXaxis().FindBin(pt/1000));


def setupPtWeight(filename):
    '''
    Method to read in a text file that contains the pt weighting information.  It is a csv that has: lower edge, bkg events, signal events.
    The number of events per bin entry for background and signal are taken as a ratio for weighting signal events.
    filename --- Input csv.
    '''
    # load the global pt weights histogram
    global ptweights
    print 'ptfile: '+filename
    # open file and set up parameters for running through the file.
    f = open(filename);
    # step size of 30
    numbins = 100
    step_size = 3000/numbins;
    counter = 1;
    running_bkg = 0
    running_sig = 0
    current_edge = 0
    next_edge = 0;
    next_edge = step_size;
    # start at 200, don't care about anything below this.
    current_edge = 200;
    # the end of the bin, so now have the initial bin up and low edges
    next_edge = ptweights.GetXaxis().GetBinUpEdge(1)
    #next_edge = current_edge+step_size;

    # loop through the input file
    for line in f:
        spl = line.strip().split(',')
        print line
        edge = spl[0]
        bkg = spl[1]
        sig = spl[2]
        # don't care about anything less than 200
        if float(current_edge) < 200:
            continue

      
        # if we are now into the next bin.
        if (float(edge) > next_edge):
            # we now have the low edge as the old up edge
            current_edge = copy.deepcopy(next_edge)
            # get the new up edge
            next_edge = ptweights.GetXaxis().GetBinUpEdge(counter+1);
            # fill in the pt weights
            if (running_sig == 0):
                ptweights.SetBinContent( counter, 0 );
            else:
                ptweights.SetBinContent( counter, running_bkg/running_sig );
            # reset the running totals and increment the bin counter
            running_bkg = 0;
            running_sig = 0;
            counter+=1;
            
        # keep track of the total bkg and signal in this bin.
        running_bkg += float(bkg);
        running_sig += float(sig);

    tc = TCanvas()
    tc.cd()
    ptweights.Draw()
    tc.SaveAs('reweight.png')
    f.close();

def setupHistogram(fname, algorithm, treename, signal=False, ptfile = '',createptfile = False):
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
    gROOT.ProcessLine("struct jet_t { Float_t mass; Float_t pt; Float_t eta;} ")
    # create an object of this struct
    jet = jet_t()
    if signal and not createptfile:
        setupPtWeight(ptfile)
    
    # open file and ttree
    f = TFile.Open(fname,"READ")
    tree = f.Get(treename)
    
    # set branch address for the groomed jet mass
    tree.SetBranchAddress("jet_"+algorithm+"_m", AddressOf(jet,'mass'))
    tree.SetBranchAddress("jet_CamKt12Truth_pt", AddressOf(jet,'pt'))
    #tree.SetBranchAddress("jet_"+algorithm.replace("LCTopo","Truth")+"_pt", AddressOf(jet,'pt'))
    tree.SetBranchAddress("jet_CamKt12Truth_eta", AddressOf(jet,'eta'))
    #tree.SetBranchAddress("jet_"+algorithm.replace("LCTopo","Truth")+"_eta", AddressOf(jet,'eta'))
    # histogram with 300 bins and 0 - 3 TeV range
    # histogram storing pt without being reweighted
    hist_pt = TH1F("pt"+sampleString+algorithm,"pt",100,200*1000,3000*1000)    
    #hist_pt = TH1F("jet_CamKt12Truth_pt","CA12Truth_pt",100,200*1000,3000*1000)    
    hist_pt.SetDirectory(0)
    # jet mass
    hist_m = TH1F("mass"+sampleString+algorithm,"mass",600,0,1200*1000)    
    hist_m.SetDirectory(0)
    # pt 
    hist_rw = TH1F("ptrw"+sampleString+algorithm,"ptrw",100,200*1000,3000*1000)    
    hist_rw.SetDirectory(0)
    # pt only if creating pt rw file
    pthist = TH1F("ptreweight"+sampleString+algorithm,"ptreweight",56,200,3000); # gives 50 gev per bin
    pthist.SetDirectory(0)
    # maximum number of entries in the tree
    entries = tree.GetEntries()
    
    # loop through
    for e in xrange(entries):
        tree.GetEntry(e)
        if e%1000 == 0:
            print 'Progress: ' + str(e) + '/' + str(entries)
        # weight is used for the mass window calculation
        # weight2 is used for the pt reweighting
        if not weightedxAOD:
            weight = 1*tree.mc_event_weight*tree.k_factor*tree.filter_eff*(1/NEvents(tree.mc_channel_number))
            weight2 = 1*tree.mc_event_weight*tree.k_factor*tree.filter_eff*(1/NEvents(tree.mc_channel_number))
        else:
            weight = 1*tree.mc_event_weight*tree.evt_kfactor*tree.evt_filtereff*(1/tree.evt_nEvts)
            weight2 = 1*tree.mc_event_weight*tree.evt_kfactor*tree.evt_filtereff*(1/tree.evt_nEvts)

        # apply basic selection criteria - slightly different for pt and mass window
        if createptfile:
            if (jet.pt < 200*1000 or abs(jet.eta) >= 1.2):# or jet.mass > 300*1000.0):# and not createptfile:
                continue
        else:
            if abs(jet.eta) > 1.2 or jet.pt >= pt_high*1000 or jet.pt < pt_low*1000 or jet.mass <= 0: #or jet.mass >= 300*1000 
                continue

        if signal and not createptfile:
            weight*=ptWeight(jet.pt)
        elif not signal:
            weight*=tree.xs
            weight2*=tree.xs
        
        hist_m.Fill(jet.mass,weight)
        hist_pt.Fill(jet.pt,weight2)

        if createptfile:
            pthist.Fill(jet.pt/1000,weight)

        if weight <= 0 and False:
            print 'weight is 0!'
            print 'mc_event_weight: ' + str(tree.mc_event_weight)
            if not weightedxAOD:
                print 'k_factor: ' + str(tree.k_factor)
                print 'filter_eff: ' + str(tree.filter_eff)
                print 'xs: ' + str(tree.xs)
                print 'nevents: ' + str(NEvents(tree.mc_channel_number))
            else:
                print 'k_factor: ' + str(tree.evt_kfactor)
                print 'filter_eff: ' + str(tree.evt_filtereff)
                print 'xs: ' + str(tree.xs)
                print 'nevents: ' + str(tree.evt_nEvts)
            print 'ptweight: ' + str(ptWeight(jet.pt))
        hist_rw.Fill(jet.pt,weight)

    return hist_m, hist_pt, hist_rw, pthist

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


def run(fname, algorithm, treename, ptfile, ptlow=200, pthigh=3000, version='v6'):
    '''
    Method for running over a single algorithm and calculating the mass window.
    Keyword args:
    fname --- the input file name
    algorithm --- the name of the algorithm
    treename --- Name of tree in input root files
    ptfile --- pt reweighting file
    ptlow/high --- pt range. default values are those used for pt reweighting file creation
    version --- version of pt file
    '''
    global filename_m
    filename_m = ''
    global success_m
    success_m = False
    global filename_pt
    filename_pt = ''
    global success_pt
    success_pt = False
    global pt_high
    pt_high = pthigh
    global pt_low
    pt_low = ptlow
    # setup the histogram - read in the mass entries from the input file and put them into a histogram
    createptfile = False
    if ptfile == '':
        # first check if a correct version file exists
        #success_pt = True
        # if it doesn't, then we create it
        createptfile = True

    hist_sig_m, hist_sig_pt,hist_rw,pthist_sig = setupHistogram(fname, algorithm, treename, True, ptfile, createptfile)
    bkg_fname = fname.replace('sig','bkg')
    thing =  pthist_sig.GetXaxis()
    hist_bkg_m,hist_bkg_pt,tmp,pthist_bkg = setupHistogram(bkg_fname, algorithm, treename, False, '', createptfile)

    # folder where input file is
    folder = fname[:fname.rfind('/')+1]

    print 'createptfile: '
    print createptfile
    # if we are creating a new pt rw file, save all of the info
    if createptfile:
        filename_pt = folder+algorithm+'.ptweights'+version
        ptfile_out = open(filename_pt,'w')
        #ptfile_cutflow = open(filename_pt.replace('.ptweights'+version,'.cutflow'),'w')
        print 'writing to file' 
        for i in range(1,pthist_sig.GetXaxis().GetNbins()+1):
            print ' first entry'
            sig = pthist_sig.GetBinContent(i)
            bkg = pthist_bkg.GetBinContent(i)
            edge = pthist_sig.GetXaxis().GetBinUpEdge(i)
            ptfile_out.write(str(edge)+','+str(bkg)+','+str(sig)+'\n')
            
        ptfile_out.close()
        #ptfile_cutflow.close()
        success_pt = True

        # write this information out as a plot
        # this is used to compare the pt before and after rw
        canv = TCanvas()
        canv.cd()
        canv.SetLogy()
        if hist_sig_pt.Integral() != 0:
            hist_sig_pt.Scale(1/hist_sig_pt.Integral())
        if hist_bkg_pt.Integral() != 0:
            hist_bkg_pt.Scale(1/hist_bkg_pt.Integral())
        if hist_rw.Integral() != 0:
            hist_rw.Scale(1/hist_rw.Integral())
        hist_sig_pt.Draw()
        hist_bkg_pt.SetLineColor(ROOT.kRed)
        hist_bkg_pt.Draw("same")
        canv.SaveAs(folder+"pt_truth.png")
        hist_rw.Draw()
        hist_bkg_pt.Draw("same")
        canv.SaveAs(folder+"pt_rw.png")
    else: # if we are just doing the mass windows
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


def runAll(input_file, file_id, treename, version='v6'):
    '''
    Method for running over a collection of algorithms and checking truth pt
    The algorithms are stored in a text file which is read in.  Each line gives the input folder, with the folder of the form ALGORITHM_fileID/.
    input_file --- the text file with the folders
    file_id --- the folder suffix
    treename --- Name of tree in input root files
    version --- version of pt reweighting file
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
        ptfile = ""
        for fil in fileslist:
            # found the signal file!
            if fil.endswith("sig.root"):#"sig2.root"):
                sigfile = folder+fil
            if fil.endswith("ptweights"+version):
                ptfile = folder+fil
        # the algorithm can be obtained from the folder by removing the leading
        # directory information and removing the file identifier
        alg = l.split('/')[-2][:-len(file_id)]
        # add these to the list to run over
        fnames.append([sigfile,alg])

    # loop through all of the algorithms and find the mass windows
    for a in fnames:
        print "running " + a[1]
        run(a[0],a[1],treename, ptfile)

if __name__=="__main__":
    print len(sys.argv)
    if len(sys.argv) < 3:
        print "usage: python ptreweighting.py text_file_with_folder_names folder_suffix(including leading underscore) tree-name [version]"
        sys.exit()
    if len(sys.argv) == 4:
        runAll(sys.argv[1],sys.argv[2],sys.argv[3])
    elif len(sys.argv) == 5:
        runAll(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
