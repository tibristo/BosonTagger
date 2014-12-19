from ROOT import *
import sys
import os
import copy
import numpy as n


ptweights = TH1F("ptreweight","ptreweight",50,0,3000);

nevents = {}

def NEvents(runNumber):
    global nevents
    if long(runNumber) in nevents.keys():
        return nevents[runNumber];
    else:
        return 1;

def loadEvents(filename):
    global nevents
    f = open(filename);
    for line in f:
        spl = line.strip().split(',')
        run = spl[0]
        ev = spl[1]
        nevents[long(run)] = float(ev)

    f.close()


def ptWeight(pt):
    global ptweights
    return ptweights.GetBinContent(ptweights.GetXaxis().FindBin(pt/1000));


def setupPtWeight(filename):
    global ptweights
    print 'ptfile: '+filename
    f = open(filename);
    numbins = 50
    step_size = 3000/numbins;
    counter = 1;
    running_bkg = 0
    running_sig = 0
    current_edge = 0
    next_edge = 0;
    next_edge = step_size;
    # normally 200 bins would be for 0 - 3500.  Now we are going
    # to say 3500/numbins instead.

    next_edge = current_edge+step_size;

    for line in f:
        spl = line.strip().split(',')
        #print line
        edge = spl[0]
        bkg = spl[1]
        sig = spl[2]
      
        running_bkg += float(bkg);
        running_sig += float(sig);
      
        if (float(edge) >= next_edge):
            next_edge += step_size;
            if (running_sig == 0):
                ptweights.SetBinContent( counter, 0 );
            else:
                ptweights.SetBinContent( counter, running_bkg/running_sig );
            running_bkg = 0;
            running_sig = 0;
            counter+=1;

    f.close();




def setupHistogram(fname, algorithm, treename, ptlow, pthigh, signal=False,ptfile=''):
    '''
    Read in the jet mass from the input file and fill it into a histogram.  The histogram gets used to calculate the mass window.
    Keyword args:
    fname --- input file name.
    algorithm --- Algorithm being used.
    treename --- treename for input root file
    '''
    # load the NEvents file to weight by number of events
    eventsfile = fname.replace('root','nevents')
    loadEvents(eventsfile)


    # set up a struct so that the SetBranchAddress method can be used.
    # normally one would use tree.variable to access the variable, but since
    # this variable name changes for each algorithm this is a better way to do it.
    gROOT.ProcessLine("struct jet_t { Float_t mass; Float_t pt; Float_t eta;} ")
    # create an object of this struct
    jet = jet_t()
    if signal:
        setupPtWeight(ptfile)
    
    # open file and ttree
    f = TFile.Open(fname,"READ")
    tree = f.Get(treename)
    
    # set branch address for the groomed jet mass
    tree.SetBranchAddress("jet_"+algorithm+"_m", AddressOf(jet,'mass'))
    tree.SetBranchAddress("jet_CamKt12Truth_pt", AddressOf(jet,'pt'))
    tree.SetBranchAddress("jet_CamKt12Truth_eta", AddressOf(jet,'eta'))
    # histogram with 300 bins and 0 - 0.3 TeV range
    hist = TH1F("mass","mass",100,0,300*1000)    
    # maximum number of entries in the tree
    entries = tree.GetEntries()
    
    # loop through
    for e in xrange(entries):
        tree.GetEntry(e)
        if e%1000 == 0:
            print 'Progress: ' + str(e) + '/' + str(entries)
        weight = 1*tree.mc_event_weight*(1/NEvents(tree.mc_channel_number))*tree.k_factor*tree.filter_eff
        #weight2 = 1*tree.mc_event_weight*tree.k_factor*tree.filter_eff*(1/NEvents(tree.mc_channel_number))

        if signal:
            weight*=ptWeight(jet.pt)
        else:
            weight*=tree.xs
        # fill the hist
        # check eta
        if abs(jet.eta) <= 1.2:
            if jet.pt < pthigh*1000 and jet.pt > ptlow*1000 and jet.mass < 300*1000 and jet.mass > 0:
            #if jet.pt < 1000*1000 and jet.pt > 500*1000 and jet.mass < 200*1000 and jet.mass > 0:
                if weight == 0:
                    print 'weight is 0'
                hist.Fill(jet.mass,weight)
    
    return hist

    
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

    # loop through each bin of the histogram
    for i in xrange(0,Nbins):

        tempFrac = 0.0
        # want to make sure we don't change i when changing imax
        imax = copy.deepcopy(i)

        # loop through until the tempFrac is above the frac (68%) criteria,
        # but making sure not to go out of range.                                                                                                                                             
        while(tempFrac<frac and imax != Nbins):
            tempFrac+=histo.GetBinContent(imax)/integral;
            imax+=1;

        width = histo.GetBinCenter(imax) - histo.GetBinCenter(i);
        # by applying this we say that the window we have just calculate MUST have at least 68%.
        if tempFrac >= frac and width<minWidth:
            # set up the best found mass window variables
            minWidth = width;
            topEdge = histo.GetBinCenter(imax);
            botEdge = histo.GetBinCenter(i)
            minidx = copy.deepcopy(i)
            maxidx = copy.deepcopy(imax)
            maxfrac = copy.deepcopy(tempFrac)

    return minWidth, topEdge, botEdge, minidx, maxidx#, maxfrac


def run(fname, algorithm, ptlow, pthigh,treename,ptfile):
    '''
    Method for running over a single algorithm and calculating the mass window.
    Keyword args:
    fname --- the input file name
    algorithm --- the name of the algorithm
    ptlow/high --- pt range
    treename --- Name of tree in input root files
    ptfile --- pt reweighting file
    '''

    # setup the histogram - read in the mass entries from the input file and put them into a histogram
    hist = setupHistogram(fname, algorithm, treename, float(ptlow), float(pthigh), True, ptfile)
    # calculate the width, top and bottom edges and the indices for the 68% mass window
    wid, topedge, botedge, minidx, maxidx = Qw(hist, 0.68)

    # folder where input file is
    folder = fname[:fname.rfind('/')+1]
    # write this information out to a text file
    fout = open(folder+algorithm+"_pt_"+ptlow+"_"+pthigh+"_masswindow.out",'w')
    fout.write("width: "+ str(wid)+'\n')
    fout.write("top edge: "+ str(topedge)+'\n')
    fout.write("bottom edge: "+ str(botedge)+'\n')
    fout.write("minidx: "+ str(minidx)+'\n')
    fout.write("maxidx: "+ str(maxidx)+'\n')
    fout.close()


def runAll(input_file, file_id, ptlow, pthigh, treename):
    '''
    Method for running over a collection of algorithms and calculating the mass window.  
    The algorithms are stored in a text file which is read in.  Each line gives the input folder, with the folder of the form ALGORITHM_fileID/.
    input_file --- the text file with the folders
    file_id --- the folder suffix
    ptlow/high --- the pt range
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
        ptfile = ""
        for fil in fileslist:
            # found the signal file!
            if fil.endswith("sig.root"):
                sigfile = folder+fil
            if fil.endswith("ptweightsv2"):
                ptfile = folder+fil
        # the algorithm can be obtained from the folder by removing the leading
        # directory information and removing the file identifier
        alg = l.split('/')[-2][:-len(file_id)]
        # add these to the list to run over
        fnames.append([sigfile,alg])

    # loop through all of the algorithms and find the mass windows
    for a in fnames:
        print "running " + a[1]
        run(a[0],a[1],ptlow,pthigh,treename,ptfile)

if __name__=="__main__":
    if len(sys.argv) < 3:
        print "usage: python calculateMassWindow.py text_file_with_folder_names folder_suffix(including leading underscore) pt-low(GeV) pt-high(GeV) [tree-name]"
        sys.exit()
    if len(sys.argv) == 5:
        runAll(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], 'physics')
    elif len(sys.argv) == 6:
        runAll(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
