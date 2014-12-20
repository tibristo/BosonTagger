import ROOT
def setweights(weights):
    # right now we are not applying the k-factors, the first set of xs weights are in pb
    weights['signal'] = 1.0 * 1.000000 * (1.0/1.0) # * 1.00 
    weights['140_280_CJetVetoBVeto'] = 20100 * 31.10010 * (1/10348.6/0.72) #* 1.11
    weights['140_280_CJetFilterBVeto'] = 20100 * 31.10010 * (1/16681.4/0.21) #* 1.11
    weights['140_280_BFilter'] = 20100 * 31.10010 * (1/81338.2/0.06) #* 1.11
    weights['280_500_CJetVetoBVeto'] = 20100 * 1.837170 * (1/14243.9/0.69) #* 1.11
    weights['280_500_CJetFilterBVeto'] = 20100 * 1.837170 * (1/41406.9/0.23) #* 1.11
    weights['280_500_BFilter'] = 20100 * 1.837170 * (1/109731 /0.08) # 1.11 *
    weights['500_CJetVetoBVeto'] = 20100 * 0.100100 * (1/2022.2 /0.66) # 1.11 *
    weights['500_CJetFilterBVeto'] = 20100 * 0.100100 * (1/2037.5 /0.24) # 1.11 *
    weights['500_BFilter'] = 20100 * 0.100100 * (1/10522.3/0.10) #1.11 * 
    # these cross sections are in nb right now
    weights['JZ3W'] = 1.6664E+03*1.9139E-03 # RunNumber 147913
    weights['JZ4W'] = 2.7646E+01*1.4296E-03 # RunNumber 147914
    weights['JZ5W'] = 3.0317E-01*5.5040E-03 # RunNumber 147915
    weights['JZ6W'] = 7.5078E-03*1.5252E-02 # RunNumber 147916
    weights['JZ7W'] = 1.3760E-03*7.6369E-02 # RunNumber 147917


def setrunnumbers(runs):
    runs[147913] = 'JZ3W'
    runs[147914] = 'JZ4W'
    runs[147915] = 'JZ5W'
    runs[147916] = 'JZ6W'
    runs[147917] = 'JZ7W'
    #signal
    runs[158225] = 'signal'
    runs[158226] = 'signal'
    runs[158227] = 'signal'
    runs[158228] = 'signal'
    runs[158229] = 'signal'
    runs[158230] = 'signal'
    runs[158231] = 'signal'
    runs[158232] = 'signal'
    runs[158233] = 'signal'
    runs[158234] = 'signal'
    runs[158235] = 'signal'
    runs[158236] = 'signal'
    runs[158237] = 'signal'
    runs[158238] = 'signal'
    runs[158239] = 'signal'
    runs[158240] = 'signal'
    runs[158241] = 'signal'
    runs[158242] = 'signal'
    # backgrounds
    runs[167770] = '140_280_BFilter'
    runs[167771] = '140_280_CJetFilterBVeto'
    runs[167772] = '140_280_CJetVetoBVeto'
    runs[167773] = '140_280_BFilter'
    runs[167774] = '140_280_CJetFilterBVeto'
    runs[167775] = '140_280_CJetVetoBVeto'
    runs[167776] = '140_280_BFilter'
    runs[167777] = '140_280_CJetFilterBVeto'
    runs[167778] = '140_280_CJetVetoBVeto'
    runs[167779] = '280_500_BFilter'
    runs[167780] = '280_500_CJetFilterBVeto'
    runs[167781] = '280_500_CJetVetoBVeto'
    runs[167782] = '280_500_BFilter'
    runs[167783] = '280_500_CJetFilterBVeto'
    runs[167784] = '280_500_CJetVetoBVeto'
    runs[167785] = '280_500_BFilter'
    runs[167786] = '280_500_CJetFilterBVeto'
    runs[167787] = '280_500_CJetVetoBVeto'
    runs[167788] = '500_BFilter'
    runs[167789] = '500_CJetFilterBVeto'
    runs[167790] = '500_CJetVetoBVeto'
    runs[167791] = '500_BFilter'
    runs[167792] = '500_CJetFilterBVeto'
    runs[167793] = '500_CJetVetoBVeto'
    runs[167794] = '500_BFilter'
    runs[167795] = '500_CJetFilterBVeto'
    runs[167796] = '500_CJetVetoBVeto'
    runs[167809] = '140_280_BFilter'
    runs[167810] = '140_280_CFilterBVeto'
    runs[167811] = '140_280_CVetoBVeto'
    runs[167812] = '140_280_BFilter'
    runs[167813] = '140_280_CFilterBVeto'
    runs[167814] = '140_280_CVetoBVeto'
    runs[167815] = '140_280_BFilter'
    runs[167816] = '140_280_CFilterBVeto'
    runs[167817] = '140_280_CVetoBVeto'
    runs[167821] = '280_500_BFilter'
    runs[167822] = '280_500_CFilterBVeto'
    runs[167823] = '280_500_CVetoBVeto'
    runs[167824] = '280_500_BFilter'
    runs[167825] = '280_500_CFilterBVeto'
    runs[167826] = '280_500_CVetoBVeto'
    runs[167827] = '280_500_BFilter'
    runs[167828] = '280_500_CFilterBVeto'
    runs[167829] = '280_500_CVetoBVeto'
    runs[167833] = '500_BFilter'
    runs[167834] = '500_CFilterBVeto'
    runs[167835] = '500_CVetoBVeto'
    runs[167836] = '500_BFilter'
    runs[167837] = '500_CFilterBVeto'
    runs[167838] = '500_CVetoBVeto'
    runs[167839] = '500_BFilter'
    runs[167840] = '500_CFilterBVeto'
    runs[167841] = '500_CVetoBVeto'
    # not using these yet....
    '''
    117050,PowhegPythia_P2011C_ttbar,253.00,1.000000,0.543,
    105200,McAtNloJimmy_CT10_ttbar_LeptonFilter,253.00,1.000000,0.543,
    117360,AcerMCPythia_AUET2BCTEQ6L1_singletop_tchan_e,8.604,1.10,1.000000,
    117361,AcerMCPythia_AUET2BCTEQ6L1_singletop_tchan_mu,8.604,1.10,1.000000,
    117362,AcerMCPythia_AUET2BCTEQ6L1_singletop_tchan_tau,8.604,1.10,1.000000,
    108343,McAtNloJimmy_CT10NLOME_AUET2CTEQ6L1MPI_SingleTopSChanWenu,0.564440,1.074,1.000000,
    108344,McAtNloJimmy_CT10NLOME_AUET2CTEQ6L1MPI_SingleTopSChanWmunu,0.564260,1.074,1.000000,
    108345,McAtNloJimmy_CT10NLOME_AUET2CTEQ6L1MPI_SingleTopSChanWtaunu,0.564040,1.074,1.000000,
    108346,McAtNloJimmy_CT10NLOME_AUET2CTEQ6L1MPI_SingleTopWtChanIncl,20.658000,1.083,1.000000,
    '''

def pTReweight(hist_sig, hist_bkg, algorithm, varBinPt, xbins):
    from ROOT import *
    from array import array
    bins = hist_bkg.GetNbinsX()
    name = 'ptreweight'+algorithm
    if not varBinPt:
        hist_reweight = TH1D(name,name, bins, 0, hist_bkg.GetXaxis().GetBinUpEdge(bins))
    else:
        hist_reweight = TH1D(name,name, len(xbins)-1, array('d',xbins))
    for x in range(1,bins):
        if hist_sig.GetBinContent(x) == 0:
            weight = -1
        else:
            weight = hist_bkg.GetBinContent(x)/hist_sig.GetBinContent(x)
        hist_reweight.SetBinContent(x,weight)
        print str(hist_sig.GetXaxis().GetBinLowEdge(x)/1000) + ' - ' + str(hist_sig.GetXaxis().GetBinUpEdge(x)/1000) + ' weight: ' + str(weight)
    tc = TCanvas("ptr")
    hist_reweight.Draw('e')
    tc.SaveAs('pt_reweight'+algorithm+'.png')

def getFileIDNumber(inputdir):
    if inputdir.lower().find('mu100') != -1:
        return '2'
    elif inputdir.lower().find('mu67') != -1:
        return '1'
    elif inputdir.lower().find('trim') != -1:
        return '3'
    else:
        return ''

def getFileBranches(inputfile, treename='physics'):
    file_branches = []
    file_branches_stubs = []
    file_in = ROOT.TFile(inputfile)
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
        if not branches[k][0][1:] in file_branches:

            todelete.append(k)

    for d in todelete:
        del branches[d]

    todeleteplot = []
    for k in plotbranches.keys():
        if not plotbranches[k][0][1:] in file_branches:
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
    from ROOT import TLatex 
    texw = TLatex();
    texw.SetNDC();
    texw.SetTextSize(0.045);
    texw.SetTextFont(72);
    texw.DrawLatex(0.6,0.88,"ATLAS");
    
    p = TLatex();
    p.SetNDC();
    p.SetTextFont(42);
    p.SetTextSize(0.045);
    p.SetTextColor(ROOT.kBlack);
    p.DrawLatex(0.71,0.88,"Internal Simulation");

    p = TLatex();
    p.SetNDC();
    p.SetTextFont(42);
    p.SetTextSize(0.035);
    p.SetTextColor(ROOT.kBlack);
    p.DrawLatex(0.65,0.64,"#sqrt{s} = "+str(E)+" TeV");
    
    p2 = TLatex();
    p2.SetNDC();
    p2.SetTextFont(42);
    p2.SetTextSize(0.035);
    p2.SetTextColor(ROOT.kBlack);
    p2.DrawLatex(0.65,0.82,algo);
    
    p2 = TLatex();
    p2.SetNDC();
    p2.SetTextFont(42);
    p2.SetTextSize(0.035);
    p2.SetTextColor(ROOT.kBlack);
    p2.DrawLatex(0.65,0.76,algosettings);
    
    p3 = TLatex();
    p3.SetNDC();
    p3.SetTextFont(42);
    p3.SetTextSize(0.035);
    p3.SetTextColor(ROOT.kBlack);
    p3.DrawLatex(0.65,0.70,str(ptrange[0]/1000.0)+' < p_{T} < ' + str(ptrange[1]/1000) + ' GeV')#, '+str(nvtxrange[0])+'<nvtx<'+str(nvtxrange[1]));

def drawHists(hist1, hist2):
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
            
            PointNumber = -1 * PointNumber;
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
