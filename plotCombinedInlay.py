'''
This is a script used for plotting multiple files.  Specifically it is here to plot the ROC curves from lots of samples - MVA/ Cut-based/ AGILE.
run it with: python plotCombined.py config-file output-file

This can be run without a config file if it is needed to just plot a single ROC curve/ rejection etc. 
In this case usage is: python plotCombined.py input-file output-file [-rej Rej curve name] [-pow].

Args:
config-file is a csv file that contains:
       inputFile, roc_curve_name, legend_name, power_curve_name, colour, style, linestyle
    inputFile is the root file, roc_curve_name is the name of the TGraph(Errors) in the root file, and legend_name is the name to go on the legend.  Colour is optional. If not specified it will be done randomly.
It can also just be a ROOT file.  In this case, the curve name, legend name, power curve name and colour are all 

output-file is the name of the output roc. If the file name does not end in ".pdf", ".pdf" will be added.

'''

import os.path
import sys
import ROOT as rt
import argparse
import copy
from operator import itemgetter
import AtlasStyle as atlas
rt.gROOT.SetBatch(True)
import numpy as np
atlas.SetAtlasStyle()
import getParams as gp

parser = argparse.ArgumentParser(description="Plot ROC curves from ROOT files")
parser.add_argument('inputfile',help='Name of the input file.')
parser.add_argument('outputfile',help='Name of the output file.')
parser.add_argument('-s', help='Whether or not to plot a single ROOT file. This can be inferred from the file type of the input file.')
parser.add_argument('--rejection',nargs='+',help="The name of the rejection curve in the input file.")
parser.add_argument('--power',nargs='+',help="The name of the power curve in the input file.")
parser.add_argument('--legend',nargs='+',help="Legend for the plots.")
parser.add_argument('--decisionF',nargs='+',help="The name of the decision function in the input file.")
parser.add_argument('--proba',nargs='+',help="The name of the decision probabilities in the input file.")
parser.add_argument('--addJZ5',dest='addJZ5',action='store_true',help="The name of the decision probabilities in the input file.")
#sub_parser = parser.add_subparsers(help = 'Set the pT range')
#pt_parser = sub_parser.add_parser('--ptrange',help='pT range')
parser.add_argument('--ptlow',type=int,help='low pT range')
parser.add_argument('--pthigh',type=int,help='high pT range')


parser.set_defaults(addJZ5=False)
args = parser.parse_args()
if (args.ptlow and not args.pthigh) or (args.pthigh and not args.ptlow):
    print 'need to define ptlow AND pthigh!'
    sys.exit(0)

#print args
# small class to hold each entry
class rocEntry:
    import ROOT as rt
    import os
    def __init__(self,**kwargs):# tfile, roc_curve, legend, power_curve=''):
        keys = kwargs.keys()
        if 'file' not in keys and 'roc_curve' not in keys and 'legend' not in keys:
            print 'missing arguments'
            self.tfile = None
            self.roc = None
            self.legend = None
            return
        
        tfile = kwargs['file']
        roc_curve = kwargs['roc_curve']
        legend = kwargs['legend']
        self.legend = legend
        if 'algorithm' in keys:
            algorithm = kwargs['algorithm']
        else:
            # get algorithm from roc_curve name
            algorithm = roc_curve[len('roc_jet_'):roc_curve.find('_',len('roc_jet_'))]
        self.isDNN = self.legend.find('AGILE') != -1 or self.legend.find("DNN") != -1
        self.isBDT = self.legend.find("BDT") != -1
        if self.isDNN:
            algorithm = 'DNN ' + algorithm
        elif self.isBDT:
            algorithm = 'BDT ' + algorithm
            #check that the file exists
        if not os.path.isfile(tfile):
            print 'file does not exist: ' + tfile
            self.tfile = None
            self.roc = None
            self.legend = None
            self.algorithm = None
            return
        #print tfile
        self.tfile = rt.TFile(tfile)
        self.tfilename = tfile
        #print roc_curve
        self.roc = self.tfile.Get(roc_curve).Clone()
        if 'power_curve' in kwargs.keys():
            self.power_curve = self.tfile.Get(kwargs['power_curve']).Clone()
            self.power_curve.SetTitle('Power Curve: ' + legend)
        else:
            self.power_curve = None


        self.roc.SetTitle('ROC: ' + legend)
        self.colour = -1
        self.algorithm = algorithm
        self.tfile.Close()
        self.pt_low = 0
        self.pt_high = 0

        
    def setLegendMVA(self, resultsfile):
        self.resultsfile = resultsfile 
        if self.isDNN or self.isBDT:
            # get the results file name            
            if self.isDNN:
                tmpdict = gp.dnn(self.resultsfile)
            else:
                tmpdict = gp.bdt(self.resultsfile)
            # try to get the param id
            pos = self.tfilename.find('paramID_')+8
            pos_end = copy.deepcopy(pos)
            while self.tfilename[pos_end].isdigit():
                pos_end+=1
            self.taggerid = self.tfilename[pos:pos_end]
            #id position
            idpos = self.legend.find('ID')
            #print self.taggerid
            if idpos == -1:
                self.legend = tmpdict[self.taggerid]
                return
                # find next space
            while idpos < len(self.legend) and self.legend[idpos] != ' ':
                idpos+=1
            self.legend = tmpdict[self.taggerid] +' '+ self.legend[idpos+1:] # the +1 is to jump over the space

    def setPt(self,l,h):
        # get the pt range from the filename
        #splt = tfile[tfile.rfind('/')+1:-5].split('_') # the -5 removes the '.root'
        self.pt_low = int(l)
        self.pt_high = int(h)

    def setColour(self, colour):
        self.colour = int(colour)

    def setStyle(self, style):
        self.style = int(style)
        
    def setLineStyle(self, style):
        self.linestyle = int(style)

    def returnROC(self):
        return self.roc

    def returnLegend(self):
        return self.legend

    def returnPower(self):
        if self.power_curve is not None:
            return self.power_curve
        return None



#TWO FUNCTIONS 
def MakeReferenceGraph( roctype=0 , debug=0):
    '''Make reference graph for plotting'''

    graph = rt.TGraph(1)

    graph.SetTitle("")
    graph.GetXaxis().SetTitleOffset(1.2)
    graph.GetYaxis().SetTitleOffset(1.3)

    graph.GetXaxis().SetRangeUser(0.0,1.0)

    if roctype==0:
        #graph.GetXaxis().SetTitle("#epsilon ^{FullTag}_{W Jets}")
        #graph.GetYaxis().SetTitle("1- #epsilon ^{FullTag}_{QCD Jets}")
        graph.SetMinimum(0.0)
        graph.SetMaximum(1.0)
    elif roctype==1:
        #graph.GetXaxis().SetTitle("#epsilon ^{FullTag}_{W Jets}")
        #graph.GetYaxis().SetTitle("1 / #epsilon ^{FullTag}_{QCD Jets}")
        graph.SetMinimum(0.0)
        graph.SetMaximum(1000.0)

    return graph


minx1=0.0
maxx1=1.0
miny1=8.0
maxy1=2000.0
minx2=0.45
maxx2=0.55
# this works for jz5
miny2=30.0
maxy2=150.0
isMVA = False

logy=1

ylabel_power="1/#varepsilon ^{G&T}_{QCD Jets}"
ylabel_roc="1-#varepsilon ^{G&T}_{QCD Jets}"
xlabel_power="#varepsilon ^{G&T}_{W}"


algorithm_labels = {'AntiKt10LCTopoTrimmedPtFrac5SmallR20':'#splitline{anti-k_{t} R=1.0 jets}{#splitline{Trimmed}{f_{cut}=5%,R_{sub}=0.2}}',
                    'AntiKt10LCTopoTrimmedPtFrac5SmallR30':'#splitline{anti-k_{t} R=1.0 jets}{#splitline{Trimmed}{f_{cut}=5%,R_{sub}=0.3}}',
                    'CamKt10LCTopoPrunedCaRCutFactor50Zcut15':'#splitline{C/A R=1.0 jets}{#splitline{Pruned}{R_{cut}=0.5,z_{cut}=0.15}}',
                    'CamKt12LCTopoBDRSFilteredMU100Y4':'#splitline{C/A R=1.2 jets}{#splitline{Split filtered}{#mu=1, R_{sub}=0.3,y_{filt}=15%}}'}

                    
# store each line of the config file
roc_entry = []
    
# check if we are doing a single root file or reading in a config file
singleFile = False
if args.s or args.inputfile.endswith('root'):
    singleFile = True

    
if not singleFile:
    # read in config file
    config = open(args.inputfile)

    for c in config:
        c = c.strip()
        spl = c.split(',')
        if len(spl) <= 3 or c.startswith('#'):
            continue
        roc_args = {'file':spl[0], 'roc_curve': spl[1], 'legend': spl[2], 'power_curve': spl[3]}
        roc_entry.append(rocEntry(**roc_args))
        if len(spl) < 7:
            print 'config file line ' + c + ' is invalid.  Should have contents in csv format: file,roc_curve,legend,power_curve,colour,style,linestyle[,mva_legend,ptlow,pthigh]'
            continue
        if len(spl) >= 5: # add the colour and the marker style
            roc_entry[-1].setColour(spl[4])
            #roc_entry[-1].setStyle(spl[4])
            #roc_entry[-1].setLineStyle(spl[4])
        #if len(spl) >= 6:
            roc_entry[-1].setStyle(spl[5])
            #roc_entry[-1].setLineStyle(spl[4])
        #if len(spl) == 7:
            roc_entry[-1].setLineStyle(spl[6])
        if len(spl) == 8:
            legendfile = spl[7]
            roc_entry[-1].setLegendMVA(legendfile)
            isMVA = True
        # bah, gotta add in the fucking other things too
        if len(spl) == 10:
            roc_entry[-1].setPt(int(spl[8]),int(spl[9]))

else:
    if not args.rejection:
        print 'rejection string is not defined, exiting'
        sys.exit()

    # need to combine elements since nargs='+'
    
    leg_entry = " ".join(args.legend) if args.legend is not None else args.rejection
    roc_args = {'file': args.inputfile, 'roc_curve': " ".join(args.rejection), 'legend': leg_entry}
    if args.power:
        roc_args['power_curve'] = " ".join(args.power)

    entry = rocEntry(**roc_args)    
    roc_entry.append(entry)

                    
# should sort roc_entry on the grooming algorithm used.....
roc_entry.sort(key=lambda x: x.algorithm)
# get the names of unique algorithms
def getUniqueAlgs():
    global roc_entry
    unique_alg_list = {} # it's a list because I want it in order
    for r in roc_entry:        
        unique_alg_list[r.algorithm] = [1.0,0.0]
    return unique_alg_list
unique_algs = getUniqueAlgs()

                    
# now we can plot all of them.
tc = rt.TCanvas("Background Rejection")
tc.SetRightMargin(0.32)
tc.SetLeftMargin(0.15)
#tc.SetLogy(logy)

tc_df = rt.TCanvas("Decision Functions")
tc_proba = rt.TCanvas("Probability")
tc_power = rt.TCanvas("Background Power")
tc_power.SetRightMargin(0.32)
tc_power.SetLeftMargin(0.15)


tc_power.SetLogy()
legend = rt.TLegend(0.2,0.25,0.5,0.55);legend.SetFillColor(rt.kWhite)
#legend_power = rt.TLegend(0.7,0.55,0.9,0.85);legend_power.SetFillColor(rt.kWhite)
markers_1 = np.linspace(1,8,8)
markers_2 = np.linspace(20,34,15)
markers = np.hstack([markers_1,markers_2])

# find the minimum background rejection
minrej = 1.0
for r in roc_entry:
    n = r.roc.GetN()
    y = r.roc.GetY()
    locmin = rt.TMath.LocMin(n, y)
    minval = y[locmin]
    minrej = min(minval, minrej)


    
# first plot the roc curves
ystart = 0.85
dy = 0.045
dy_dnn = 0.052
nplot = 0
curr_algorithm = ''
prev_algorithm = 'default'
sameAlg = False
offset = 0
#if we're doing lots on one plot then use smaller spacing
smaller = False
if len(roc_entry) > 12:
    smaller = True
    ystart = 0.88
for i, r in enumerate(roc_entry):
    # set the colour
    nplot+=1
    curr_algorithm = r.algorithm
    if prev_algorithm != curr_algorithm:
        #if nplot != 1:
        #    nplot+=0.5
        startpos = ystart-nplot*dy_dnn
        endpos = startpos + 0.05
        unique_algs[r.algorithm] = [endpos, startpos]
        sameAlg = False
        if smaller:
            nplot+=0.3
        else:
            nplot+=0.4
        offset += 0.1
    else:
        sameAlg = True
    
    col = int(r.colour) if r.colour != -1 else int(markers[i%len(markers)])
    style = r.style if r.colour != -1 else int(markers[i%len(markers)])
    linestyle = r.linestyle if r.colour != -1 else int(markers[i%len(markers)]%10)


    legend_first = r.legend
    legend_last = ''
    if r.isDNN:
        # first half
        if smaller:
            dy_dnn = 0.05
            #nplot+=0.2
        idx = r.legend.find(', u=')
        if idx != -1:
            legend_first = r.legend[:idx+1]
            legend_last = r.legend[idx+1:]

    elif r.isBDT:
        # first half
        idx = r.legend.find(')')
        if smaller:
            dy_dnn = 0.05
        #nplot+=0.1
        if idx != -1:
            legend_first = r.legend[:idx+1]
            legend_last = r.legend[idx+1:]

        
    r.roc.SetMinimum(minrej*0.98)
    r.roc.SetMarkerColor(col)
    r.roc.SetLineColor(col)
    # set the marker and line style
    r.roc.SetMarkerStyle(style);r.roc.SetMarkerSize(1);
    r.roc.SetLineStyle(linestyle)
    r.roc.GetYaxis().SetTitle(ylabel_roc)
    r.roc.GetXaxis().SetTitle(xlabel_power)
    #r.roc.
    # draw the roc curves
    tc.cd()
    if i == 0:
        r.roc.Draw()
        r.roc.GetXaxis().SetRangeUser(0.,0.75)
        r.roc.Draw()
    else:
        r.roc.Draw('same')
        r.roc.GetXaxis().SetRangeUser(0.,0.75)
        r.roc.Draw('same')
    #legend.AddEntry(r.roc, r.legend)
    if r.isBDT or r.isDNN:
        atlas.myLineText(0.76, ystart-nplot*dy_dnn, 0.05, col, linestyle, 0.025, legend_first)
        atlas.myText(0.78, ystart-nplot*dy_dnn-0.033, 1, 0.025, legend_last)
    else:
        atlas.myLineText(0.76, ystart-nplot*dy_dnn, 0.05, col, linestyle, 0.025, r.legend)

    # draw the power curves
    if r.power_curve is None:
        pass
    # set the colour
    r.power_curve.SetMarkerColor(col)
    r.power_curve.SetLineColor(col)
    # set the marker and line style
    r.power_curve.SetMarkerStyle(style);r.power_curve.SetMarkerSize(1)
    r.power_curve.SetLineStyle(linestyle)
    r.power_curve.GetYaxis().SetTitle(ylabel_power)
    r.power_curve.GetXaxis().SetTitle(xlabel_power)
    tc_power.cd()
    if i == 0:
        r.power_curve.Draw()
    else:
        r.power_curve.Draw('same')
    # dnn label must go over two lines...
    if r.isBDT or r.isDNN:
        atlas.myLineText(0.76, ystart-nplot*dy_dnn, 0.05, col, linestyle, 0.025, legend_first)
        atlas.myText(0.78, ystart-nplot*dy_dnn-0.033, 1, 0.025, legend_last)
    else:
        atlas.myLineText(0.76, ystart-nplot*dy_dnn, 0.05, col, linestyle, 0.025, r.legend)

    #print 0.8-nplot*dy
    unique_algs[r.algorithm] = [max(ystart-nplot*dy_dnn,unique_algs[r.algorithm][0]), min(ystart-nplot*dy_dnn,unique_algs[r.algorithm][1])]
    #startpos = 0.8-nplot*dy# no this is not right
    #legend_power.AddEntry(r.power_curve, r.legend)
    prev_algorithm = copy.deepcopy(curr_algorithm)

# check if pt range is defined in args
if args.ptlow and args.pthigh:
    pt_low = int(args.ptlow)
    pt_high = int(args.pthigh)
else:
    pt_low = roc_entry[0].pt_low
    pt_high = roc_entry[0].pt_high
    
for canvas in [tc_power, tc]:
    # try adding the algorithm name
    canvas.cd()
    # add the pt range, sqrt{s}=13 TeV
    atlas.myText    (0.7, 0.90, 1, 0.03, "#sqrt{s} = 13 TeV, "+str(pt_low)+"<p_{T}^{Truth}<"+str(pt_high)+" GeV")# any roc entry should ahve the same pt limits
    # what if showing full + jz5 roc curve? there are two pt ranges!!!!
    if args.addJZ5:
        atlas.myText    (0.8, 0.88, 1, 0.03, "800<p_{T}^{Truth}<1200 GeV (JZ5)")# any roc entry should ahve the same pt limits
    atlas.myText    (0.7, 0.85, 1, 0.03, "68% mass window cut applied")
    # if we only have a single algorithm present then we can just put that at the top
    unique_keys = unique_algs.keys()
    if len(unique_keys) == 1:
        key = unique_keys[0]
        atlas.myText(0.81, unique_algs[key][0]-abs(unique_algs[key][0]-unique_algs[key][1])/2.-0.02, rt.kBlack, 0.025, algorithm_labels[key])
        atlas.myText(0.81, unique_algs[key][0]-abs(unique_algs[key][0]-unique_algs[key][1])/2.-0.02, rt.kBlack, 0.025, algorithm_labels[key])
    else:
        for a in unique_keys:
            if a in algorithm_labels.keys():
                atlas.myText(0.81, unique_algs[a][0]-abs(unique_algs[a][0]-unique_algs[a][1])/2.-0.02, rt.kBlack, 0.025, algorithm_labels[a])
    
#print unique_algs
output_file = args.outputfile
output_file = output_file.replace('.root','')

# entry 0: for power, 1 for power with mva, 2 for roc
inlayx1=[0.43,0.16,0.2]
inlayy1=[0.51,0.18,0.2]
inlayx2=[0.67,0.38,0.43]
inlayy2=[0.91,0.51,0.61]
inlayx1roc=0.23
inlayy1roc=0.11
inlayx2roc=0.47
inlayy2roc=0.51
def drawInlay(power=False):
    global tc_power, tc, roc_entry, output_file
    idx = 0 # this is power, 1 is roc!
    if isMVA:
        idx = 1
    if power:
        tc_power.cd()
    else:
        idx = 2
        tc.cd()
    pinlay = rt.TPad("p","p",inlayx1[idx],inlayy1[idx],inlayx2[idx],inlayy2[idx]); # the inlay is in the wrong place and has the wrong limits when looking at ROC!
    pinlay.SetLeftMargin(0.13);
    pinlay.SetRightMargin(0.02);
    pinlay.SetTopMargin(0.02);
    pinlay.SetBottomMargin(0.13);
    pinlay.Draw();
    pinlay.cd();
    pinlay.SetGrid()
    grefinlay = MakeReferenceGraph( 0 );

    #grefinlay.GetYaxis().SetTitle(ylabel)

    if power:
        grefinlay.SetMinimum(miny2)
        grefinlay.SetMaximum(maxy2)
        grefinlay.GetXaxis().SetRangeUser(minx2,maxx2)
    else:
        grefinlay.SetMinimum(0.95)
        grefinlay.SetMaximum(1.0)
        grefinlay.GetXaxis().SetRangeUser(0.4,0.6)
    grefinlay.Draw("AC");
    for r in roc_entry:
        if power:
            r.power_curve.Draw('csame')
        else:
            r.roc.Draw('csame')
    if power:
        tc_power.cd()
        #legend_power.Draw('same')
        tc_power.SaveAs(output_file+'_power.pdf')
    else:
        tc.cd()
        tc.SaveAs(output_file+'_roc.pdf')


drawInlay(True)
drawInlay(False)
            
# save the roc_curves
#tc.cd()
#legend.Draw('same')




