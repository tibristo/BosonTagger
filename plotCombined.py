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
import AtlasStyle as atlas
rt.gROOT.SetBatch(True)
import numpy as np
atlas.SetAtlasStyle()

parser = argparse.ArgumentParser(description="Plot ROC curves from ROOT files")
parser.add_argument('inputfile',help='Name of the input file.')
parser.add_argument('outputfile',help='Name of the output file.')
parser.add_argument('-s', help='Whether or not to plot a single ROOT file. This can be inferred from the file type of the input file.')
parser.add_argument('--rejection',nargs='+',help="The name of the rejection curve in the input file.")
parser.add_argument('--power',nargs='+',help="The name of the power curve in the input file.")
parser.add_argument('--legend',nargs='+',help="Legend for the plots.")
parser.add_argument('--decisionF',nargs='+',help="The name of the decision function in the input file.")
parser.add_argument('--proba',nargs='+',help="The name of the decision probabilities in the input file.")

args = parser.parse_args()
print args
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
        #check that the file exists
        if not os.path.isfile(tfile):
            print 'file does not exist: ' + tfile
            self.tfile = None
            self.roc = None
            self.legend = None
            return
        print tfile
        self.tfile = rt.TFile(tfile)
        print roc_curve
        self.roc = self.tfile.Get(roc_curve).Clone()
        if 'power_curve' in kwargs.keys():
            self.power_curve = self.tfile.Get(kwargs['power_curve']).Clone()
            self.power_curve.SetTitle('Power Curve: ' + legend)
        else:
            self.power_curve = None

        self.legend = legend
        self.roc.SetTitle('ROC: ' + legend)
        self.colour = -1

        self.tfile.Close()


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
        if len(spl) <= 3:
            pass
        roc_args = {'file':spl[0], 'roc_curve': spl[1], 'legend': spl[2], 'power_curve': spl[3]}
        roc_entry.append(rocEntry(**roc_args))
        if len(spl) >= 5: # add the colour and the marker style
            roc_entry[-1].setColour(spl[4])
            roc_entry[-1].setStyle(spl[4])
            roc_entry[-1].setLineStyle(spl[4])
        if len(spl) >= 6:
            roc_entry[-1].setStyle(spl[5])
            roc_entry[-1].setLineStyle(spl[4])
        if len(spl) == 7:
            roc_entry[-1].setLineStyle(spl[6])
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

# now we can plot all of them.
tc = rt.TCanvas("Background Rejection")
tc_df = rt.TCanvas("Decision Functions")
tc_proba = rt.TCanvas("Probability")
tc_power = rt.TCanvas("Background Power")
tc_power.SetLogy()
legend = rt.TLegend(0.2,0.25,0.5,0.45);legend.SetFillColor(rt.kWhite)
legend_power = rt.TLegend(0.7,0.55,0.9,0.85);legend_power.SetFillColor(rt.kWhite)
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
for i, r in enumerate(roc_entry):
    # set the colour
    col = int(r.colour) if r.colour != -1 else int(markers[i%len(markers)])
    style = r.style if r.colour != -1 else int(markers[i%len(markers)])
    linestyle = r.linestyle if r.colour != -1 else int(markers[i%len(markers)]%10)

    r.roc.SetMinimum(minrej*0.98)
    r.roc.SetMarkerColor(col)
    r.roc.SetLineColor(col)
    # set the marker and line style
    r.roc.SetMarkerStyle(style);r.roc.SetMarkerSize(1);
    r.roc.SetLineStyle(linestyle)
    # draw the roc curves
    tc.cd()
    if i == 0:
        r.roc.Draw()
    else:
        r.roc.Draw('same')
    legend.AddEntry(r.roc, r.legend)

    # draw the power curves
    if r.power_curve is None:
        pass
    # set the colour
    r.power_curve.SetMarkerColor(col)
    r.power_curve.SetLineColor(col)
    # set the marker and line style
    r.power_curve.SetMarkerStyle(style);r.power_curve.SetMarkerSize(1)
    r.power_curve.SetLineStyle(linestyle)
    tc_power.cd()
    if i == 0:
        r.power_curve.Draw()
    else:
        r.power_curve.Draw('same')
    legend_power.AddEntry(r.power_curve, r.legend)


# save the roc_curves
tc.cd()
legend.Draw('same')
output_file = args.outputfile
output_file = output_file.replace('.root','')
tc.SaveAs(output_file+'_roc.pdf')


tc_power.cd()
legend_power.Draw('same')
tc_power.SaveAs(output_file+'_power.pdf')
