'''
This is a script used for plotting multiple files.  Specifically it is here to plot the ROC curves from lots of samples - MVA/ Cut-based/ AGILE.

run it with python plotCombined.py config-file output-file

Args:
config-file is a csv file that contains:
       inputFile, roc_curve_name, legend_name, colour
    inputFile is the root file, roc_curve_name is the name of the TGraph(Errors) in the root file, and legend_name is the name to go on the legend.  Colour is optional. If not specified it will be done randomly.

output-file is the name of the output roc. If the file name does not end in ".pdf", ".pdf" will be added.

'''

import os.path
import sys
import ROOT as rt

# check that we have enough arguments
if len(sys.argv) <=2 :
    print 'not enough arguments! Usage is python plotCombined.py config-file output-file'
    sys.exit()

# small class to hold each entry
class rocEntry:
    import ROOT as rt
    import os
    def __init__(self, tfile, roc_curve, legend):
        #check that the file exists
        if not os.path.exists(tfile):
            self.tfile = None
            self.roc = None
            self.legend = None
            return
        self.tfile = rt.TFile(tfile)
        self.roc = self.tfile.Get(roc_curve).Clone()
        self.roc.SetDirectory(0)
        self.tfile.Close()
        self.legend = legend
        self.roc.SetTitle('ROC: ' + legend)

    def setColour(self, colour):
        self.colour = colour

    def returnROC(self):
        return self.roc

    def returnLegend(self):
        return self.legend

    
# store each line of the config file
roc_entry = []
    
# read in config file
config = open(sys.argv[1])

for c in config:
    c = c.strip()
    spl = c.split(',')
    if len(spl) != 3:
        pass
    roc_entry.append(rocEntry(spl[0], spl[1], spl[2]))
    if len(spl) == 4: # add the colour
        roc_entry[-1].SetColour(spl[3])

# now we can plot all of them.
tc = TCanvas()
legend = TLegend()

for i, r in enumerate(roc_entry):
    # set the colour
    r.roc.SetMarkerColor()
    r.roc.SetMarkerStyle()
    r.Draw('same')
    legend.Add()

tc.SaveAs()

