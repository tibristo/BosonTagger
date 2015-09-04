'''
This is a script used for plotting multiple files.  Specifically it is here to plot the ROC curves from lots of samples - MVA/ Cut-based/ AGILE.

run it with python plotCombined.py config-file output-file

Args:
config-file is a csv file that contains:
       inputFile, roc_curve_name, legend_name, power_curve_name, colour
    inputFile is the root file, roc_curve_name is the name of the TGraph(Errors) in the root file, and legend_name is the name to go on the legend.  Colour is optional. If not specified it will be done randomly.

output-file is the name of the output roc. If the file name does not end in ".pdf", ".pdf" will be added.

'''

import os.path
import sys
import ROOT as rt
import AtlasStyle as atlas
rt.gROOT.SetBatch(True)
import numpy as np
atlas.SetAtlasStyle()
# check that we have enough arguments
if len(sys.argv) <=2 :
    print 'not enough arguments! Usage is python plotCombined.py config-file output-file'
    sys.exit()

# small class to hold each entry
class rocEntry:
    import ROOT as rt
    import os
    def __init__(self, tfile, roc_curve, legend, power_curve=''):
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
        if power_curve != '':
            self.power_curve = self.tfile.Get(power_curve).Clone()
            self.power_curve.SetTitle('Power Curve: ' + legend)
        else:
            self.power_curve = None

        self.tfile.Close()
        self.legend = legend
        self.roc.SetTitle('ROC: ' + legend)
        self.colour = -1


    def setColour(self, colour):
        self.colour = colour

    def setStyle(self, style):
        self.style = style
        
    def setLineStyle(self, style):
        self.linestyle = style

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
    
# read in config file
config = open(sys.argv[1])

for c in config:
    c = c.strip()
    spl = c.split(',')
    if len(spl) != 3:
        pass
    roc_entry.append(rocEntry(spl[0], spl[1], spl[2], spl[3]))
    if len(spl) >= 5: # add the colour and the marker style
        roc_entry[-1].setColour(spl[4])
        roc_entry[-1].setStyle(spl[4])
        roc_entry[-1].setLineStyle(spl[4])
    if len(spl) >= 6:
        roc_entry[-1].setStyle(spl[5])
        roc_entry[-1].setLineStyle(spl[4])
    if len(spl) == 7:
        roc_entry[-1].setLineStyle(spl[6])

# now we can plot all of them.
tc = rt.TCanvas("Background Rejection")
tc_power = rt.TCanvas("Background Power")
tc_power.SetLogy()
legend = rt.TLegend(0.7,0.55,0.9,0.85);legend.SetFillColor(rt.kWhite)
legend_power = rt.TLegend(0.7,0.55,0.9,0.85);legend_power.SetFillColor(rt.kWhite)
markers_1 = np.linspace(1,8,8)
markers_2 = np.linspace(20,34,15)
markers = np.hstack([markers_1,markers_2])
# first plot the roc curves
for i, r in enumerate(roc_entry):
    # set the colour
    if r.colour != -1:
        r.roc.SetMarkerColor(int(r.colour))
        r.roc.SetLineColor(int(r.colour))
        # set the marker and line style
        r.roc.SetMarkerStyle(int(r.style))
        r.roc.SetLineStyle(int(r.linestyle))
    else:
        r.roc.SetMarkerColor(int(markers[i%len(markers)]))
        r.roc.SetLineColor(int(markers[i%len(markers)]))
        # set the marker and line style
        r.roc.SetMarkerStyle(int(markers[i%len(markers)]))
        r.roc.SetLineStyle(int(markers[i%len(markers)])%10)
    r.roc.SetMarkerSize(1);
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
    if r.colour != -1:
        r.power_curve.SetMarkerColor(int(r.colour))
        r.power_curve.SetLineColor(int(r.colour))
        # set the marker and line style
        r.power_curve.SetMarkerStyle(int(r.style))
        r.power_curve.SetLineStyle(int(r.linestyle))
    else:
        r.power_curve.SetMarkerColor(int(markers[i%len(markers)]))
        r.power_curve.SetLineColor(int(markers[i%len(markers)]))
        # set the marker and line style
        r.power_curve.SetMarkerStyle(int(markers[i%len(markers)]))
        r.power_curve.SetLineStyle(int(markers[i%len(markers)])%10)
    r.power_curve.SetMarkerSize(1)
    tc_power.cd()
    if i == 0:
        r.power_curve.Draw()
    else:
        r.power_curve.Draw('same')
    legend_power.AddEntry(r.power_curve, r.legend)

# save the roc_curves
tc.cd()
legend.Draw('same')
output_file = sys.argv[2]
tc.SaveAs(output_file+'_roc.pdf')


tc_power.cd()
legend_power.Draw('same')
tc_power.SaveAs(output_file+'_power.pdf')
