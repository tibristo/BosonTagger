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

def writeResponsePlots(weightedxAOD, Algorithm, plotconfig, trees, cutstring, fileid, ptreweight = True, varpath = "", mass_min = "0.0", mass_max = "1200000.0", scaleLumi = 1, applyMassWindow = True):
    '''
    Run through the Algorithm for a given mass range.  Returns the bkg rej at 50% signal eff.
    Keyword args:
    weightedxAOD -- bool indicating if the xaod is already weighted.
    Algorithm -- Name of the algorithm.  Set in main, comes from config file.
    plotconfig -- Contains the settings for the plot like bin number and hist limits. The key of the dict is the variable name.
    trees -- contains the actual data.
    cutstring -- basic selection cuts to be applied.
    fileid -- File identifier that gets used in the output file names
    ptreweight -- Reweight signal according to pT
    varpath -- Output directory
    mass_min -- Mass window minimum
    mass_max -- Mass window maximum
    scaleLumi -- luminosity scale factor
    applyMassWindow -- Whether or not to apply the 68% mass window cut when plotting.

    Writes out the response distributions for signal and background and then returns the dictionary of histograms.
    '''

    #global weightedxAOD
    hist = {}
    variables = []
    # remove all of the leading underscores
    for x, v in enumerate(plotconfig.keys()):
        stub = plotconfig[v][STUB]
        if not plotconfig[v][1] == True: # if it is not a jet variable
            continue
        if stub.startswith('_'):
            stub = stub[1:]
        variables.append(stub)

    # create all of the histograms
    for typename in ['sig','bkg']:
        histnamestub = typename + '_jet_' + Algorithm + '_response' 
        for br in plotconfig.keys(): # variable names
            if plotconfig[br][1] == True: # if it is a jet variable
                histname = histnamestub+plotconfig[br][STUB]
            else:
                continue
            hist_title = br
            hist[histname] = TH1D(histname, hist_title, 100, 0, 4)#plotconfig[br][MINX], plotconfig[br][MAXX])
            hist[histname].SetYTitle("Normalised Entries")
            hist[histname].SetXTitle("response " + br)
  


    # set the cutstring
    if applyMassWindow:
        cutstring_mass = cutstring+ " * (jet_" +Algorithm + "_m <= " +mass_max+ ")" + " * (jet_" +Algorithm + "_m > " +mass_min+ ") " 
    else:
        cutstring_mass = cutstring

    # the colours for the background and signal
    col_sig = 4
    col_bkg = 2

    # loop through all of the variables
    for index, branchname in enumerate(variables):
        col = 1
        responseName = "jet_" +Algorithm+'_response_' + branchname
        # loop through the background and signal input trees
        for indexin, datatype in enumerate(trees):
            histname =  datatype + "_jet_" +Algorithm+'_response_' + branchname
            varexp = 'response_'+branchname + '>>' + histname
            # the weights are stored in the ntuple for the weighted xAODs.
            if not weightedxAOD:
                cutstringandweight = '*mc_event_weight*1./NEvents(mc_channel_number)'
            else:
                cutstringandweight = '*mc_event_weight*1./evt_nEvts'
            # add the cross section and filter efficiency for the background            
            if datatype == 'bkg': 
                # according to Chris we don't apply the kfactor
                if not weightedxAOD:
                    cutstringandweight += '*filter_eff*xs'#*k_factor'
                else:
                    cutstringandweight += '*evt_filtereff*evt_xsec'#*evt_kfactor'
                hist[histname].SetMarkerStyle(21)
                col = col_bkg
            # filter efficiency for signal
            elif datatype == 'sig':
                # according to Chris we do not apply any of these to signal
                # so reset the cutstring
                cutstringandweight = ''
                #if not weightedxAOD:
                #    cutstringandweight += '*filter_eff'
                #else:
                #    cutstringandweight += '*evt_filtereff'
                # apply pt reweighting to the signal
                if ptreweight:
                    cutstringandweight +='*SignalPtWeight3(jet_CamKt12Truth_pt)'

                col = col_sig
            
            hist[histname].Sumw2();
            #apply the selection.
            trees[datatype].Draw(varexp,cutstring_mass+cutstringandweight)
            # scale the histogram
            if hist[histname].Integral() > 0.0:
                if scaleLumi != 1:
                    hist[histname].Scale(scaleLumi);
                else:
                    hist[histname].Scale(1.0/hist[histname].Integral());
            # set the style for the histogram.
            hist[histname].SetLineStyle(1); hist[histname].SetFillStyle(0); hist[histname].SetMarkerSize(1);
            hist[histname].SetFillColor(col); hist[histname].SetLineColor(col); hist[histname].SetMarkerColor(col); 
        # need to clear leg1
        # create a tlegend to go on the plot
        leg1 = TLegend(0.8,0.55,0.9,0.65);leg1.SetFillColor(kWhite)
        leg1.AddEntry(hist["sig_" + responseName],"W jets","l");    leg1.AddEntry(hist["bkg_" + responseName],"QCD jets","l");
        # plot the maximum histogram
        canv1 = TCanvas("tempCanvas")
        if (hist['sig_'+responseName].GetMaximum() > hist['bkg_'+responseName].GetMaximum()):
            fn.drawHists(hist['sig_' + responseName], hist['bkg_' + responseName])
        else:
            fn.drawHists(hist['bkg_' + responseName], hist['sig_' + responseName])
        leg1.Draw()

        # add correctly formatted text to the plot for the ATLAS collab text, energy, etc.
        fn.addLatex(fn.getAlgorithmString(),fn.getAlgorithmSettings(),fn.getPtRange(), fn.getE(), [fn.getNvtxLow(), fn.getNvtx()])
        # save individual plots
        if applyMassWindow:
            canv1.SaveAs(varpath+responseName+".png")
        else:
            canv1.SaveAs(varpath+responseName+"_noMW.png")

        del canv1
    return hist
