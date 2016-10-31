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
STUB = 0
MINX = 2
MAXX = 3
BINS = 4
FN = 5
# store all of the rejection results
totalrejection = []

# these are used for the maximum rejection results
max_rej = 0
maxrejvar = ''
maxrejm_min = 0
maxrejm_max = 0
weightedxAOD = False
singleSidedROC = ''

# more settings
writeResponse = False
if writeResponse:
    import responsePlots as resp


colours = [1,2,3,4,5,6,7,8,9,11,12,20,26,28,30,32,34,38,41,43,46,49]

def getMassWindow(massfile):
    '''
    Get the mass window limits from the input file. Input file will have the lines 'top edge: MAX' and 'bottom edge: MIN' in it somewhere.
    Keyword args:
    massfile -- input file
    
    returns:
    max_mass, min_mass
    '''
    f = open(massfile)
    #print massfile
    m_max = 0.0
    m_min = 0.0
    for l in f:
        # find the correct lines
        if l.startswith('top edge'):
            # convert the last element after splitting to a float.
            m_max = float(l.strip().split()[-1])
        elif l.startswith('bottom edge'):
            # convert the last element after splitting to a float.
            m_min = float(l.strip().split()[-1])
    f.close()
    return m_max, m_min

def writePlots(Algorithm, fileid, canv1, canv2, writeROC, roc = {}, power_canvas = None):
    '''
    Write plots of variables and ROCs to file - png/ pdf
    Keyword args:
    Algorithm -- name of algorithm being used.
    fileid -- file identifier from config file
    canv1 -- TCanvas of variables to be saved
    canv2 -- TCanvas of ROC curve
    writeROC -- Flag indicating if ROC plots should be drawn in addition to variable plots
    '''
    # plot the variables
    canv1.SaveAs('plots/' + Algorithm + fileid + '-Tim2-VariablePlot.pdf')
    #canv1.SaveAs('plots/' + Algorithm + fileid + '-Tim2-VariablePlot.eps')
    # using linux tool 'convert' gives much higher quality .pngs than if you do canv.SaveAs(xyz.png)
    cmd = 'convert -verbose -density 150 -trim plots/' + Algorithm + fileid + '-Tim2-VariablePlot.pdf -quality 100 -sharpen 0x1.0 plots/'+ Algorithm + fileid +'-Tim2-VariablePlot.png'
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    
    if not writeROC:
        return
    #plot the rocs
    canv2.SaveAs('plots/' + Algorithm + fileid + '-Tim2-ROCPlot.pdf')
    #canv2.SaveAs('plots/' + Algorithm + fileid + '-Tim2-ROCPlot.eps')
    cmd = 'convert -verbose -density 150 -trim plots/' +  Algorithm + fileid + '-Tim2-ROCPlot.pdf -quality 100 -sharpen 0x1.0 plots/' +  Algorithm + fileid +'-Tim2-ROCPlot.png'
    p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

    if power_canvas is not None:
        #plot the rocs
        power_canvas.SaveAs('plots/' + Algorithm + fileid + '-Tim2-ROCPowPlot.pdf')
        #canv2.SaveAs('plots/' + Algorithm + fileid + '-Tim2-ROCPlot.eps')
        cmd = 'convert -verbose -density 150 -trim plots/' +  Algorithm + fileid + '-Tim2-ROCPowPlot.pdf -quality 100 -sharpen 0x1.0 plots/' +  Algorithm + fileid +'-Tim2-ROCPowPlot.png'
        p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()


def writePlotsToROOT(Algorithm, fileid, hist, rocs={}, rocs_rejpow={}, rocs_nomw={}, rocs_rejection_scores={},recreate=True, power_curves={}):
    '''
    Write plots to a ROOT file instead of png/pdf
    Keyword args:
    Algorithm -- Algorithm name
    fileid -- Identifier for output file
    hist -- dictionary of histograms
    rocs -- dictionary of ROCs
    rocs_rejpow -- dictionary of ROCs rejection power
    '''
    if recreate:
        fo = TFile.Open('plots/'+Algorithm+fileid+'.root','RECREATE')
    else:
        fo = TFile.Open('plots/'+Algorithm+fileid+'.root','UPDATE')
    for h in hist.keys():
        if hist[h].Integral() != 0:
            hist[h].Write()
    for r in rocs.keys():
        rocs[r].Write('roc_'+r)
    for r in rocs_nomw.keys():
        rocs_nomw[r].Write('roc_nomw_'+r)
    for r in rocs_rejpow.keys():
        rocs_rejpow[r].Write('bkgrej_roc_'+r)
    for r in power_curves.keys():
        power_curves[r].Write('bkgpower_roc_'+r)

    for s in rocs_rejection_scores.keys():
        info = s+'_'+str(rocs_rejection_scores[s])
        n = TNamed(info,info)
        n.Write()
        
    fo.Close()



    
def analyse(Algorithm, plotbranches, plotreverselookup, plotconfig, trees, cutstring, hist, leg1, leg2, fileid, records, ptreweight = True, varpath = "", savePlots = True, mass_min = "0.0", mass_max = "1200000.0", scaleLumi = 1, nTracks='999'):
    '''
    Run through the Algorithm for a given mass range.  Returns the bkg rej at 50% signal eff.
    Keyword args:
    Algorithm -- Name of the algorithm.  Set in main, comes from config file.
    plotbranches -- the variables to be plotted.  Set in main.
    plotreverselookup -- Lookup for an algorithm from a variable stub
    plotconfig --- Dictionary of the plotting arguments for different variables.
    trees -- contains the actual data.
    cutstring -- basic selection cuts to be applied.
    hist -- Histograms that will be filled/ saved.
    leg1 -- TLegend for histograms
    leg2 -- TLegend for ROC curves
    fileid -- File identifier that gets used in the output file names
    ptreweight -- Reweight signal according to pT
    varpath -- output folder to save single variable plots to
    saveplots -- whether or not to write out plots to file
    mass_min -- Mass window minimum
    mass_max -- Mass window maximum
    scaleLumi -- luminosity scale factor
    nTracks -- cut on NTracks to apply.  Default is 999 which is essentially no cut.

    Returns:
    Background rejection at 50% signal efficiency using the ROC curve and variable used to achieve maximum rejection.
    '''

    # open log file
    #logfile = open(records,'w')
    cutflow = open(records.replace('.out','.cutflow'),'w')

    # canvas for histogram plots
    canv1 = TCanvas("canv1")
    canv1.Divide(5,5)
    # canvas for rejection ROC curves
    canv2 = TCanvas("canv2", "ROC curves showing 1-background rejection vs signal efficiency")
    # canvas for power roc curves
    power_canv = TCanvas('powercurves',"ROC curves showing background rejection power vs signal efficiency")
    power_canv.SetLogy()
    power_legend = TLegend(0.6,0.5,0.9,0.9); power_legend.SetFillColor(kWhite);

    tempCanv = TCanvas("temp")
    # reset hists
    for h in hist.keys():
        hist[h].Reset()
    
    global weightedxAOD
    global singleSidedROC

    global totalrejection
    # dict containing all of the ROC curves
    roc={}
    roc_nomw = {}
    bkgRejROC= {}
    bkgPowerROC = {}
    # bool that is set to false if no ROC curves are drawn - this will happen if any 
    # hist added to the roc is empty
    writeROC = False

    # dictionary containing the 50% signal eff bkg rejection power
    roc_rejection_scores = {}

    # dictionary holding all of the histograms without a mass cut
    hist_nomw = {}
    saveNoMassWindowPlots = savePlots
    hist_mass_nomw = {}

    # record the integral of pt for mw and nomw
    mw_int_pt = {}
    full_int_pt = {}

    # maximum rejection
    maxrej = 0
    maxrejvar = ''
    #set up the cutstring/ selection to cut on the correct jet masses
    cutstring_mass = cutstring+ " * (jet_" +Algorithm + "_m <= " +mass_max+ ")" + " * (jet_" +Algorithm + "_m > " +mass_min+ ") " 
    # loop through the indices and branchnames
    for index, branchname in enumerate(plotbranches):
        # add ROC dictionary entry
        roc[branchname] = TGraph()#Errors()
        roc[branchname].SetTitle(branchname)
        roc[branchname].SetName('roc_'+branchname)
        roc_nomw[branchname] = TGraph()#Errors()
        roc_nomw[branchname].SetTitle('No mass window ' + branchname)
        roc_nomw[branchname].SetName('roc_nomw_'+branchname)
        # add bkg rej power dictionary entry
        bkgRejROC[branchname] = TGraph()#Errors()
        bkgRejROC[branchname].SetTitle(branchname)
        bkgRejROC[branchname].SetName('bkgrej_roc_'+branchname)
        bkgPowerROC[branchname] = TGraph()#Errors()
        bkgPowerROC[branchname].SetTitle(branchname)
        bkgPowerROC[branchname].SetName('bkgpower_roc_'+branchname)
        # new canvas
        canv1.cd(index+1)

        # keep the integral when not applying mass window cuts, this
        # allows us to calculate the efficiency of the mass window cut
        signal_eff = 1.0
        bkg_eff = 1.0

        # setting the maximum of the plot - normally multiply by 1.2, but for some, like tauwta21 must be higher
        max_multiplier = 1.2
        if branchname.lower().find('tauwta2tauwta1') != -1:
            max_multiplier = 1.25


        # loop through the datatypes: signal and background
        for indexin, datatype in enumerate(trees):
            canv1.cd(index+1)
            histname =  datatype + "_" + branchname

            print "plotting " + datatype + branchname
            #logfile.write("plotting " + datatype + branchname+"\n")


            minxaxis = hist[histname].GetXaxis().GetXmin()
            maxxaxis = hist[histname].GetXaxis().GetXmax()
            # add the mc_weight and weighted number of events to the selection string
            # also make sure that the variable being plotted is within the bounds specified 
            # in the config file (the limits on the histogram)
            if not weightedxAOD:
                cutstringandweight = '*mc_event_weight*1./NEvents(mc_channel_number)'
            else:
                cutstringandweight = '*mc_event_weight*1./evt_nEvts'

            # add the cross section and filter efficiency for the background
            
            if datatype == 'bkg': 
                # no longer apply the k factor
                if not weightedxAOD:
                    cutstringandweight += '*filter_eff*xs'#*k_factor'
                else:
                    cutstringandweight += '*evt_filtereff*evt_xsec'#*evt_kfactor'
                hist[histname].SetMarkerStyle(21)
            # filter efficiency for signal
            elif datatype == 'sig':
                # we only apply pt rw now, so reset the cutstring
                cutstringandweight=''
                # apply pt reweighting to the signal
                if ptreweight:
                    cutstringandweight +='*SignalPtWeight3(jet_CamKt12Truth_pt)'
            
            hist[histname].Sumw2();
            # set up the tree.Draw() variable expression for the histogram
            varexp = branchname + '>>' + histname
            # apply the selection to the tree and store the output in the histogram
            if branchname.find('_m')==-1:
                trees[datatype].Draw(varexp,cutstring_mass+cutstringandweight)#+'*(nTracks<'+nTracks+')')
            else:
                trees[datatype].Draw(varexp,cutstring+cutstringandweight)#+'*(nTracks<'+nTracks+')')

            # if the histogram is not empty then normalise it
            
            #mw_int = hist[histname].Integral()

            if hist[histname].Integral() > 0.0:
                if scaleLumi != 1:
                    hist[histname].Scale(scaleLumi);
                else:
                    hist[histname].Scale(1.0/hist[histname].Integral());


            # set up the axes titles and colours/ styles
            hist[histname].SetLineStyle(1); hist[histname].SetFillStyle(0); hist[histname].SetMarkerSize(1);
            if (branchname.find('jet_')!=-1):
                hist[histname].SetXTitle(plotreverselookup[branchname.replace("jet_"+Algorithm,"")])
            else:
                hist[histname].SetXTitle(plotreverselookup[branchname])
            hist[histname].SetYTitle("Normalised Entries")

            #now get the same plot for no mass window cut to get the eff
            hist_full_massonly = hist[histname].Clone()
            hist_full_massonly.Reset()
            hist_full_massonly.SetName(histname+'_massonly')
            # need to store the variable in this histogram
            varexpfull = branchname + ' >>' + histname+'_massonly'
            # no nTracks cut here!!!!
            if branchname.find('_m')==-1:
                trees[datatype].Draw(varexpfull,cutstring_mass+cutstringandweight)
            else:
                trees[datatype].Draw(varexpfull,cutstring+cutstringandweight)
            # get the integral and normalise
            mw_int = hist_full_massonly.Integral()
            
            hist_full_massonly.Reset()
            hist_full_massonly.SetName(histname+'_full_massonly')
            # need to store the variable in this histogram
            varexpfull = branchname + ' >>' + histname+'_full_massonly'
            # no nTracks cut here!!!!
            trees[datatype].Draw(varexpfull,cutstring+cutstringandweight+"*(jet_" +Algorithm + "_m < 1200*1000)" + " * (jet_" +Algorithm + "_m > 0)")
            # get the integral and normalise
            full_int = hist_full_massonly.Integral()

            
            if histname.find('_pt') != -1:
                mw_int_pt[datatype] = mw_int
                full_int_pt[datatype] = full_int


            #now get the same plot for no mass window cut to get the eff
            hist_full = hist[histname].Clone()
            hist_full.Reset()
            hist_full.SetName(histname+'_full')
            # need to store the variable in this histogram
            varexpfull = branchname + ' >>' + histname+'_full'

            trees[datatype].Draw(varexpfull,cutstring+cutstringandweight+"*(jet_" +Algorithm + "_m < 1200*1000)" + " * (jet_" +Algorithm + "_m > 0)")#+'*(nTracks<'+nTracks+')')
            histInt = hist_full.Integral()
            # now scale
            if histInt != 0.0:
                if scaleLumi!=1:
                    hist_full.Scale(scaleLumi)
                else:
                    hist_full.Scale(1./histInt)
            # change the x title
            hist_full.SetXTitle(hist[histname].GetXaxis().GetTitle()+' (no mass window)')

            #save this histogram to the no mass window histo dictionary
            hist_nomw[histname+'_full'] = hist_full.Clone()

            if False:#datatype == 'sig':
                if full_int !=0:
                    signal_eff = float(mw_int/full_int)
                else:
                    signal_eff = 0.0
            if False:#else:
                if full_int != 0:
                    bkg_eff = float(mw_int/full_int)
                else:
                    bkg_eff = 0.0

        #Make ROC Curves before rebinning, but only if neither of the samples are zero
        if (hist["sig_" +branchname].Integral() != 0 and hist["bkg_" +branchname].Integral() != 0):

            if singleSidedROC == 'M':
                # check where the median is so we can correctly choose L or R cut.
                sig_median = fn.median(hist['sig_'+branchname])
                bkg_median = fn.median(hist['bkg_'+branchname])

                if sig_median > bkg_median:
                    side = 'R'
                else:
                    side = 'L'

                roc[branchname] = fn.RocCurve_SingleSided(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside=side)

                #roc[branchname],hsigreg50,hcutval50,hsigreg25,hcutval25 = fn.RocCurve_SingleSided_WithUncer(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside=side)
                bkgRejROC[branchname] = roc[branchname]
                bkgPowerROC[branchname] = fn.RocCurve_SingleSided(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside=side,rejection=False)
                

            elif singleSidedROC == 'L' or singleSidedROC == 'R':
                roc[branchname] = fn.RocCurve_SingleSided(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside=singleSidedROC)
                #roc[branchname],hsigreg50,hcutval50,hsigreg25,hcutval25 = fn.RocCurve_SingleSided_WithUncer(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside=singleSidedROC)
                bkgRejROC[branchname] = roc[branchname]
                bkgPowerROC[branchname] = fn.RocCurve_SingleSided(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside=singleSidedROC,rejection=False)

            else:
                MakeROCBen(1, hist["sig_" +branchname], hist["bkg_" +branchname], roc[branchname], bkgRejROC[branchname], signal_eff, bkg_eff)
                bkgPowerROC[branchname] = fn.RocCurve_SingleSided(hist["sig_" +branchname], hist["bkg_" +branchname], signal_eff, bkg_eff, cutside='R',rejection=False)
            writeROC = True
            
        canv1.cd(index+1)
        pX = Double(0.5)
        pY = Double(0.0)

        # find the corresponding bkg rejection for the 50% signal efficiency point from bkg rejection power ROC curve
        # However, if we want the background rejection power for the mass variable we do not want to take 50% as we already have made
        # a cut on the mass to get it to 68%.
        # calculate error on this as well -> deltaX = X*(deltaY/Y)
        eval_roc = 1.0
        err_up = 1
        err_do = 1
        if not branchname.endswith("_m"):
            #eval_roc = roc[branchname].Eval(0.5)
            eval_roc = fn.GetBGRej50(roc[branchname])
            # get the bin for this point so that we can find the associated error in roc error tgraphs
            #rocBin = fn.findYValue(roc[branchname],Double(0.5), pY, 0.01, True, True)
            bin_num = roc[branchname].GetXaxis().FindBin(0.5)#roc_errUp[branchname].Eval(0.5)
            eval_rocup = eval_roc+roc[branchname].GetErrorX(bin_num)
            eval_rocdo = eval_roc-roc[branchname].GetErrorX(bin_num)
        else:
            #eval_roc = roc[branchname].Eval(0.68)
            eval_roc = fn.GetBGRej(roc[branchname], 0.68)
            #rocBin = fn.findYValue(roc[branchname],Double(0.68), pY, 0.01, True, True)
            bin_num = roc[branchname].GetXaxis().FindBin(0.68)#roc_errUp[branchname].Eval(0.5)
            eval_rocup = eval_roc+roc[branchname].GetErrorX(bin_num)
            eval_rocdo = eval_roc-roc[branchname].GetErrorX(bin_num)

        if eval_roc != 1:
            bkgrej = 1/(1-eval_roc)
        else:
            bkgrej = -1
        
        roc_rejection_scores[branchname] = bkgrej

        if (eval_rocup != 1):
            bkgrej_errUp = abs(bkgrej-1/(1-eval_rocup))
        else:
            bkgrej_errUp = -1
        if (eval_rocdo!= 1):
            bkgrej_errDo = abs(bkgrej-1/(1-eval_rocdo))
        else:
            bkgrej_errDo = -1

        # store a record of all background rejection values
        # want to store only the variable name, not the algorithm name, so string manipulation.  here it is stored as sig_jet_ALGO_variable.
        groups = branchname.split('_')
        j = '_'.join(groups[:2]), '_'.join(groups[2:])

        if not j[1] == 'pt':
            totalrejection.append([j[1], float(bkgrej), float(bkgrej_errUp), float(bkgrej_errDo)])

        if bkgrej > maxrej:
            maxrej = bkgrej
            maxrejvar = branchname

        # once the background rejection power has been calculated using the 200 bins the histograms can be rebinned.
            
        hist['sig_'+branchname].SetFillColor(4); hist['sig_'+branchname].SetLineColor(4); hist['sig_'+branchname].SetMarkerColor(4); hist['sig_'+branchname].Rebin(4);
        hist['bkg_'+branchname].SetFillColor(2); hist['bkg_'+branchname].SetLineColor(2);  hist['bkg_'+branchname].SetMarkerColor(2);  hist['bkg_'+branchname].Rebin(4);
        if saveNoMassWindowPlots:
            if singleSidedROC == 'M':
                # check where the median is so we can correctly choose L or R cut.
                sig_median = fn.median(hist_nomw['sig_'+branchname+'_full'])
                bkg_median = fn.median(hist_nomw['bkg_'+branchname+'_full'])

                if sig_median > bkg_median:
                    side = 'R'
                else:
                    side = 'L'
                    
                #roc_nomw[branchname],v1,v2,v3,v4 = fn.RocCurve_SingleSided_WithUncer(hist_nomw["sig_" +branchname+'_full'], hist_nomw["bkg_" +branchname+'_full'], 1,1, cutside=side)
                roc_nomw[branchname] = fn.RocCurve_SingleSided(hist_nomw["sig_" +branchname+'_full'], hist_nomw["bkg_" +branchname+'_full'], 1,1, cutside=side, rejection=True, debug_flag=False)

            elif singleSidedROC == 'L' or singleSidedROC == 'R':
                #roc_nomw[branchname],v1,v2,v3,v4 = fn.RocCurve_SingleSided_WithUncer(hist_nomw["sig_" +branchname+'_full'], hist_nomw["bkg_" +branchname+'_full'], 1,1, cutside=singleSidedROC)
                roc_nomw[branchname] = fn.RocCurve_SingleSided(hist_nomw["sig_" +branchname+'_full'], hist_nomw["bkg_" +branchname+'_full'], 1,1, cutside=singleSidedROC, rejection=True, debug_flag=False)

            else:
                MakeROCBen(1, hist_nomw["sig_" +branchname+'_full'], hist_nomw["bkg_" +branchname+'_full'], roc_nomw[branchname], TGraphErrors(), signal_eff, bkg_eff)
            canv1.cd(index+1)
            #roc_nomw[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
            #roc_nomw[branchname].GetYaxis().SetTitle("1 - Efficiency_{QCD jets}")
            hist_nomw['sig_'+branchname+'_full'].SetFillColor(4); hist_nomw['sig_'+branchname+'_full'].SetLineColor(4); hist_nomw['sig_'+branchname+'_full'].SetMarkerColor(4); 
            hist_nomw['bkg_'+branchname+'_full'].SetFillColor(2); hist_nomw['bkg_'+branchname+'_full'].SetLineColor(2);  hist_nomw['bkg_'+branchname+'_full'].SetMarkerColor(2);
            # resize the mass plots to be in a better range
            if branchname.endswith('_m'):
                hist['sig_'+branchname].SetAxisRange(0.0,300.0*1000.0)
                hist['bkg_'+branchname].SetAxisRange(0.0,300.0*1000.0)
                hist['sig_'+branchname].GetXaxis().SetLimits(0.0,300.0)
                hist['bkg_'+branchname].GetXaxis().SetLimits(0.0,300.0)
            else:
                hist_nomw['sig_'+branchname + '_full'].Rebin(4);
                hist_nomw['bkg_'+branchname + '_full'].Rebin(4);

        leg1.Clear()
        # add legend entries for bkg and signal histograms
        leg1.AddEntry(hist["sig_" + branchname],"W jets","l");    leg1.AddEntry(hist["bkg_" + branchname],"QCD jets","l");

        y_high = max(hist['sig_'+branchname].GetMaximum(), hist['bkg_'+branchname].GetMaximum())
        hist['bkg_'+branchname].SetMaximum(y_high*max_multiplier);hist['sig_'+branchname].SetMaximum(y_high*max_multiplier)
        # plot the maximum histogram
        #if (hist['sig_'+branchname].GetMaximum() > hist['bkg_'+branchname].GetMaximum()):
        fn.drawHists(hist['sig_' + branchname], hist['bkg_' + branchname])
        # else:
        #    fn.drawHists(hist['bkg_' + branchname], hist['sig_' + branchname])
        # change the coordinates to LHS of the canvas if we are plotting ThrustMaj or YFilt.
        offset_x = False
        if branchname.lower().find('thrustmaj')!=-1 or branchname.lower().find('yfilt')!=-1:
            offset_x = True
            leg1.SetX1NDC(0.25)
            leg1.SetX2NDC(0.35)
        else:
            leg1.SetX1NDC(0.8)
            leg1.SetX2NDC(0.9)
        leg1.Draw("same")

        # add correctly formatted text to the plot for the ATLAS collab text, energy, etc.
        fn.addLatex(fn.getAlgorithmString(),fn.getAlgorithmSettings(),fn.getPtRange(), fn.getE(), [fn.getNvtxLow(), fn.getNvtx()], offset_x)
        # save individual plots
        if savePlots:
            p = canv1.cd(index+1).Clone() 
            tempCanv.cd()
            p.SetPad(0,0,1,1) # resize
            p.Draw()
            tempCanv.SaveAs(varpath+branchname+".pdf")
            #tempCanv.SaveAs(varpath+branchname+".eps")
            del p

        # now save "no mass window" plots
        if saveNoMassWindowPlots:
            # resize the mass plots to be in a better range
            if branchname.find('_m') != -1:
                hist_nomw['sig_'+branchname+'_full'].SetAxisRange(0.0,300.0*1000.0)
                hist_nomw['sig_'+branchname+'_full'].GetXaxis().SetRangeUser(0.0,1000.0*300.0)
                hist_nomw['sig_'+branchname+'_full'].GetXaxis().SetLimits(0.0,300.0)
                hist_nomw['bkg_'+branchname+'_full'].SetAxisRange(0.0,300.0*1000.0)
                hist_nomw['bkg_'+branchname+'_full'].GetXaxis().SetRangeUser(0.0,1000.0*300.0)
                hist_nomw['bkg_'+branchname+'_full'].GetXaxis().SetLimits(0.0,300.0)
            tempCanv2 = TCanvas("tempnomw"+branchname)
            tempCanv2.cd()
            y_high = max(hist_nomw['sig_'+branchname+'_full'].GetMaximum(), hist_nomw['bkg_'+branchname+'_full'].GetMaximum())
            y_low = 0.0 #min(hist_nomw['sig_'+branchname+'_full'].GetMinimum(), hist_nomw['bkg_'+branchname+'_full'].GetMaximum())
            hist_nomw['bkg_'+branchname+'_full'].SetMaximum(y_high*max_multiplier);hist_nomw['sig_'+branchname+'_full'].SetMaximum(y_high*max_multiplier)
            #if (hist_nomw['sig_'+branchname+'_full'].GetMaximum() > hist_nomw['bkg_'+branchname+'_full'].GetMaximum()):
            fn.drawHists(hist_nomw['sig_' + branchname+'_full'], hist_nomw['bkg_' + branchname+'_full'])
            #else:
            #    fn.drawHists(hist_nomw['bkg_' + branchname+'_full'], hist_nomw['sig_' + branchname+'_full'])

            offset_x = False
            if branchname.lower().find('thrustmaj')!=-1 or branchname.lower().find('yfilt')!=-1:
                offset_x = True
                leg1.SetX1NDC(0.25)
                leg1.SetX2NDC(0.35)
            else:
                leg1.SetX1NDC(0.8)
                leg1.SetX2NDC(0.9)
            leg1.Draw("same")

            # if we are doing the mass plt we want to indicate the mass window
            if branchname.find('_m') == -1:
                mrange = []
            else:
                mrange = [float(mass_min), float(mass_max)]
                # if it is the mass variable then draw lines indicating the mass window
                line_min = TLine(float(mass_min)/1000.0, y_low, float(mass_min)/1000.0, y_high); line_min.SetLineColor(kBlack);
                line_max = TLine(float(mass_max)/1000.0, y_low, float(mass_max)/1000.0, y_high); line_max.SetLineColor(kBlack);
                line_min.Draw('same')
                line_max.Draw('same')

            # add the latex parts to the plot -> ATLAS, tev range, etc.
            fn.addLatex(fn.getAlgorithmString(),fn.getAlgorithmSettings(),fn.getPtRange(), fn.getE(), [fn.getNvtxLow(), fn.getNvtx()], offset_x, massrange = mrange)

            tempCanv2.SaveAs(varpath+branchname+"_noMW.pdf")
            #tempCanv2.SaveAs(varpath+branchname+"_noMW.eps")
            del tempCanv2

        
        canv1.cd(index+1)

    min_roc = 1.0
    for b in roc.keys():
        if b.find('_m') == -1 and b.find('_pt') == -1:
            n = roc[b].GetN()
            y = roc[b].GetY()
            locmin = TMath.LocMin(n,y)
            minval = y[locmin]
            min_roc = min(min_roc, minval)
            
    for branchidx, branchname in enumerate(roc.keys()):
        # plot the ROC curves
        canv2.cd()
        roc[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
        roc[branchname].GetYaxis().SetTitle("1 - Efficiency_{QCD jets}")
        roc[branchname].SetMinimum(min_roc*0.98)
        bkgPowerROC[branchname].GetXaxis().SetTitle("Efficiency_{W jets}")
        bkgPowerROC[branchname].GetYaxis().SetTitle("1/Efficiency_{QCD jets}")
    

        # only plot the roc and power curves if this is not a mass
        if branchname.find('_m') == -1 and branchname.find('_pt') == -1:
            roc[branchname].SetLineStyle(branchidx%10)
            if branchidx==0 and roc[branchname].Integral() != 0:
                roc[branchname].Draw("al")        
            elif roc[branchname].Integral() != 0:
                roc[branchname].SetLineColor(colours[branchidx])
                roc[branchname].Draw("same")
                # legend for the roc curve
            leg2.AddEntry(roc[branchname],branchname,"l");
            leg2.Draw("same")

            # plot the power curves
            power_canv.cd()
            bkgPowerROC[branchname].SetLineStyle(branchidx%10)
            if branchidx==0 and bkgPowerROC[branchname].Integral() != 0:
                bkgPowerROC[branchname].Draw("al")        
            elif bkgPowerROC[branchname].Integral() != 0:
                bkgPowerROC[branchname].SetLineColor(colours[branchidx])
                bkgPowerROC[branchname].Draw("same")
            power_legend.AddEntry(bkgPowerROC[branchname],branchname,'l')
            power_legend.Draw('same')

    # write out canv1 and roc curves on one page/ png each
    if savePlots:
        #write out the plots after cuts
        writePlots(Algorithm, fileid, canv1, canv2, writeROC, roc, power_canv)
        writePlotsToROOT(Algorithm, fileid, hist, roc, bkgRejROC, roc_nomw, roc_rejection_scores, recreate=True, power_curves = bkgPowerROC)
        #responseHists = resp.writeResponsePlots(weightedxAOD, Algorithm, plotconfig, trees, cutstring, fileid, ptreweight, varpath, mass_min, mass_max, scaleLumi)
        #responseHists = resp.writeResponsePlots(weightedxAOD, Algorithm, plotconfig, trees, cutstring, fileid, ptreweight, varpath, mass_min, mass_max, scaleLumi, applyMassWindow=False)
        #writePlotsToROOT(Algorithm, fileid, hist, recreate=False)

    # write out event counts for mass window cuts (not weighted)
    cutflow.write('Signal:\njet selection no mass window: '+'pt' +' '+ str(hist_nomw['sig_jet_'+Algorithm+'_pt_full'].GetEntries())+'\n')
    if 'sig' in full_int_pt.keys():
        cutflow.write('jet selection no mass window: '+'pt weighted '+ str(full_int_pt['sig'])+'\n')
    cutflow.write('mass window and jet selection: pt ' + str(hist['sig_jet_'+Algorithm+'_pt'].GetEntries())+'\n')
    if 'sig' in mw_int_pt.keys():
        cutflow.write('mass window and jet selection: pt weighted ' + str(mw_int_pt['sig'])+'\n')
    cutflow.write('Background:\njet selection no mass window: pt '+ str(hist_nomw['bkg_jet_'+Algorithm+'_pt_full'].GetEntries())+'\n')
    if 'bkg' in full_int_pt.keys():
        cutflow.write('jet selection no mass window: pt weighted '+ str(full_int_pt['bkg'])+'\n')
    cutflow.write('mass window and jet selection: pt ' + str(hist['bkg_jet_'+Algorithm+'_pt'].GetEntries())+'\n')
    if 'bkg' in mw_int_pt.keys():
        cutflow.write('mass window and jet selection: pt weighted ' + str(mw_int_pt['bkg'])+'\n')
    # close logfile
    cutflow.close()

    # return the variable with the maximum background rejection
    return maxrej, maxrejvar



def main(args):
    '''
    Main method which takes in all of the parameters for the tagger and sets up the
    configuration.  All of the histograms are set up here, ready to be filled.
    Main method launches "analyse" which runs over a tagger configuration.
    '''
    # read in and parse all of the command line arguments
    parser = argparse.ArgumentParser(description='Plot some variables.')
    parser.add_argument('config', help = 'required config file')
    parser.add_argument('-i','--inputfile',help = 'the input file')
    parser.add_argument('-a', '--algorithm',help='The algorithm: filter100, filter67 or trimmed')
    parser.add_argument('-f','--fileid', help = 'An optional fileid to append to the name of the output file')
    parser.add_argument('--pthigh', help = 'Optional high pT cut in GeV')
    parser.add_argument('--ptlow', help = 'Optional low pT cut in GeV')
    parser.add_argument('--nvtx', help = 'Upper cut on number of primary vertices')
    parser.add_argument('--nvtxlow', help = 'Lower cut on number of primary vertices')
    parser.add_argument('--ptreweighting', help = 'Apply pT reweighting')
    parser.add_argument('--saveplots', help = 'Apply pT reweighting')
    parser.add_argument('--tree', help = 'Name of tree in input file')
    parser.add_argument('--channelnumber', help = 'RunNumber/ mc_channel_number to use for selection')
    parser.add_argument('--lumi', help = 'Luminosity scale factor')
    parser.add_argument('--massWindowCut', help = 'Whether a mass window cut should be applied')
    parser.add_argument('-v','--version',help = 'Version number')
    parser.add_argument('--weightedxAOD', help = 'If the xAOD has been weighted already.')
    parser.add_argument('--ROCside', help = 'L or R for left or right sided ROC cut, leave blank for sorted version.')
    parser.add_argument('--massWindowOverwrite', help = 'Overwrite the current mass window file if it exists.')
    parser.add_argument('--writecsv', help = 'Write the data into a csv file, do not run analyse.')
    parser.add_argument('--nTracks', help = 'The nTrack cut to apply.  Default is no cut.')
    parser.add_argument('--clean', help = 'When creating the csv file cuts will be applied on all variables. These are set in functions.py.')

    args = parser.parse_args()

    config_f = ''
    # if no config file is specified the program exits
    if not args.config:
        print 'Need more args! usage: python Tagger.py config [-i inputfile] [-a algorithm] [-f fileid] [--pthigh=x] [--ptlow=y] [--nvtx=n] [--nvtxlow=l] [--ptreweighting=true/false] [--saveplots=true/false] [--tree=name] [--channelnumber=number] [--lumi=scalefactor] [--massWindowCut=true/false] [-v,--version=versionNumber] [--weightexAOD=true/false] [--ROCside=L,R(default=sorted)] [--massWindowOverwrite=true/false(default=False)]'
        sys.exit(0)
    else:
        config_f = args.config
    # get the input file
    if not args.inputfile:
        args.inputfile = ''

    # load ROOT macros for pt reweighting and event weighting - would it make sense
    # to do this in the functions file and then call everything with fn.?
    SetAtlasStyle()
    ROOT.gROOT.LoadMacro("MakeROCBen.C")
    ROOT.gROOT.LoadMacro("SignalPtWeight3.C")
    ROOT.gROOT.LoadMacro("NEvents.C")

    # declare the dictionaries for trees, input files, weights and run numbers
    trees,files,pthists = ( {} for i in range(3) ) 

    # read in config file
    if args.ptlow:
        fn.readXML(config_f,str(args.ptlow))
    else:
        fn.readXML(config_f,"default")
    # set the treename for the input file
    treename = ''
    if not args.tree:
        treename = fn.getTree()
    else:
        treename = args.tree
        fn.tree = treename
    if treename == '':
        print "specify a tree name in command line args or config file"
        sys.exit()

    # string for algorithm
    Algorithm = ''
    # flags for plotting truth variables
    setTruth = False
    plotTruth = False

    # if the algorithm name is not given in the command line args look for it in the config file
    if not args.algorithm:
        Algorithm = fn.getAlgorithm() # from config file
    else:
        alg_lower = args.algorithm.lower()
        if alg_lower.find('truth') != -1:
            plotTruth = True
            setTruth = True

    if Algorithm == '':
        print "No algorithm given in command line or config file!"
        sys.exit(0)

    # save each variable plot for a given groomer into a folder
    varpath = 'plots/'+Algorithm

    # truth algorithm
    AlgorithmTruth = Algorithm.replace('LCTopo','Truth')##'CamKt12Truth'
    if not setTruth:
        plotTruth = fn.getTruth() # get truth flag from config file if not set in command line args
    if plotTruth:
        varpath += 'Truth'

    # file identifier for the output files
    fileid = ''
    if args.fileid:
        fileid = args.fileid # set from command line
    else:
        fileid = fn.getFileID() # set from config file

    # flag indicating if pt reweighting should be done on signal
    ptreweight = True
    if not args.ptreweighting:
        ptreweight = fn.getPtReweightFlag()
    else:
        if args.ptreweighting.lower() == 'false':
            ptreweight = False

    massWindowCut = False
    if args.massWindowCut.lower() == 'true':
        massWindowCut = True
    #ptreweight = False
    # output path for storing individual variable plots - there are a lot of these so it is 
    # useful to be able to store these in a separate folder
    varpath += fileid +'/'
    if not os.path.exists(varpath):
        os.makedirs(varpath)

    # define cuts on the pt
    if args.pthigh and args.ptlow:
        ptrange = [float(args.ptlow)*1000., float(args.pthigh)*1000.]
        fn.pt_high = float(args.pthigh)*1000
        fn.pt_low = float(args.ptlow)*1000
    else:
        ptrange = fn.getPtRange()

    # set cut on the number of primary vertices
    nvtx = 999 # default is no cut, also set to 999 in functions.py as default
    if args.nvtx:
        nvtx = int(args.nvtx)
    else:
        nvtx = int(fn.getNvtx())
    fn.nvtx = nvtx
    nvtxlow = 0
    if args.nvtxlow:
        nvtxlow = int(args.nvtxlow)
    else:
        nvtxlow = int(fn.getNvtxLow())
    fn.nvtxlow = nvtxlow

    # set the saveplots option - whether we want to save individual plots for each var
    saveplots = True
    if args.saveplots:
        if args.saveplots == 'false' or args.saveplots == 'False' or args.saveplots == 'off':
            saveplots = False

    # this is if we are making selection on only one channel number
    if not args.channelnumber:
        channelcut = ''
    else:
        channelcut = ' * (mc_channel_number == '+str(args.channelnumber)+')'

    # set the weightedxAOD flag
    global weightedxAOD
    if not args.weightedxAOD:
        weightedxAOD = False
    elif args.weightedxAOD == 'true' or args.weightedxAOD == 'True':
        weightedxAOD = True
    else:
        weightedxAOD = False
        

    global singleSidedROC
    if args.ROCside:
        if args.ROCside == 'L' or args.ROCside == 'R' or args.ROCside == 'M':
            singleSidedROC = args.ROCside

    # lumi scaling
    lumi = 1.0
    if not args.lumi:
        lumi = fn.getLumi()

    if not args.massWindowOverwrite:
        args.massWindowOverwrite = 'false'

    # write csv flag
    writecsv = False
    clean_data = True
    if args.writecsv and args.writecsv.lower() == 'true':
        writecsv = True
        # only worried about clean_data if writecsv is true
        if args.clean and args.clean.lower() == 'false':
            clean_data = False

    # default selection string
    cutstring = "(jet_CamKt12Truth_pt > "+str(ptrange[0])+") * (jet_CamKt12Truth_pt <= "+str(ptrange[1])+") * (jet_CamKt12Truth_eta > -1.2) * (jet_CamKt12Truth_eta < 1.2) " + channelcut


    nTracks = '999'
    if args.nTracks:
        print 'applying an nTracks cut of ' + str(args.nTracks)
        nTracks = args.nTracks

    # set up the input signal file
    signalFile = fn.getSignalFile()
    # set up background file
    backgroundFile = fn.getBackgroundFile()
    eventsFileSig = ''
    eventsFileBkg = ''
    massWinFile = ''


    # get all of the filenames. Note that signal and background file are not changed
    # if they have been set already.
    signalFile, backgroundFile, eventsFileSig, eventsFileBkg, massWinFile = fn.getFiles(args.inputfile, signalFile, backgroundFile, massWinFile, ptrange)



    for typename in ['sig','bkg']:
        if typename == 'sig':
            filename = signalFile
        else:
            filename = backgroundFile
        # open the files
        files[typename] = TFile(filename)
        # read in the pt histograms
        pthists[typename] = files[typename].Get('pt_reweight'+typename)
        pthists[typename].SetDirectory(0)
        # read in the trees
        trees[typename] = files[typename].Get(treename)

    # load the pt histograms into the reweighting tool
    loadHistograms(signalFile, backgroundFile)
    # do this for the massptFunctions module too
    ptweightsHist = None


    print "***************************************signalFile: " + signalFile
    massPtFunctions.setPtWeightFile(pthists['sig'],pthists['bkg'])
    if (massWindowCut and massWinFile == '') or (args.massWindowOverwrite.lower() == 'true'):
        print 'calculating mass window since no existing file present'
        massPtFunctions.run(signalFile, Algorithm, treename, ptweightsHist, float(ptrange[0])/1000.,float(ptrange[1])/1000.)
        # check that the mass window calculation was a success, if not, quit.
        if massPtFunctions.success_m:
            massWinFile = massPtFunctions.filename_m
        else:
            print 'mass window calculation failed'
            sys.exit()

    # load all of the event weights from the event weighting file
    loadEvents(eventsFileSig)
    loadEvents(eventsFileBkg)
       
    # remove any branches from the config file that are not in the actual input files
    file_branches = fn.getFileBranches(signalFile, fn.getTree())
    fn.pruneBranches(file_branches)

    # configuration for each variable to plot - axis ranges, titles
    plotconfig = fn.getPlotBranches()

    # just the variables like m, pt, etc
    plotbranchstubs = [item[0] for item in plotconfig.values()]

    # set the plotting range for the pt
    plotconfig['pt'][MINX] = ptrange[0]
    plotconfig['pt'][MAXX] = ptrange[1]

    # dictionary containing the plotting ranges, using the stub as the key
    plotranges = {}
    for p in plotconfig.keys():
        plotranges[plotconfig[p][0]] = [plotconfig[p][MINX],plotconfig[p][MAXX]]

    # get extra variables to be used for selection from tree
    # this is a dictionary with [variable]= [stub, jetvariable flag]
    AlgBranchStubs = fn.getBranches() 
    #create reverse lookup as well, since they shouldn't have duplicate entries this should be okay. 
    # this allows looking up the branch name given just the variable pt, m, etc
    plotreverselookup = {v[0]: k for k, v in AlgBranchStubs.items()}
    # keep track of whether or not the plot is a jet variable and needs the algorithm name appended
    plotjetlookup = {v[0]: v[1] for k, v in AlgBranchStubs.items()}

    # default branches to be plotted
    #branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']#, 'avgIntPerXing']
    branches = ['mc_event_weight']#, 'jet_'+AlgorithmTruth+'_pt', 'jet_'+AlgorithmTruth+'_eta']#, 'avgIntPerXing']

    # set up the full branch names for each variable
    if not plotTruth:
        branches.extend(['jet_' + Algorithm + vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == True]) 
        branches.extend([vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == False]) 
    else:
        branches.extend([vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == False]) 
        branches.extend(['jet_' + AlgorithmTruth + vals[0] for branch, vals in AlgBranchStubs.items() if AlgBranchStubs[branch][1] == True]) 


    # add algorithm names to branches
    plotbranches = ['jet_' + Algorithm + branch for branch in plotbranchstubs if plotjetlookup[branch] == True]
    for branch in plotbranchstubs:
        if not plotjetlookup[branch]:
            plotbranches.append(str(branch))
    #plotbranches += [branch for branch in plotbranchstubs if plotjetlookup[branch] == False]
    #plotbranches.append('avgIntPerXing')
    # flag if variable bin widths are being used - right now not being used anymore, will re-implement
    varBinPt = False

    # dictionary to hold all histograms
    hist = {}

    # at this point change the Algorithm to AlgorithmTruth, since we needed the normal
    # Algorithm name up until now to get pathnames correct.
    if plotTruth:
        Algorithm = AlgorithmTruth

    # set up all of the histograms and names - only do this if running analyse, not writecsv
    if not writecsv:
        types = ['sig','bkg']
    else:
        types = []
    for typename in types:
        histnamestub = typename + '_jet_' + Algorithm
        for br in plotconfig.keys():
            if plotconfig[br][1] == True: # if it is a jet variable
                if plotconfig[br][FN]!='':
                    histname = typename+'_'+plotconfig[br][FN]+'(jet_'+Algorithm+plotconfig[br][STUB]+')'
                    
                else:
                    histname = histnamestub+plotconfig[br][STUB]
            else:
                histname = typename+"_"+plotconfig[br][STUB]
            hist_title = br
            hist[histname] = TH1D(histname, hist_title, plotconfig[br][BINS], plotconfig[br][MINX], plotconfig[br][MAXX])
  
    # legends for histograms and roc curves
    leg1 = TLegend(0.8,0.58,0.9,0.68);leg1.SetFillColor(kWhite)
    leg2 = TLegend(0.2,0.2,0.5,0.7);leg2.SetFillColor(kWhite)

    # set up the mass window cuts
    mass_max = 1200*1000.;
    mass_min = 0.0;

    if massWindowCut:
        mass_max, mass_min = getMassWindow(massWinFile)
        print 'calc mass window'

    masses = [[mass_min,mass_max]]
    # make sure out optimisation folder exists
    if not os.path.exists('optimisation'):
        os.makedirs('optimisation')
    
    # store teh maximum background rejection
    global max_rej, maxrejvar, maxrejm_min, maxrejm_max
    max_rej = 0
    maxrejvar = ''
    maxrejm_min = 0
    maxrejm_max = 0

    for m in masses:

        m_min = m[0]
        m_max = m[1]

        # log the output, make sure that the logs folder exists
        if not os.path.exists('logs'):
            os.makedirs('logs')
        records = 'logs/TaggerOpt'+Algorithm+fileid+'_'+str(m_max)+'_'+str(m_min)+'_.out'

        # run the analysis for mass range
        mass_cut = " * (jet_" +Algorithm + "_m <= " +str(m_max)+ ")" + " * (jet_" +Algorithm + "_m > " +str(m_min)+ ") " 
        if not writecsv:
            rej,rejvar = analyse(Algorithm, plotbranches, plotreverselookup, plotconfig, trees, cutstring, hist, leg1, leg2, fileid, records, ptreweight, varpath, saveplots, str(m_min), str(m_max), lumi, nTracks=args.nTracks)

            if rej > max_rej:
                max_rej = rej
                maxrejvar = rejvar
                maxrejm_min = m_min
                maxrejm_max = m_max
        else:
            # the last argument here is whether to create a csv or root file. Both = csvroot
            sigevents, bkgevents = fn.writeCSV(signalFile, backgroundFile, branches, cutstring, treename, Algorithm, fileid+'_nomw', massPtFunctions.getPtWeightsFile(), plotranges,'root', clean=clean_data)
            # now use the number of signal and background events to calculate the efficiency of the
            # the mass window cut
            fn.writeCSV(signalFile, backgroundFile, branches, cutstring+mass_cut, treename, Algorithm, fileid+'_mw', massPtFunctions.getPtWeightsFile(), plotranges,'root', sigevents, bkgevents, clean=clean_data)
        #records.write(str(rej) + ' ' + rejvar + ' ' + str(m_min) + ' ' + str(m_max)+'/n')

    # close all of the tfiles
    files['sig'].Close()
    files['bkg'].Close()

    # we're not worried about rejection if we are just writing to csv
    if writecsv:
        return
    #records.close()
    # dump totalrejection in pickle to be read in by the scanBkgrej module which runs this module
    print totalrejection

    if not args.version:
        version = 'v1'
    else:
        version = args.version
    with open("tot_rej_"+version+".p","wb") as f:
        pickle.dump(totalrejection, f)
    # print out all of the info, which is read from stdout by the scanner script.
    # this is not really needed anymore as it was a hack to get around ROOT's annoying
    # global memory management.  This is fixed now by using pickle files.  However,
    # this hasn't been fully implemented everywhere yet, so it will take a little while to remove it.
    print "MAXREJSTART:" +str(max_rej)+","+maxrejvar+","+str(maxrejm_min)+","+str(maxrejm_max)+ "MAXREJEND"
    output = "MAXREJSTART:" +str(max_rej)+","+maxrejvar+","+str(maxrejm_min)+","+str(maxrejm_max)+ "MAXREJEND"
    # dump the output to a pickle file
    with open("TaggerOutput_"+version+".p","wb") as f:
        pickle.dump(output,f)

if __name__ == '__main__':
    #max_rej, maxrejvar, maxrejm_min, maxrejm_max=main(sys.argv)
    main(sys.argv)
    sys.exit()

def runMain(args):
    '''
    Use a list for the arguments that would be used from command line.
    '''
    sys.argv = args
    main(args)


    
