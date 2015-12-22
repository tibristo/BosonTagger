from ROOT import *
import os
from numpy import *
import math

def RocCurve_SingleSided(sig, bkg, sig_eff, bkg_eff, cutside='L', bdt = True, normalise = False, rejection=True, debug_flag = True):
    '''
    Produce a single sided roc curve.  
    
    Keyword arguments:
    sig and bkg -- are the signal and background histograms
    sig/bkg_eff -- the signal and background efficiencies (for example, if a mass window cut has been applied the efficiency is already < 1)
    cutside -- L or R, depending on the median of the sig and bkg
    rejection -- Whether to calculate the background rejection or power (1-eff or 1/eff).
    '''
    
    n = bkg.GetNbinsX()
    
    print "NBins",n

    # normalise hists
    if normalise:
        if sig.Integral()!=0:
            sig.Scale(1.0/sig.Integral());
        if(bkg.Integral()!=0):
            bkg.Scale(1.0/bkg.Integral());

    totalBerr=Double()
    totalSerr=Double()
    totalB = bkg.Integral(0,n)
    totalS = sig.Integral(0,n)
    sigEff = float(sig_eff)
    bkgEff = float(bkg_eff)

    gr = TGraph(n)
    # this was from 1->n+1, but if we are using gr.SetPoint() it starts from 0, whereas
    # the hist.Integral(x,y) starts from 1. So i is now 0 <= i < n+1, and hist integral is
    # i+1 -> n or n-> i+1 and then gr.SetPoint(i,x,y)
    for i in range(0,n+1):
        myS = 0.
        myB = 0.

        myBerr=Double()
        mySerr=Double()
        #myB = bkg.Integral(1,i)
        #myS = sig.Integral(1,i)
        #print 'myS: ' + str(myS)
        #print 'myB: ' + str(myB)
        #gr.SetPoint(i, myS*sig_eff, (1-myB*bkgEff))
        #gr.SetPointError(i, mySerr*sig_eff, myBerr*bkgEff)
        if cutside=="R":
            #loop from i to end
            myB = bkg.Integral(i+1,n)
            myS = sig.Integral(i+1,n)
            #print i,"  myS=",myS,"  myB=",myB
            #gr.SetPoint(i, 100*myS*sigEff/totalS, 100.0*(myB*bkgEff)/totalB)
            if myB!=0:
                signif = myS/math.sqrt(myB)
                gr.SetPoint(i, sig.GetBinCenter(i), signif*1000)
            elif myS==0:
                gr.SetPoint(i, sig.GetBinCenter(i), 0)
            else:
                gr.SetPoint(i, sig.GetBinCenter(i), 999)
            #gr.SetPoint(i, myS*sigEff/totalS, 100.0*(myB*bkgEff)/totalB)
        elif cutside=="L":
            #loop from 0 to i
            myB = bkg.Integral(1,i+1)
            myS = sig.Integral(1,i+1)
            #print i,"  myS=",myS,"  myB=",myB
            #print 100*myS*sigEff/totalS
            #print 100*myS*sigEff/totalS
            #gr.SetPoint(i, 100*myS*sigEff/totalS, 100.0*(1-myB*bkgEff/totalB))
            #gr.SetPointError(i, mySerr*sigeff, myBerr*bkgeff)
            #gr.SetPoint(i, 100*myS*sigEff/totalS, 100.0*(myB*bkgEff)/totalB)
            # for s/sqrt(b)


            if myB!=0:
                signif = myS/math.sqrt(myB)
                gr.SetPoint(i, sig.GetBinCenter(i), signif*1000)
            elif myS==0:
                gr.SetPoint(i, sig.GetBinCenter(i), 0)
            else:
                gr.SetPoint(i, sig.GetBinCenter(i), 999)
            
    #artificially set the first point to (1,1) to avoid overflow issues
    #gr.SetPoint(0, 1.0, 1.0)
    
            
    #ctest = TCanvas("ctest","ctest",400,400)
    #gr.Draw("AC")
    if not rejection:
        gr.GetYaxis().SetRangeUser(0.0,400)
        gr.GetYaxis().SetTitle('Background Power: 1/eff')
        gr.SetMinimum(8.0)
        gr.SetMaximum(400.0)
    else:
        gr.GetYaxis().SetTitle('Background Rejection: 1-eff')
        gr.SetMinimum(0.0)
        gr.SetMaximum(100.0)
    #gr.GetXaxis().SetLimits(0.0,100.0)
    #gr.GetXaxis().SetRangeUser(0.0,100.0)
    #gr.GetXaxis().SetTitle('Signal Efficiency')
    if bdt:
        gr.GetXaxis().SetLimits(0.3,0.8)
        gr.GetXaxis().SetRangeUser(0.3,0.8)
        gr.GetXaxis().SetTitle('MVA Score')
    else:
        gr.GetXaxis().SetLimits(0.0,0.8)
        gr.GetXaxis().SetRangeUser(0.0,0.8)
        gr.GetXaxis().SetTitle('MVA Score')
        #gr.Draw("AE3")
    #ctest.SaveAs("ctest.png")
    curve = gr
    bkgRejPower = gr
    #print "RETURNING from Single sided ROC calculation"
    return gr.Clone()


sigfile = TFile.Open('histos_HVT_2000.root','READ')
bkgfile = TFile.Open('histos_background.root','READ')

jet1_bdt_sig = sigfile.Get('Signal_jet1_bdt_masswindow')
jet1_bdt_bkg = bkgfile.Get('QCD_jet1_bdt_masswindow')
jet1_dnn_sig = sigfile.Get('Signal_jet1_dnn_masswindow')
jet1_dnn_bkg = bkgfile.Get('QCD_jet1_dnn_masswindow')

jet1_bdt_roc = RocCurve_SingleSided(jet1_bdt_sig, jet1_bdt_bkg, 1.0, 1.0, 'R', True, False, True)
jet1_dnn_roc = RocCurve_SingleSided(jet1_dnn_sig, jet1_dnn_bkg, 1.0, 1.0, 'R', False, False, True)

tc = TCanvas()

jet1_bdt_roc.Draw()
gPad.Print('jet1_bdt_roc.pdf')
jet1_dnn_roc.Draw()
gPad.Print('jet1_dnn_roc.pdf')
