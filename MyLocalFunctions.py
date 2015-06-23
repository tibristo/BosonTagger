import os
import sys
from ROOT import *
from AtlasStyle import *
from numpy import log,arange
SetAtlasStyle();
gStyle.SetPalette(1)
TH1.SetDefaultSumw2()

#######
gROOT.ProcessLineSync(".x "+os.getcwd()+"/../MyPackages/PtReweighting.C+")
gROOT.ProcessLineSync(".x "+os.getcwd()+"/../MyPackages/GetCalibration.C+")
#######

###########################################
# Weights for normalization calculated from plotting macro
# L*(sigma*fe)*(1/N)
###########################################
# GWW = "1.0 * 1.0000000 * 1.00 * (1.000000/00001.0)"
# JX3 = "1.0 * 544.18    * 1.00 * (0.001222/15999.0)"
# JX4 = "1.0 *   6.443   * 1.00 * (0.000708/10000.0)"
# JX5 = "1.0 *   0.0397  * 1.00 * (0.002152/15000.0)"
# JX6 = "1.0 *   0.00041 * 1.00 * (0.004677/15000.0)"

#OLD STATS
# GWW = "1.0"
# JX3 = "(1.220E-3 * 5.4418E2)  / 70897."
# JX4 = "(7.0841E-4 * 6.443)     / 99571."
# JX5 = "(2.1516E-3 * 3.9739E-2) / 99664."
# JX6 = "(4.6774E-3 * 4.2610E-4) / 99749."

#NEW STATS
GWW = "1.0"
JX3 = "(1.220E-3 * 5.4418E2)  / 499555."
JX4 = "(7.0841E-4 * 6.443)     / 483193."
JX5 = "(2.1516E-3 * 3.9739E-2) / 498014."


def TranslateRegion(reg):
    #print "Translating reg: ",reg

    AllRegs={}
    AllRegs["pt200350mNONE"] = "|#eta^{Truth}|<1.2 , p_{T}^{Truth}=[200,350] GeV"
    AllRegs["pt200350mOPT"]  = "|#eta^{Truth}|<1.2 , p_{T}^{Truth}=[200,350] GeV,  M^{Reco} Cut"   
    AllRegs["pt350500mNONE"] = "|#eta^{Truth}|<1.2 , p_{T}^{Truth}=[350,500] GeV"
    AllRegs["pt350500mOPT"]  = "|#eta^{Truth}|<1.2 , p_{T}^{Truth}=[350,500] GeV,  M^{Reco} Cut"
    AllRegs["pt5001000mNONE"] = "|#eta^{Truth}|<1.2 , p_{T}^{Truth}=[500,1000] GeV"
    AllRegs["pt5001000mOPT"]  = "|#eta^{Truth}|<1.2 , p_{T}^{Truth}=[500,1000] GeV,  M^{Reco} Cut"


    if reg not in AllRegs.keys():
        print "This reg is not in your list: ",reg," EXITTING ..."
        sys.exit()
        
    regout=AllRegs[reg]

    return regout    
    

def TranslateAlg(alg):
    #print "Translating alg: ",alg
    
    AllAlgs={}
    AllAlgs["CA12LCTRIMF5R20"]      = "C/A^{R=1.2} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["AK12LCTRIMF5R20"]      = "anti-k_{T}^{R=1.2} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["CA10LCTRIMF5R20"]      = "C/A^{R=1.0} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["AK10LCTRIMF5R20"]      = "anti-k_{T}^{R=1.0} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["AK10LCTRIMF5R30"]      = "anti-k_{T}^{R=1.0} _{Trimmed(f_{cut}=5%,R_{sub}=0.3)}"   
    AllAlgs["CA08LCTRIMF5R20"]      = "C/A^{R=0.8} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["AK08LCTRIMF5R20"]      = "anti-k_{T}^{R=0.8} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["CA06LCTRIMF5R20"]      = "C/A^{R=0.6} _{Trimmed(f_{cut}=5%,R_{sub}=0.2)}"   
    AllAlgs["CA12LCPRUNR50Z10"]     = "C/A^{R=1.2} _{Pruned(R_{cut}=0.5,Z_{cut}=0.1)}"   
    AllAlgs["CA12LCPRUNR50Z15"]     = "C/A^{R=1.2} _{Pruned(R_{cut}=0.5,Z_{cut}=0.15)}"   
    AllAlgs["CA10LCPRUNR50Z10"]     = "C/A^{R=1.0} _{Pruned(R_{cut}=0.5,Z_{cut}=0.1)}"   
    AllAlgs["CA10LCPRUNR50Z15"]     = "C/A^{R=1.0} _{Pruned(R_{cut}=0.5,Z_{cut}=0.15)}"   
    AllAlgs["CA08LCPRUNR50Z10"]     = "C/A^{R=0.8} _{Pruned(R_{cut}=0.5,Z_{cut}=0.1)}"   
    AllAlgs["CA08LCPRUNR50Z15"]     = "C/A^{R=0.8} _{Pruned(R_{cut}=0.5,Z_{cut}=0.15)}"   
    AllAlgs["CA06LCPRUNR50Z10"]     = "C/A^{R=0.6} _{Pruned(R_{cut}=0.5,Z_{cut}=0.1)}"   
    AllAlgs["CA06LCPRUNR50Z15"]     = "C/A^{R=0.6} _{Pruned(R_{cut}=0.5,Z_{cut}=0.15)}"   
    AllAlgs["CA12LCBDRSM100R30Y0"]  = "C/A^{R=1.2} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=0%)}"   
    AllAlgs["CA12LCBDRSM100R30Y4"]  = "C/A^{R=1.2} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=4%)}"   
    AllAlgs["CA12LCBDRSM100R30Y9"]  = "C/A^{R=1.2} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=9%)}"   
    AllAlgs["CA12LCBDRSM100R30Y12"] = "C/A^{R=1.2} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=12%)}"   
    AllAlgs["CA12LCBDRSM100R30Y15"] = "C/A^{R=1.2} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=15%)}"   
    AllAlgs["CA08LCBDRSM100R30Y0"]  = "C/A^{R=0.8} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=0%)}"   
    AllAlgs["CA08LCBDRSM100R30Y4"]  = "C/A^{R=0.8} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=4%)}"   
    AllAlgs["CA08LCBDRSM100R30Y9"]  = "C/A^{R=0.8} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=9%)}"   
    AllAlgs["CA06LCBDRSM100R30Y0"]  = "C/A^{R=0.6} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=0%)}"   
    AllAlgs["CA06LCBDRSM100R30Y4"]  = "C/A^{R=0.6} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=4%)}"   
    AllAlgs["CA06LCBDRSM100R30Y9"]  = "C/A^{R=0.6} _{Split-Filtered(#mu=1,,R_{sub}=0.3,y_{filt}=9%)}"   

    if alg not in AllAlgs.keys():
        print "This alg is not in your list: ",alg," EXITTING ..."
        sys.exit()
        
    algout=AllAlgs[alg]

    return algout



def TranslateVar(var):
    #print "Translating var: ",var
    
    AllVars={}

    AllVars["m"]=              "M [GeV]"
    AllVars["Tau2"]=           "#tau_{2}"
    AllVars["Tau2Tau1"]=       "#tau_{2}^{wta}"
    AllVars["TauWTA2"]=        "#tau_{21}"
    AllVars["TauWTA2TauWTA1"]= "#tau_{21}^{wta}"
    AllVars["SPLIT12"]=        "#sqrt{d_{12}}"
    AllVars["ZCUT12"]=         "#sqrt{z_{12}}"
    #AllVars["Dip12"]=          "#font[92]{D}"
    AllVars["Dip12"]=          "D"
    #AllVars["PlanarFlow"]=     "#font[92]{P}"
    AllVars["PlanarFlow"]=     "P"
    AllVars["Angularity"]=     "a_{3}"
    AllVars["FoxWolfram20"]=   "R^{FW}_{2}"
    AllVars["Aplanarity"]=     "A"
    AllVars["Sphericity"]=     "S"
    AllVars["ThrustMin"]=      "T_{min}"
    AllVars["ThrustMaj"]=      "T_{maj}"
    AllVars["mu12"]=           "#mu_{12}"
    AllVars["ys12"]=           "#sqrt{y_{12}}"
    AllVars["QJetVolRig1"]=        "#nu_{Q}^{#alpha=1.0}"
    AllVars["QJetVolRig01"]=        "#nu_{Q}^{#alpha=0.1}"
    AllVars["QJetVolRig001"]=        "#nu_{Q}^{#alpha=0.01}"
    AllVars["EECExp2Beta10"]=  "C_{2}^{(#beta=1)}"
    AllVars["EECExp2Beta20"]=  "C_{2}^{(#beta=2)}"
    AllVars["EECExp3Beta10"]=  "D_{2}^{(#beta=1)}"
    AllVars["EECExp3Beta20"]=  "D_{2}^{(#beta=2)}"
    AllVars["SoftDropTag"]=       "L_{SD}"
    AllVars["MassCutOnly"]=    "M^{68% Window}"
    
    if var not in AllVars.keys():
        print "This var is not in your list: ",var," EXITTING ..."
        sys.exit()
        
    varout=AllVars[var]

    return varout


def GetOptimalMCut(outputdir, outputlabel):

    print"Getting mass window: ",outputdir,outputlabel

    mlow  = 70
    mhigh = 110

    fout = TFile(outputdir+"MassWindow_"+outputlabel+".root")
    fout.ls()
    MassWindow = fout.Get("MassWindow")
    mlow  = MassWindow.GetBinContent(1)
    mhigh = MassWindow.GetBinContent(2)
    fout.Close()

    print mlow,mhigh

    return str(mlow),str(mhigh)

def FindMassWindow(InputDir, alg, alglabel, variable, range, logy, pt1, pt2, m1, m2, cutslabel, outputdir, rocoption, debug=0):
    '''Implementation of simple signal and background comparison'''
    print "Find Mass Window: ",alg,variable
    
    c = TCanvas("c","c",300,300)

    dry=0.045

    weight=""
    weight+="weight_mc*weight_pu*"
    weight+="("+alg+"_pt<9999 && "
    weight+=""+alg.replace("LC","TR")+"_pt<9999 && "
    weight+="abs("+alg+"_eta)<1.2 && "
    weight+="CA12TRLEAD_pt>"+pt1+" && "
    weight+="CA12TRLEAD_pt<"+pt2+" && "
    weight+=""+alg+"_TruthRecoMatch && "
    weight+=""+alg+"_m<"+str(m2)+" && "
    weight+=""+alg+"_m>"+str(m1)+")* "
    

# def FindMassWindow(InputDir, alg, alglabel, cuts, cutlabels, outputdir, outputlabel, debug=0):
#     print "Finding mass window by fitting:",alg,cuts
# 
#     dry=0.045
# 
#     weight=""
#     for cut in cuts:
#         if cut!="":
#             weight+="("+cut+")*"
#     weight+="weight_mc*weight_pu*"

    #=======================
    #1D SEPARATION
    #=======================
    print weight
    weight = weight.replace(alg+"_pt","CA12TRLEAD_pt")
    weight = weight.replace(alg+"_eta","CA12TRLEAD_eta")
    print weight

    c = TCanvas("c","c",300,300)

    signalweight=weight+GWW+"*weight_pt"
    if debug: print "SignalWeight: ",signalweight
    sh = GetHist1D(InputDir+"merged.LumpedSignal.root", "ntup_"+alg+"_m", "100,0,200", signalweight)

    bgh = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", "ntup_"+alg+"_m", "100,0,200", weight+JX3)
    bg2 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", "ntup_"+alg+"_m", "100,0,200", weight+JX4)
    #bg3 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", "ntup_"+alg+"_m", "100,0,200", weight+JX5)
    #bg4 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", "ntup_"+alg+"_m", "100,0,200", weight+JX6)

    bgh.Add(bg2)
    #bgh.Add(bg3)
    #bgh.Add(bg4)

    # NORMALIZE
    sh  = NormalizeHist(sh)
    bgh = NormalizeHist(bgh)
    
    # find mass window by fitting
    totsig=sh.Integral()
    totbg=bgh.Integral()
    
    # find mass window by 68% interval method
    MPV,minWidth,valhighpetar=MinWindow( sh, 0.68 )
    vallowpetar = valhighpetar-minWidth
    
    binlowpetar  = sh.FindBin(vallowpetar)
    binhighpetar = sh.FindBin(valhighpetar)
    cutsigpetar = sh.Integral(binlowpetar,binhighpetar)
    cutbgpetar = bgh.Integral(binlowpetar,binhighpetar)
    fracsigpetar = cutsigpetar/totsig
    fracbgpetar = cutbgpetar/totbg
    print "Petar: ",str(vallowpetar),str(valhighpetar),"|",str(binlowpetar),str(binhighpetar),"|",fracsigpetar,fracbgpetar

    myFit =  TF1("myFit","gaus",50,120)
    myFit.SetParameter(0,0.5)
    myFit.SetParameter(1,70)
    myFit.SetParameter(2,20)

    sh.Fit("myFit","R")

    mean  = myFit.GetParameter(1)
    sigma = myFit.GetParameter(2)

    vallow1  = mean-sigma
    valhigh1 = mean+sigma
    binlow1  = sh.FindBin(vallow1)
    binhigh1 = sh.FindBin(valhigh1)
    cutsig1 = sh.Integral(binlow1,binhigh1)
    cutbg1 = bgh.Integral(binlow1,binhigh1)
    fracsig1 = cutsig1/totsig
    fracbg1 = cutbg1/totbg
    print "Fitted 1-sigma: ",str(vallow1),str(valhigh1),"|",str(binlow1),str(binhigh1),"|",fracsig1,fracbg1

    # DRAW
    bgh.SetFillStyle(0)
    bgh.SetLineColor(2)
    bgh.SetLineStyle(2)
    bgh.SetLineWidth(3)
    bgh.GetXaxis().SetTitle("Jet Mass [GeV]")
    bgh.GetXaxis().SetTitleOffset(1.2)
    bgh.GetYaxis().SetTitleOffset(1.7)
    bgh.GetYaxis().SetTitle("Normalised Entries")

    sh.SetFillStyle(0)
    sh.SetLineColor(4)
    sh.SetLineStyle(1)
    sh.SetLineWidth(3)

    maxval = GetMaxVal(bgh, sh)
    bgh.SetMaximum(maxval*2.0)
    bgh.SetMinimum(0.0)

    bgh.Draw("hist")
    sh.Draw("histsame")
    myFit.Draw("LSame")
    
    arrow1a = TArrow()
    arrow1a.SetLineWidth(3)
    arrow1a.SetLineColor(3)
    arrow1a.DrawArrow(vallow1,0,vallow1,0.05,0.5,"same")
    arrow1b = TArrow()
    arrow1b.SetLineWidth(3)
    arrow1b.SetLineColor(3)
    arrow1b.DrawArrow(valhigh1,0,valhigh1,0.05,0.5,"same")
    
    arrow2a = TArrow()
    arrow2a.SetLineWidth(3)
    arrow2a.SetLineColor(7)
    arrow2a.DrawArrow(vallowpetar,0,vallowpetar,0.05,0.5,"same")
    arrow2b = TArrow()
    arrow2b.SetLineWidth(3)
    arrow2b.SetLineColor(7)
    arrow2b.DrawArrow(valhighpetar,0,valhighpetar,0.05,0.5,"same")

    ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
    myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
    myText(       0.50,0.90,1,0.03, alglabel)
    myText(       0.50,0.85,1,0.03, cutslabel)
    myLineBoxText( 0.25,0.80,0.05,2,2,3004,0.03,"QCD jets")
    myLineBoxText( 0.25,0.75,0.05,4,1,3005,0.03,"W jets")
    myLineBoxText( 0.25,0.70,0.05,1,1,3005,0.03,"Signal Fit")
    myLineBoxText( 0.25,0.65,0.05,3,1,3005,0.03,"1#sigma Fit Window")
    myLineBoxText( 0.25,0.60,0.05,7,1,3005,0.03,"68% Window")
    myText(       0.50,0.70,1,0.03, "Sample:  #epsilon_{ 1#sigma-fit} ,  #epsilon_{68% Window}")
    myText(       0.50,0.65,1,0.03, "Signal:  "+str(round(fracsig1,3))+" , "+str(round(fracsigpetar,3)))
    myText(       0.50,0.60,1,0.03, "BG:       "+str(round(fracbg1,3))+" , "+str(round(fracbgpetar,3)))
    c.SaveAs(outputdir+"MassWindow_"+alg+"_"+cutslabel+".eps")



    hout = TH1F()
    hout.Fill("petar68_boundlow",vallowpetar)
    hout.Fill("petar68_boundhigh",valhighpetar)
    hout.Fill("petar68_mpv",MPV)
    hout.Fill("petar68_signaleff",fracsigpetar)
    hout.Fill("petar68_bgeff",fracbgpetar)
    hout.Fill("1sigma_boundlow",vallow1)
    hout.Fill("1sigma_boundhigh",valhigh1)
    hout.Fill("1sigma_mpv",mean)
    hout.Fill("1sigma_signaleff",fracsig1)
    hout.Fill("1sigma_bgeff",fracbg1)


    fout = TFile(outputdir+"MassWindow_"+alg+"_"+cutslabel+".root","RECREATE")
    hout.Write("MassWindow")
    fout.Close()


def NPVCorrelation(InputDir, alg, alglabel, variable, range, logy, pt1, pt2, m1, m2, cutslabel, outputdir, rocoption, debug=0):
    '''Implementation of simple signal and background comparison'''
    print "Find Mass Window: ",alg,variable
    
    c = TCanvas("c","c",300,300)

    dry=0.045

    weight=""
    weight+="weight_mc*weight_pu*"
    weight+="("+alg+"_pt<9999 && "
    weight+=""+alg.replace("LC","TR")+"_pt<9999 && "
    weight+="abs("+alg+"_eta)<1.2 && "
    weight+="CA12TRLEAD_pt>"+pt1+" && "
    weight+="CA12TRLEAD_pt<"+pt2+" && "
    weight+=""+alg+"_TruthRecoMatch && "
    weight+=""+alg+"_m<"+str(m2)+" && "
    weight+=""+alg+"_m>"+str(m1)+")* "

# def NPVCorrelation(InputDir, alg, alglabel, cuts, cutlabels, outputdir, outputlabel, debug=0):
#     print "Finding NPV Correlation:",alg,cuts
# 
#     dry=0.045
# 
#     weight=""
#     for cut in cuts:
#         if cut!="":
#             weight+="("+cut+")*"
#     weight+="weight_mc*weight_pu*"
        
    #=======================
    #1D SEPARATION
    #=======================

#     print weight
#     weight.replace(alg+"_pt","CA12TRLEAD_pt");
#     weight.replace(alg+"_eta","CA12TRLEAD_eta");
#     print weight
#     trmatch="("+alg+"_pt<9000)";
#     weight+=trmatch+"*";
#     print weight
#     trmatch.replace("LC","TR");
#     weight+=trmatch+"*";
#     print weight
    
    
    corr = TH1F()

    alga=alg;
    alga.replace("LC","TR");

    algb=alg;
    algb.replace("LC","TR");

    algc=alg;

    #=======================
    #SIGNAL
    #=======================
    signalweight=weight+GWW+"*weight_pt"
    if debug: print "SignalWeight: ",signalweight
    sh1 = GetHist1D(InputDir+"merged.LumpedSignal.root", "ntup_"+alg+"_m", "50,0,200", signalweight+"*(nvtx<14)")
    sh2 = GetHist1D(InputDir+"merged.LumpedSignal.root", "ntup_"+alg+"_m", "50,0,200", signalweight+"*(nvtx>=14)")

    #=======================
    #BACKGROUND
    #=======================
    bgh1 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX3+"*(nvtx<14)")
    bg21 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX4+"*(nvtx<14)")
    bg31 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX5+"*(nvtx<14)")
    bg41 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX6+"*(nvtx<14)")
    bgh1.Add(bg21)
    bgh1.Add(bg31)
    bgh1.Add(bg41)
    
    bgh2 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX3+"*(nvtx>=14)")
    bg22 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX4+"*(nvtx>=14)")
    bg32 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX5+"*(nvtx>=14)")
    bg42 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", "ntup_"+alg+"_m", "50,0,200", weight+JX6+"*(nvtx>=14)")
    bgh2.Add(bg22)
    bgh2.Add(bg32)
    bgh2.Add(bg42)




    #rescale to unit area
    scale=1.0;
    if sh1.Integral()!=0:
        scale=1.0/sh1.Integral();
    sh1.Scale(scale);
    if bgh1.Integral()!=0:
        scale=1.0/bgh1.Integral();
    bgh1.Scale(scale);

    if sh2.Integral()!=0:
        scale=1.0/sh2.Integral();
    sh2.Scale(scale);
    if bgh2.Integral()!=0:
        scale=1.0/bgh2.Integral();
    bgh2.Scale(scale);

    #maxval
    m1 = bgh1.GetMaximum();
    m2 = bgh2.GetMaximum();
    m3 = sh1.GetMaximum();
    m4 = sh1.GetMaximum();
    maxval=0.0;
    if m1>=maxval:
        maxval=m1;
    if m2>=maxval:
        maxval=m2;
    if m3>=maxval:
        maxval=m3;
    if m4>=maxval:
        maxval=m4;
    bgh1.SetMaximum(maxval*1.6);
    bgh1.SetMinimum(0.0);

    c1 = TCanvas("c1","c1",500,500);

    bgh1.GetXaxis().SetTitle("M(jet) [GeV]");
    bgh1.GetXaxis().SetTitleOffset(1.2);
    bgh1.GetYaxis().SetTitleOffset(1.4);
    bgh1.GetYaxis().SetTitle("Normalised Entries");

    bgh1.SetLineColor(2);
    bgh1.SetLineStyle(1);
    bgh1.SetLineWidth(3);
    bgh1.SetMarkerSize(0);

    bgh2.SetLineColor(2);
    bgh2.SetLineStyle(2);
    bgh2.SetLineWidth(3);
    bgh2.SetMarkerSize(0);

    sh1.SetLineColor(4);
    sh1.SetLineStyle(1);
    sh1.SetLineWidth(3);
    sh1.SetMarkerSize(0);

    sh2.SetLineColor(4);
    sh2.SetLineStyle(2);
    sh2.SetLineWidth(3);
    sh2.SetMarkerSize(0);

    bgh1.Draw("hist");
    bgh2.Draw("histsame");
    sh1.Draw("histsame");
    sh2.Draw("histsame");

    ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal");
    myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV");
    myLineBoxText(0.6,0.69,0.05,2,1,3004,0.03,"QCD jets - N(vtx)<14");
    myLineBoxText(0.6,0.65,0.05,2,2,3004,0.03,"QCD jets - N(vtx)#geq14");
    myLineBoxText(0.6,0.61,0.05,4,1,3004,0.03,"W jets - N(vtx)<14");
    myLineBoxText(0.6,0.57,0.05,4,2,3004,0.03,"W jets - N(vtx)#geq14");
    myText(       0.50,0.90,1,0.03, alg);
    myText(       0.50,0.85,1,0.03, cutslabel)
    
    c1.SaveAs(outputdir+"PileupMassWindow_"+alg+"_"+cutslabel+".eps");
    
    

    #=======================
    #2D SEPARATION
    #=======================
    if debug: print weight
    weight = weight.replace("ntup_"+alg+"_pt","CA12TRLEAD_pt")
    weight = weight.replace("ntup_"+alg+"_eta","CA12TRLEAD_eta")
    if debug: print weight

    signalweight=weight+GWW+"*weight_pt"
    if debug: print "SignalWeight: ",signalweight
    sh = GetHist2D(InputDir+"merged.LumpedSignal.root", "nvtx", "ntup_"+alg+"_m", "30,0,30", "100,0,200", signalweight)

    bgh = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", "nvtx", "ntup_"+alg+"_m", "30,0,30", "100,0,200", weight+JX3)
    bg2 = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", "nvtx", "ntup_"+alg+"_m", "30,0,30", "100,0,200", weight+JX4)
    bg3 = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", "nvtx", "ntup_"+alg+"_m", "30,0,30", "100,0,200", weight+JX5)
    bg4 = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", "nvtx", "ntup_"+alg+"_m", "30,0,30", "100,0,200", weight+JX6)

    bgh.Add(bg2)
    bgh.Add(bg3)
    bgh.Add(bg4)
    
    
    #signal
    sh.GetXaxis().SetRangeUser(0,200)
    sh.GetXaxis().SetTitle("N(vtx)")
    sh.GetYaxis().SetTitle("M(jet) [GeV]")
    psig = sh.ProfileX("psig")
    psig.SetMarkerSize(0.5)
    corrsig = sh.GetCorrelationFactor()
    corr.Fill("sig",corrsig)
    strcorrsig="Corr. = "+str(round(corrsig,3))
    c1 = TCanvas("c1","c1",500,500);
    sh.Draw("colz");
    psig.Draw("psame");
    ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal");
    myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV");
    myText(       0.20,0.80,1,0.03, "W jets");
    myText(       0.20,0.75,1,0.03, strcorrsig);
    myText(       0.50,0.90,1,0.03, alg);
    myText(       0.50,0.85,1,0.03, cutslabel)
    c1.SaveAs(outputdir+"Pileup_"+alg+"_"+cutslabel+"_signal.eps");
    
    
    
    #background
    bgh.GetXaxis().SetRangeUser(0,200)
    bgh.GetXaxis().SetTitle("N(vtx)")
    bgh.GetYaxis().SetTitle("M(jet) [GeV]")
    pbg = bgh.ProfileX("pbg")
    pbg.SetMarkerSize(0.5)
    corrbg = bgh.GetCorrelationFactor()
    corr.Fill("bg",corrbg)
    strcorrbg="Corr. = "+str(round(corrbg,3))
    c1 = TCanvas("c1","c1",500,500);
    bgh.Draw("colz");
    pbg.Draw("psame");
    ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal");
    myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV");
    myText(       0.20,0.80,1,0.03, "QCD jets");
    myText(       0.20,0.75,1,0.03, strcorrbg);
    myText(       0.50,0.90,1,0.03, alg);
    myText(       0.50,0.85,1,0.03, cutslabel)
    c1.SaveAs(outputdir+"Pileup_"+alg+"_"+cutslabel+"_background.eps");

    fout = TFile(outputdir+"PileupCorr_"+alg+"_"+cutslabel+".root","RECREATE");
    corr.Write("NPVCorrelation")
    fout.Close()

def RunSignalBG(InputDir, alg, alglabel, variable, range, cuts, cutlabels, outputdir, outputlabel, debug=0):

    if debug: print range,cuts

    xaxis = GetAxisLabel(variable)

    SignalBGCompare1D(InputDir, alg, alglabel, variable, cuts, cutlabels, xaxis, range, outputdir, outputlabel)

    range_expanded = range

    if debug: print range

    range_expanded = range_expanded.split(",")[0]+"0,"+range_expanded.split(",")[1]+","+range_expanded.split(",")[2]

    if debug: print range_expanded

    SignalBGMakeROCs( InputDir, alg,           variable, cuts, cutlabels, range_expanded, outputdir, outputlabel)




def MakeReferenceGraph( roctype=0 , debug=0):
    '''Make reference graph for plotting'''

    graph = TGraph(1)

    graph.SetTitle("")
    graph.GetXaxis().SetTitleOffset(1.2)
    graph.GetYaxis().SetTitleOffset(1.3)

    graph.GetXaxis().SetRangeUser(0.0,1.0)

    if roctype==0:
        graph.GetXaxis().SetTitle("#epsilon ^{FullTag}_{W Jets}")
        graph.GetYaxis().SetTitle("1- #epsilon ^{FullTag}_{QCD Jets}")
        graph.SetMinimum(0.0)
        graph.SetMaximum(1.0)
    elif roctype==1:
        graph.GetXaxis().SetTitle("#epsilon ^{FullTag}_{W Jets}")
        graph.GetYaxis().SetTitle("1 / #epsilon ^{FullTag}_{QCD Jets}")
        graph.SetMinimum(0.0)
        graph.SetMaximum(1000.0)

    return graph

def GetBGRej(gr, targetseff=0.5, debug=0):
    '''Get BG rejection at fixed targetseff'''

    n = gr.GetN()

    xinit  = 1
    yinit  = 1
    mindistance=10.0
    foundseff=1.0
    foundbgeff=1.0
    itarget=1

    for ipoint in range(n):

        gr.GetPoint(ipoint,xinit,yinit)

        distance = abs(targetseff-xinit)

        if distance<mindistance:
            mindistance = distance
            foundseff   = xinit
            foundbgeff  = yinit
            itarget     = ipoint

    if debug:
        print itarget,"  ",mindistance,"  ",foundseff,"  ",foundbgeff<<"\n"

    return foundbgeff

def GetGraphs( paths , debug=0):
    '''Return a list of graphs corresponding to the paths of ROC files'''

    if debug:
        print "Getting ",paths.size()," graphs"

    graphs=[]

    for path in paths:
        if debug:
            cout<<path<<endl
        tempf = TFile(path)
        gr = tempf.Get("ROCLikelihoodIncl")
        graphs.append(gr)
        tempf.Close()


    return graphs

def ConvertROC_BGRej_BGPow( gr, debug=0):
    '''Converts from bg rejection to power'''

    grTemp = gr
    n=grTemp.GetN()
    
    print "Converting NPoints ",n

    xinit = Double()
    yinit = Double()
    mindistance=10.0
    foundseff=1.0
    foundbgeff=1.0
    itarget=1

    for ipoint in range(n):

        grTemp.GetPoint(ipoint,xinit,yinit)

        xfinal = xinit

        if yinit>0.999:
            yfinal = 10000.0
        else:
            yfinal = 1.0/((1.0-yinit))

        gr.SetPoint(ipoint,xfinal,yfinal)

    return gr


def ConvertROC_BGRej_BGPow_WithUncer( gr, debug=0):
    '''Converts from bg rejection to power'''

    grTemp = gr
    n = grTemp.GetN()
    
    print "Type: ",type(gr)
    
    print "Converting NPoints ",n

    xinit = Double()
    yinit = Double()
    mindistance=10.0
    foundseff=1.0
    foundbgeff=1.0
    itarget=1

    for i in range(n):
    
        sigeff_init = Double()
        bkgrej_init = Double()
        grTemp.GetPoint(i,sigeff_init,bkgrej_init)
        
        sigefferr_init = Double()
        bkgrejerr_init = Double()
        sigefferr_init = gr.GetErrorX(i)
        bkgrejerr_init = gr.GetErrorY(i)


        #no conversion for signal efficiency
        sigeff_final    = sigeff_init
        sigefferr_final = sigefferr_init

        #convert bg rejection
        if bkgrej_init >0.999:
            bkgrej_final  = 10000.0
            bkgrejerr_final = 100.0
        else:
            h_one    = TH1D("h_one","h_one",1,0,1)
            h_one.SetBinContent(1,1.0)
            h_one.SetBinError(1,0.0)
    
            h_bkgrej = TH1D("h_bkgrej","h_bkgrej",1,0,1)
            h_bkgrej.SetBinContent(1,bkgrej_init)
            h_bkgrej.SetBinError(1,bkgrejerr_init)            
        
            h_bkgrej.Add(h_bkgrej,h_one,-1.0,1.0)
            h_one.Divide(h_one,h_bkgrej)
        
            bkgrej_final    = h_one.GetBinContent(1)
            bkgrejerr_final = h_one.GetBinError(1)

        print "Power converted: ",i,sigeff_final,sigefferr_final,bkgrej_final,bkgrejerr_final
        gr.SetPoint(i,sigeff_final,bkgrej_final)
        gr.SetPointError(i,sigefferr_final,bkgrejerr_final)

    return gr


def FoldMassEffGraphs( gr, seff, beff, debug=0):
    if debug:
        print "Fold mass effs",seff,beff
    n = gr.GetN();
    grOut = TGraph(n)
    if debug:
        print seff,"   ",beff
    for i in range(n):

        xinit = Double()
        yinit = Double()
        gr.GetPoint(i,xinit,yinit)
        #print "Init: ",i,"  ",xinit,"  ",yinit

        xfinal = xinit*seff
        yfinal = 1.0-((1-yinit)*(1-beff))
        #print "Final: ",i,"  ",xfinal,"  ",yfinal
        grOut.SetPoint(i,xfinal,yfinal)

    #print type(grOut)
    return grOut

def FoldMassEffGraphs_WithUncer( gr, masssigeff, masssigefferr, massbkgeff, massbkgefferr):
    #print "Fold mass effs",masssigeff,massbkgeff
    n = gr.GetN();
    grOut = TGraphErrors(n)

    #for calculations    
    h_one    = TH1D("h_one","h_one",1,0,1)
    h_one.SetBinContent(1,1.0)
    h_one.SetBinError(1,0.0)
    
    h_masssigeff = TH1D("h_masssigeff","h_masssigeff",1,0,1)
    h_masssigeff.SetBinContent(1,masssigeff)
    h_masssigeff.SetBinError(1,masssigefferr)
    
    h_massbkgeff = TH1D("h_massbkgeff","h_massbkgeff",1,0,1)
    h_massbkgeff.SetBinContent(1,massbkgeff)
    h_massbkgeff.SetBinError(1,massbkgefferr)
    
    h_sig = TH1D("h_sig","h_sig",1,0,1)
    h_bkg = TH1D("h_bkg","h_bkg",1,0,1)
    
    #only do this once for use in the following calculations of converted bgrejection for each loop step
    #to get "(1-masseff)"
    h_massbkgeff.Add(h_massbkgeff,h_one,-1.0,1.0)
    


    for i in range(n):

        sigeffinit = Double()
        bkgrejinit = Double()
        gr.GetPoint(i,sigeffinit,bkgrejinit)
        
        sigeffiniterr = Double()
        bkgrejiniterr = Double()
        sigeffiniterr = gr.GetErrorX(i)
        bkgrejiniterr = gr.GetErrorY(i)

        #print "Init: ",i,"  ",sigeffinit,"  ",sigeffiniterr,"  ",bkgrejinit,"  ",bkgrejiniterr

        h_sig.SetBinContent(1,sigeffinit)
        h_sig.SetBinError(1,sigeffiniterr)
        
        h_sig.Multiply(h_masssigeff)
        
        sigefffinal    = h_sig.GetBinContent(1)
        sigefffinalerr = h_sig.GetBinError(1)
        #print "FromHist: ",sigefffinal,sigefffinalerr

#         sigefffinal = sigeffinit*masssigeff
#         print "FromMultiply: ",sigefffinal,sigefffinalerr
#         sigefffinalerr = 0.0

        h_bkg.SetBinContent(1,bkgrejinit)
        h_bkg.SetBinError(1,bkgrejiniterr)

        #final = 1.0-((1-masseff)(1-init))
        bkgrejfinal = h_bkg.GetBinContent(1)
        #print "step ",bkgrejfinal        
        #(1-init)
        h_bkg.Add(h_bkg,h_one,-1.0,1.0)
        bkgrejfinal = h_bkg.GetBinContent(1)
        #print "step ",bkgrejfinal
        #(1-masseff)(1-init)
        h_bkg.Multiply(h_massbkgeff,h_bkg)
        bkgrejfinal = h_bkg.GetBinContent(1)
        #print "step ",bkgrejfinal        
        #1-(1-masseff)(1-init)
        h_bkg.Add(h_bkg,h_one,-1.0,1.0)
        bkgrejfinal = h_bkg.GetBinContent(1)
        #print "step ",bkgrejfinal        
        
        bkgrejfinal    = h_bkg.GetBinContent(1)
        bkgrejfinalerr = h_bkg.GetBinError(1)
        #print "FromHist: ",bkgrejfinal,bkgrejfinalerr
        
#         bkgrejfinal = 1.0-((1-bkgrejinit)*(1-massbkgeff))
#         bkgrejfinalerr=0.0
#         print "FromMultiply: ",bkgrejfinal,bkgrejfinalerr
        

        #print "Final: ",i,"  ",sigefffinal,"  ",bkgrejfinal

        grOut.SetPoint(i,sigefffinal,bkgrejfinal)
        grOut.SetPointError(i,sigefffinalerr,bkgrejfinalerr)

#         grOut.SetMinimum(0.0)
#         grOut.SetMaximum(0.0)
#         grOut.GetXaxis().SetRangeUser(0.0,1.0)
#         grOut.SetLineColor(2)
#         grOut.SetFillColor(2) 
#         grOut.Draw("CE3Same")

    return grOut


def UnFoldMassEffGraphs( gr, seff, beff, debug=0):
    if debug:
        print "UnFold mass effs",seff,beff
    n = gr.GetN();
    grOut = TGraph(n)
    if debug:
        print seff,"   ",beff
    for i in range(n):

        xinit = Double()
        yinit = Double()
        gr.GetPoint(i,xinit,yinit)
        if debug:
            print "Init: ",i,"  ",xinit,"  ",yinit

        xfinal = xinit/seff
        yfinal = 1.0-((1.0-yinit)/beff)

        if debug:
            print "Final: ",i,"  ",xfinal,"  ",yfinal
        grOut.SetPoint(i,xfinal,yfinal)

    return grOut

def RocCurve_SoverBOrdered(sig, bg, debug=0):
    print 'Make ROC curve using S over B ordering'
    
    binnum=[]
    s=[]
    b=[]
    r=[]

    for i in range(1,sig.GetNbinsX()+1):
        #print "bin ",i

        binnumtemp = i
        stemp = sig.GetBinContent(i)
        btemp = bg.GetBinContent(i)

        if btemp==0:
            rtemp=0.0
        else:
            rtemp = stemp/btemp

        binnum.append(binnumtemp)
        s.append(stemp)
        b.append(btemp)
        r.append(rtemp)

    for i in range(len(s)):
        ifix = len(s)-i
        #print ifix
        for j in range(0,ifix-1):
            if r[j]<r[j+1]:
                tbinnum=binnum[j]
                tr=r[j]
                tb=b[j]
                ts=s[j]

                binnum[j]=binnum[j+1]
                r[j] = r[j+1]
                b[j] = b[j+1]
                s[j] = s[j+1]

                binnum[j+1]=tbinnum
                r[j+1] = tr
                b[j+1] = tb
                s[j+1] = ts



    totalB = sum(b)
    totalS = sum(s)

    n = len(s)
    print n
    
    hsigout = TH1F("hsigout","hsigout",n,0,1)
    hsigout.SetDirectory(0)
    hbkgout = TH1F("hbkgout","hbkgout",n,0,1)
    hbkgout.SetDirectory(0)
    
    siglow  = sig.GetXaxis().GetXmin()
    sighigh = sig.GetXaxis().GetXmax()
    h1 = TH1F("h1","h1",n,siglow,sighigh)
    h1.SetDirectory(0)
    
    print "Npoints: ",n
    gr = TGraph(n)
    for i in range(0,n):
    


        #fill roc points
        gr.SetPoint(i, myS, (1-myB))

        #fill signal regions histogram        
        if myS<=0.73:
            h1.SetBinContent(binnum[i], s[i])
        
        
    #get histograms that are colored for signal efficiency at 50%
    
    

    return gr,hsigout,hbkgout,h1

def RocCurve_SoverBOrdered_WithUncer(sig, bg, debug=0):
    print 'Make ROC curve using S over B ordering'
    
    binnum=[]
    s=[]
    serr=[]
    b=[]
    berr=[]
    r=[]

    for i in range(1,sig.GetNbinsX()+1):
        #print "bin ",i

        binnumtemp = i
        stemp = sig.GetBinContent(i)
        serrtemp = sig.GetBinError(i)
        btemp = bg.GetBinContent(i)
        berrtemp = bg.GetBinError(i)

        if btemp==0:
            rtemp=0.0
        else:
            rtemp = stemp/btemp

        binnum.append(binnumtemp)
        s.append(stemp)
        serr.append(serrtemp)
        b.append(btemp)
        berr.append(berrtemp)
        r.append(rtemp)

    for i in range(len(s)):
        ifix = len(s)-i
        #print ifix
        for j in range(0,ifix-1):
            if r[j]<r[j+1]:
#                 tbinnum=binnum[j]
#                 tr=r[j]
#                 tb=b[j]
#                 ts=s[j]
# 
#                 binnum[j]=binnum[j+1]
#                 r[j] = r[j+1]
#                 b[j] = b[j+1]
#                 s[j] = s[j+1]
# 
#                 binnum[j+1]=tbinnum
#                 r[j+1] = tr
#                 b[j+1] = tb
#                 s[j+1] = ts

                binnum[j],binnum[j+1]=binnum[j+1],binnum[j]
                b[j],b[j+1]=b[j+1],b[j]
                berr[j],berr[j+1]=berr[j+1],berr[j]
                s[j],s[j+1]=s[j+1],s[j]
                serr[j],serr[j+1]=serr[j+1],serr[j]
                r[j],r[j+1]=r[j+1],r[j]


    #make reordered histograms
    n = len(s)
    print n
    
    hsigout = TH1F("hsigout","hsigout",n,0,1)
    hsigout.SetDirectory(0)
    hbkgout = TH1F("hbkgout","hbkgout",n,0,1)
    hbkgout.SetDirectory(0)
    for i in range(0,n):
        hsigout.SetBinContent(i+1,s[i])
        hsigout.SetBinError(i+1,serr[i])
        hbkgout.SetBinContent(i+1,b[i])
        hbkgout.SetBinError(i+1,berr[i])


    
    siglow  = sig.GetXaxis().GetXmin()
    sighigh = sig.GetXaxis().GetXmax()
    h1 = TH1F("h1","h1",n,siglow,sighigh)
    h1.SetDirectory(0)
    
    print "Npoints: ",n
    gr = TGraphErrors(n)
    for i in range(1,n):

        #integrate from 0 to i
        myBerr=Double()
        mySerr=Double()
        myB = hbkgout.IntegralAndError(0,i,myBerr)
        myS = hsigout.IntegralAndError(0,i,mySerr)
        print i,"  myS=",myS,mySerr,"  myB=",myB,myBerr
        gr.SetPoint(i, myS, (1-myB))
        gr.SetPointError(i, mySerr, myBerr)

        #get histograms that are colored for signal efficiency at 50%
        if myS<=0.73:
            print binnum[i]
            print s[i]
            h1.SetBinContent(binnum[i], s[i])
        
    return gr,hsigout,hbkgout,h1


def RocCurve_SoverBOrdered2D(sig, bg, debug=0):
    print 'Make ROC curve using S over B ordering for 2D'

    c0 = TCanvas("c0","c0",300,300);
    
    ratio = sig.Clone("ratio")
    ratio.Draw("colz")
    #c0.SaveAs("test1.eps")  
    
#    hsum = sig.Clone("hsum")
#    hsum.Add(bg)

    hsum = bg.Clone("hsum")

    
    hsum.Draw("colz")
    #c0.SaveAs("test2.eps")       
    
    ratio.Divide(hsum)
    ratio.Draw("colz")
    #c0.SaveAs("test3.eps")    
    
    cb = TCanvas("cb","cb",900,300);
    cb.Divide(3,1);
    cb1 = cb.cd(1);
    cb2 = cb.cd(2);
    cb3 = cb.cd(3);
    cb1.cd()
    bg.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    cb2.cd()
    sig.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    cb3.cd()
    ratio.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    
    #cb.SaveAs("test4.eps")    
    
    binnumX=[]
    binnumY=[]
    s=[]
    b=[]
    r=[]

    #convert hist to 1D histogram
    for i in range(1,sig.GetNbinsX()+1):
        for j in range(1,sig.GetNbinsY()+1):
            #print "bin ",i,j

            binnumtempX = i
            binnumtempY = j
            stemp = sig.GetBinContent(i,j)
            btemp = bg.GetBinContent(i,j)
            rtemp = ratio.GetBinContent(i,j)

            binnumX.append(binnumtempX)
            binnumY.append(binnumtempY)
            s.append(stemp)
            b.append(btemp)
            r.append(rtemp)

    #order bins
    for i in range(len(s)):
        ifix = len(s)-i
        #print ifix
        for j in range(1,ifix-1):
            #print "reordering: ",j,str(j+1),r[j],r[j+1]
            if r[j]<r[j+1]:
                tbinnumX=binnumX[j]
                tbinnumY=binnumY[j]
                tr=r[j]
                tb=b[j]
                ts=s[j]

                binnumX[j]=binnumX[j+1]
                binnumY[j]=binnumY[j+1]
                r[j] = r[j+1]
                b[j] = b[j+1]
                s[j] = s[j+1]

                binnumX[j+1]=tbinnumX
                binnumY[j+1]=tbinnumY
                r[j+1] = tr
                b[j+1] = tb
                s[j+1] = ts
              
    #print "ORDERED:"
    #for j in range(len(s)):
    #    print binnumX[j], binnumY[j],r[j],b[j],s[j]

    totalB = sum(b)
    totalS = sum(s)
    n = len(s)
    
    hsigout = TH1F("hsigout","hsigout",n,0,1)
    hsigout.SetDirectory(0)
    hbkgout = TH1F("hbkgout","hbkgout",n,0,1)
    hbkgout.SetDirectory(0)
    
    siglowX  = sig.GetXaxis().GetXmin()
    sighighX = sig.GetXaxis().GetXmax()
    nbinsX   = sig.GetXaxis().GetNbins()
    siglowY  = sig.GetYaxis().GetXmin()
    sighighY = sig.GetYaxis().GetXmax()
    nbinsY   = sig.GetYaxis().GetNbins()
    
    h1 = TH2F("h1","h1",nbinsX,siglowX,sighighX,nbinsY,siglowY,sighighY)
    h1.SetDirectory(0)
    
    print "Npoints: ",n
    gr = TGraph(n)
    for i in range(0,n):
    
        hsigout.SetBinContent(i+1,s[i])
        hbkgout.SetBinContent(i+1,b[i])
    
        myS = 0.
        myB = 0.

        for j in range(i):
            myS += s[j]/totalS
            myB += b[j]/totalB

        #print i,myS,(1-myB)

        #print "Fillingroc: ",i,"  ",s[i],"  ",b[i],"  ",myS,"  ",(1-myB)
        gr.SetPoint(i, myS, (1-myB))
        
        if myS<=0.50:
        #    print "CutHist: ",binnum[i],"  ",s[i]
            h1.SetBinContent(binnumX[i], binnumY[i], 1)
        
        
    #get histograms that are colored for signal efficiency at 50%
    
    cc = TCanvas("cc","cc",1200,300);
    cc.Divide(4,1);
    cb1 = cc.cd(1);
    cb2 = cc.cd(2);
    cb3 = cc.cd(3);
    cb4 = cc.cd(4);
    cb1.cd()
    bg.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    cb2.cd()
    sig.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    cb3.cd()
    ratio.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    cb4.cd()
    h1.Draw("colz")
    ATLASLabel(   0.20,0.90,1,0.12,0.04,"Internal")
    #cc.SaveAs("test5.eps")    

    return gr,hsigout,hbkgout,h1

def RocCurve_SingleSided(sig, bg, rightleft, debug=0):
    print "\n\nNO UNCER Make ROC curve using S over B ordering",rightleft

    s=[]
    b=[]
    r=[]

    for i in range(0,sig.GetNbinsX()):
        stemp = sig.GetBinContent(i)
        btemp = bg.GetBinContent(i)

        if btemp==0:
            rtemp=0.0
        else:
            rtemp = stemp/btemp

        s.append(stemp)
        b.append(btemp)
        r.append(rtemp)

    totalB = sum(b)
    totalS = sum(s)

    n = len(s)
    if debug: print n

    gr = TGraph(n)
    for i in range(0,n):
        myS = 0.
        myB = 0.
        
        if rightleft=="R":
            #loop grom i to end
            for j in range(i,len(s)):
                myS += s[j]/totalS
                myB += b[j]/totalB
            print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS, (1-myB))
        elif rightleft=="L":
            #loop grom 0 to i
            for j in range(i):
                myS += s[j]/totalS
                myB += b[j]/totalB
            print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS, (1-myB))
        else:
            print "You did not choose a left or right handed cut - EXITTING ..."
            sys.exit()
            
    ctest = TCanvas("ctest","ctest",400,400)
    gr.SetMinimum(0.0)
    gr.SetMaximum(1.0)
    gr.GetXaxis().SetRangeUser(0.0,1.0)
    gr.Draw("AC")

    return gr
 
 
def RocCurve_SingleSided_WithUncer(sig, bkg, rightleft=0, debug=0):
    print "\n\nMake ROC curve using right/left cut",rightleft

    n = bkg.GetNbinsX()
    print "NBins",n

    totalBerr=Double()
    totalSerr=Double()
    totalB = bkg.IntegralAndError(0,n,totalBerr)
    totalS = sig.IntegralAndError(0,n,totalSerr)
    
    siglow  = sig.GetXaxis().GetXmin()
    sighigh = sig.GetXaxis().GetXmax()
    hsigreg50 = TH1F("hsigreg50","hsigreg50",n,siglow,sighigh)
    hsigreg50.SetDirectory(0)
    hcutval50 = TH1F("hcutval50","hcutval50",5,0,5)
    hcutval50.SetDirectory(0)
    hcutval50.GetXaxis().SetBinLabel(1,"Left(0) , Right(1)")
    hcutval50.GetXaxis().SetBinLabel(2,"LowerCut")
    hcutval50.GetXaxis().SetBinLabel(3,"UpperCut")
    hsigreg25 = TH1F("hsigreg25","hsigreg25",n,siglow,sighigh)
    hsigreg25.SetDirectory(0)
    hcutval25 = TH1F("hcutval25","hcutval25",5,0,5)
    hcutval25.SetDirectory(0)
    hcutval25.GetXaxis().SetBinLabel(1,"Left(0) , Right(1)")
    hcutval25.GetXaxis().SetBinLabel(2,"LowerCut")
    hcutval25.GetXaxis().SetBinLabel(3,"UpperCut")
    if rightleft=="R":
        hcutval50.SetBinContent(1,1)
        hcutval50.SetBinContent(3,sig.GetXaxis().GetBinLowEdge(n)+sig.GetXaxis().GetBinWidth(n))
        extrema50 = 100000
        hcutval25.SetBinContent(1,1)
        hcutval25.SetBinContent(3,sig.GetXaxis().GetBinLowEdge(n)+sig.GetXaxis().GetBinWidth(n))
        extrema25 = 100000
    elif rightleft=="L":
        hcutval50.SetBinContent(1,0)
        hcutval50.SetBinContent(2,sig.GetXaxis().GetBinLowEdge(1))
        extrema50 = -100000
        hcutval25.SetBinContent(1,0)
        hcutval25.SetBinContent(2,sig.GetXaxis().GetBinLowEdge(1))
        extrema25 = -100000

    gr = TGraphErrors(n)
    for i in range(1,n+1):
        myS = 0.
        myB = 0.

        if rightleft=="R":
            #loop grom i to end
            myBerr=Double()
            mySerr=Double()
            myB = bkg.IntegralAndError(i,n,myBerr)
            myS = sig.IntegralAndError(i,n,mySerr)
            print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS, (1-myB))
            gr.SetPointError(i, mySerr, myBerr)
            if myS<=0.73:
                hsigreg50.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)
                print tempex,extrema50
                if tempex<extrema50:
                    extrema50 = tempex
                    print "found extrema R: ",extrema50
            if myS<=0.36:
                hsigreg25.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)
                print tempex,extrema25
                if tempex<extrema25:
                    extrema25 = tempex
                    print "found extrema R: ",extrema50
        elif rightleft=="L":
            #loop grom 0 to i
            myBerr=Double()
            mySerr=Double()
            myB = bkg.IntegralAndError(1,i,myBerr)
            myS = sig.IntegralAndError(1,i,mySerr)
            print i,"  myS=",myS,"  myB=",myB
            gr.SetPoint(i, myS, (1-myB))
            gr.SetPointError(i, mySerr, myBerr)
            if myS<=0.73:
                hsigreg50.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)+sig.GetXaxis().GetBinWidth(i)
                print tempex,extrema50
                if tempex>extrema50:
                    extrema50 = tempex
                    print "found extrema L: ",extrema50
            if myS<=0.36:
                hsigreg25.SetBinContent(i, sig.GetBinContent(i))
                tempex=sig.GetXaxis().GetBinLowEdge(i)+sig.GetXaxis().GetBinWidth(i)
                print tempex,extrema25
                if tempex>extrema25:
                    extrema25 = tempex
                    print "found extrema L: ",extrema50
            
            
        else:
            print "You did not choose a left or right handed cut - EXITTING ..."
            sys.exit()
            
    #artificially set the first point to (1,1) to avoid overflow issues
    gr.SetPoint(0, 1.0, 1.0)
    gr.SetPointError(0, 0.0, 0.0)
            
    ctest = TCanvas("ctest","ctest",400,400)
    gr.SetMinimum(0.0)
    gr.SetMaximum(1.0)
    gr.GetXaxis().SetRangeUser(0.0,1.0)
    gr.Draw("AE3")
    
    if rightleft=="R":
        hcutval50.SetBinContent(2,extrema50)
        hcutval25.SetBinContent(2,extrema25)
    elif rightleft=="L":
        hcutval50.SetBinContent(3,extrema50)
        hcutval25.SetBinContent(3,extrema25)


    print "RETURNING"
    return gr,hsigreg50,hcutval50,hsigreg25,hcutval25

def RocCurve_Petar_Original(sig, bg, debug=0):
    '''Make ROC curve using the min window that fits some fraction of signal'''

    se=[]
    sereal=[]
    be=[]
    cutlow=[]
    cuthigh=[]

    r=arange(0.0,1.0,0.04)
    
    totsig=sig.Integral()
    totbg=bg.Integral()

    for i in r:
        print i
        # find mass window by 68% interval method
        MPV,minWidth,valhighpetar=MinWindow( sig, i )
        vallowpetar = valhighpetar-minWidth
    
        binlowpetar  = sig.FindBin(vallowpetar)
        binhighpetar = sig.FindBin(valhighpetar)
        cutsigpetar  = sig.Integral(binlowpetar,binhighpetar)
        cutbgpetar   = bg.Integral(binlowpetar,binhighpetar)
        fracsigpetar = cutsigpetar/totsig
        fracbgpetar  = cutbgpetar/totbg
        
        print vallowpetar,valhighpetar,fracsigpetar,fracbgpetar


#THIS IS MORE CORRECT AND NEEDS TO BE CHANGED BACK
        se.append(fracsigpetar)
#BECKY USES THIS BELOW
#        se.append(i)
        sereal.append(fracsigpetar)
        be.append(fracbgpetar)
        cutlow.append(vallowpetar)
        cuthigh.append(valhighpetar)
        
    print se
    print be
    print cutlow
    print cuthigh
    
    n = len(se)
    if debug: print n

    print "Npoints: ",n
    gr = TGraph(n)
    hlow=TH1F()
    hhigh=TH1F()
    hsigeff=TH1F()
    hbeeff=TH1F()
    for i in range(0,n):
        gr.SetPoint(i, se[i], (1-be[i]))
        hlow.Fill(str(i),cutlow[i])
        hhigh.Fill(str(i),cuthigh[i])
        hsigeff.Fill(str(i),sereal[i])
        hbeeff.Fill(str(i),be[i])

    return gr,hlow,hhigh,hsigeff,hbeeff
 
 
def RocCurve_Petar(sig, bkg, debug=0):
    '''Make ROC curve using the min window that fits some fraction of signal'''

    se=[]
    seerr=[]
    sereal=[]
    be=[]
    beerr=[]
    cutmpv=[]
    cutlow=[]
    cuthigh=[]

    sigefflow  = 0.04
    sigeffhigh = 1.00
    sigeffstep = 0.02
    r=arange(sigefflow,sigeffhigh,sigeffstep)
    
    nbins = sig.GetNbinsX()
    
    totsigerr = Double()
    totsig    = sig.IntegralAndError(0,nbins,totsigerr)
    totbkgerr = Double()
    totbkg    = bkg.IntegralAndError(0,nbins,totbkgerr)
    
    print "TotalHists: ",totsig,totsigerr,totbkg,totbkgerr

    for i in r:
        print i
        # find mass window by 68% interval method
        MPV,minWidth,valhighpetar = MinWindow( sig, i )
        vallowpetar = valhighpetar-minWidth
    
        print "PetarCuts: ",MPV,minWidth,vallowpetar,valhighpetar
    
        binlowpetar  = sig.FindBin(vallowpetar)
        binhighpetar = sig.FindBin(valhighpetar)
        
        cutsigpetarerr  = Double()
        cutsigpetar     = sig.IntegralAndError(binlowpetar,binhighpetar,cutsigpetarerr)
        cutbkgpetarerr  = Double()
        cutbkgpetar     = bkg.IntegralAndError(binlowpetar,binhighpetar,cutbkgpetarerr)
        
        num=TH1D("num","num",1,0,1)
        den=TH1D("den","den",1,0,1)
        
        #signal
        print "\n\nSignal"
        num.SetBinContent(1,cutsigpetar)
        num.SetBinError(1,cutsigpetarerr)
        den.SetBinContent(1,totsig)
        den.SetBinError(1,totsigerr)
        print "Numerator:   ",num.GetBinContent(1),num.GetBinError(1)
        print "Denominator: ",den.GetBinContent(1),den.GetBinError(1)
        num.Divide(den)
        print num.GetBinContent(1),num.GetBinError(1)
        fracsigpetar    = num.GetBinContent(1)
        fracsigpetarerr = num.GetBinError(1)
        print fracsigpetar
        
        #background
        print "\n\nBackground"
        num.SetBinContent(1,cutbkgpetar)
        num.SetBinError(1,cutbkgpetarerr)
        den.SetBinContent(1,totbkg)
        den.SetBinError(1,totbkgerr)
        print "Numerator:   ",num.GetBinContent(1),num.GetBinError(1)
        print "Denominator: ",den.GetBinContent(1),den.GetBinError(1)
        num.Divide(den)
        print num.GetBinContent(1),num.GetBinError(1)
        fracbkgpetar    = num.GetBinContent(1)
        fracbkgpetarerr = num.GetBinError(1)
        print fracbkgpetar        
        
        #append values to list 
        se.append(fracsigpetar)
        seerr.append(fracsigpetarerr)
        sereal.append(fracsigpetar)
        be.append(fracbkgpetar)
        beerr.append(fracbkgpetarerr)
        cutmpv.append(MPV)
        cutlow.append(vallowpetar)
        cuthigh.append(valhighpetar)
        
    #Make graph for returning
    n = len(se)
    print "Npoints: ",n
    gr = TGraphErrors(n)

    hsigeff=TH1F()
    hsigefferr=TH1F()
    hbkgeff=TH1F()
    hbkgefferr=TH1F()

    hmpv=TH1F()
    hlow=TH1F()
    hhigh=TH1F()

    for i in range(0,n):
        gr.SetPoint(i, se[i], (1-be[i]))
        gr.SetPointError(i, seerr[i], beerr[i])

        hsigeff   .Fill(str(i),se[i])
        hsigefferr.Fill(str(i),seerr[i])
        hbkgeff   .Fill(str(i),be[i])
        hbkgefferr.Fill(str(i),beerr[i])

        hlow      .Fill(str(i),cutlow[i])
        hmpv      .Fill(str(i),cutmpv[i])
        hhigh     .Fill(str(i),cuthigh[i])

    return gr, hmpv,hlow,hhigh, hsigeff,hbkgeff,hsigefferr,hbkgefferr


'''
void GetMassEfficiency(InputDir, algorithm, jettype, groomtype, variable, varLabel, vector<TString> cuts, cut1, cut2, cut3, xaxis, range, int zlog, outname, outputdir, int rocdir){

    atlaslabel="Internal"

    float dry=0.045

    weight=""
    for(int i=0 i<(int)cuts.size() i++){
        weight+="("+cuts.at(i)+")*"
    }
    weight+="weight_mc*"

    #=======================
    #1D SEPARATION
    #=======================

    cout<<weight<<endl
    weight.replace(algorithm+"_pt","CA12Truth_pt")
    weight.replace(algorithm+"_eta","CA12Truth_eta")
    cout<<weight<<endl

    #=======================
    #SIGNAL
    #=======================
    TFile *sig = new TFile(InputDir+"LumpedSignal.root")
    NTupVTagepW.Draw(algorithm+"_"+variable+">>sh("+range+")",weight+m_GWW+"*(SignalPtWeight(CA12Truth_pt))","e")
    TH1D *sh = (TH1D*)gDirectory.Get("sh")

    #=======================
    #BACKGROUND
    #=======================
    TFile *bg = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt280_500_CJetVetoBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bgh1d("+range+")",weight+m_GWW,"e")
    TH1D *bgh1d = (TH1D*)gDirectory.Get("bgh1d")
    bgh1d.Reset()

    TFile *bc1 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt140_280_CJetVetoBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch11d("+range+")",weight+m_wmunu140_l,"e")
    TH1D *bch11d = (TH1D*)gDirectory.Get("bch11d")
    TFile *bc2 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt140_280_CJetFilterBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch21d("+range+")",weight+m_wmunu140_c,"e")
    TH1D *bch21d = (TH1D*)gDirectory.Get("bch21d")
    TFile *bc3 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt140_280_BFilter.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch31d("+range+")",weight+m_wmunu140_b,"e")
    TH1D *bch31d = (TH1D*)gDirectory.Get("bch31d")

    TFile *bc4 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt500_CJetVetoBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch41d("+range+")",weight+m_wmunu500_l,"e")
    TH1D *bch41d = (TH1D*)gDirectory.Get("bch41d")
    TFile *bc5 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt500_CJetFilterBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch51d("+range+")",weight+m_wmunu500_c,"e")
    TH1D *bch51d = (TH1D*)gDirectory.Get("bch51d")
    TFile *bc6 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt500_BFilter.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch61d("+range+")",weight+m_wmunu500_b,"e")
    TH1D *bch61d = (TH1D*)gDirectory.Get("bch61d")

    TFile *bc7 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt280_500_CJetVetoBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch71d("+range+")",weight+m_wmunu280_l,"e")
    TH1D *bch71d = (TH1D*)gDirectory.Get("bch71d")
    TFile *bc8 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt280_500_CJetFilterBVeto.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch81d("+range+")",weight+m_wmunu280_c,"e")
    TH1D *bch81d = (TH1D*)gDirectory.Get("bch81d")
    TFile *bc9 = new TFile(InputDir+"Sherpa_CT10_WmunuMassiveCBPt280_500_BFilter.root")
    NTupVTag.Draw(algorithm+"_"+variable+">>bch91d("+range+")",weight+m_wmunu280_b,"e")
    TH1D *bch91d = (TH1D*)gDirectory.Get("bch91d")

    bgh1d.Add(bch11d)
    bgh1d.Add(bch21d)
    bgh1d.Add(bch31d)
    bgh1d.Add(bch41d)
    bgh1d.Add(bch51d)
    bgh1d.Add(bch61d)
    bgh1d.Add(bch71d)
    bgh1d.Add(bch81d)
    bgh1d.Add(bch91d)

    sh1d.GetXaxis().SetTitle(xaxis)
    sh1d.GetYaxis().SetTitleOffset(1.6)

    bgh1d.GetXaxis().SetTitle(xaxis)
    bgh1d.GetYaxis().SetTitleOffset(1.6)

    #rescale to unit area
    float scale=1.0
    if sh1d.Integral()!=0:
        scale=1.0/sh1d.Integral()
    sh1d.Scale(scale)
    if bgh1d.Integral()!=0:
        scale=1.0/bgh1d.Integral()
    bgh1d.Scale(scale)

    bgh1d.GetXaxis().SetTitle(xaxis)
    bgh1d.GetXaxis().SetTitleOffset(1.2)
    bgh1d.GetYaxis().SetTitleOffset(1.7)
    bgh1d.GetYaxis().SetTitle("Normalised Entries")
    bgh1d.SetLineColor(2)

    float scalefactor=1
    if sh.Integral()!=0:
        scalefactor=bgh.Integral()/sh.Integral()
    sh.Scale(scalefactor)
    sh.SetLineColor(4)
    sh.SetLineWidth(4)

    float maxval=1
    if bgh.GetMaximum()>sh.GetMaximum())
        maxval=bgh.GetMaximum()
    else
        maxval=sh.GetMaximum()
    bgh.SetMaximum(maxval*2.0)
    bgh.SetLineWidth(4)
    bgh.SetLineStyle(2)


    c = TCanvas("c","c",300,300)

    bgline = bgh
    sigline = sh

    bgh.SetFillColor(2)
    bgh.SetLineColor(2)
    bgh.SetLineStyle(2)
    bgh.SetFillStyle(3004)
    bgh.SetMarkerSize(0)

    sh.SetFillColor(4)
    sh.SetLineColor(4)
    sh.SetLineStyle(1)
    sh.SetFillStyle(3005)
    sh.SetMarkerSize(0)

    #calculate efficiency
    int bin1 = sh1d.FindBin(70.0)
    int bin2 = sh1d.FindBin(110.0)

    cout<<"Bins: "<<bin1<<"  "<<bin2<<endl


    float sigtotal=sh1d.Integral()
    float sigcut=sh1d.Integral(bin1,bin2)
    float seff=1.0
    if(sigtotal<=0)
        cout<<"something is wrong with your signal integral"<<endl
    else
        seff = sigcut/sigtotal

    float bgtotal=bgh1d.Integral()
    float bgcut=bgh1d.Integral(bin1,bin2)
    float beff=1.0
    if(bgtotal<=0)
        cout<<"something is wrong with your background integral"<<endl
    else
        beff = bgcut/bgtotal

    cout<<"Seff="<<seff<<"     Beff="<<beff<<endl

    TH1F *heff = new TH1F()
    heff.Fill("beff",beff)
    heff.Fill("seff",seff)


    TFile *f = new TFile(outputdir+"MassEfficiency_70110_"+outname+"_"+algorithm+"_"+variable.replace("\\","")+".root","RECREATE")
    heff.Write("heff")
    f.Close()

    #CLOSE FILES
    sig  .Close()
    bg  .Close()
    bc1 .Close()
    bc2 .Close()
    bc3 .Close()
    bc4 .Close()
    bc5 .Close()
    bc6 .Close()
    bc7 .Close()
    bc8 .Close()
    bc9 .Close()

}
'''

def ReformatRegion( reg ):
    tempRegion = reg
    tempRegion = tempRegion.replace("eta","#eta^{TR}=").replace("-1.21.2","[-1.2,1.2] ")

    tempRegion = tempRegion.replace("pt","p_{T}^{TR}=").replace("200350", "[200,350] GeV ")
    tempRegion = tempRegion.replace("pt","p_{T}^{TR}=").replace("350500", "[350,500] GeV ")
    tempRegion = tempRegion.replace("pt","p_{T}^{TR}=").replace("5001000","[500,1000] GeV ")

    tempRegion = tempRegion.replace("m","M^{Reco}=").replace("70110", "[200,350] GeV ")

    return tempRegion

def GetLegendParameters( nplot ):

    color  = (nplot%9)+1
    style  = (nplot/9)+1
    shiftX = (nplot/16)
    shiftY = (nplot%16)

    if shiftX!=0:
        shiftY+=1

    return color,style,shiftX,shiftY

def GetAxisLabel( initlabel ):

    outlabel = initlabel

    #print "VaribaleSet=",outlabel

    if initlabel=="m": outlabel="Mass [GeV]";
    if initlabel=="Tau1": outlabel= "#tau_{1}";
    if initlabel=="Tau2": outlabel= "#tau_{2}";
    if initlabel=="Tau3": outlabel= "#tau_{3}";
    if initlabel=="Tau2Tau1": outlabel= "#tau_{21}";
    if initlabel=="TauWTA1": outlabel= "#tau_{1}^{WTA}";
    if initlabel=="TauWTA2": outlabel= "#tau_{1}^{WTA}";
    if initlabel=="TauWTA3": outlabel= "#tau_{1}^{WTA}";
    if initlabel=="TauWTA2TauWTA1": outlabel= "#tau_{21}^{WTA}";
    if initlabel=="WIDTH": outlabel= "W";
    if initlabel=="SPLIT12": outlabel= "#sqrt{d_{12}} [GeV]";
    if initlabel=="SPLIT23": outlabel= "#sqrt{d_{23}} [GeV]";
    if initlabel=="SPLIT34": outlabel= "#sqrt{d_{34}} [GeV]";
    if initlabel=="ZCUT12": outlabel= "ZCut(1,2)";
    if initlabel=="ZCUT23": outlabel= "ZCut(2,3)";
    if initlabel=="ZCUT34": outlabel= "ZCut(3,4)";
    if initlabel=="Dip12": outlabel= "Dipolarity(12)";
    if initlabel=="Dip13": outlabel= "Dipolarity(13)";
    if initlabel=="Dip23": outlabel= "Dipolarity(23)";
    if initlabel=="DipExcl12": outlabel= "Dipolarity^{Excl}(12)";
    if initlabel=="PlanarFlow": outlabel= "P";
    if initlabel=="Angularity": outlabel= "Angularity";
    if initlabel=="ActiveArea": outlabel= "Area_{Active}";
    if initlabel=="VoronoiArea": outlabel= "Area_{Voronoi}";
    if initlabel=="QW": outlabel= "QW";
    if initlabel=="PullMag": outlabel= "Pull Magnitude";
    if initlabel=="PullPhi": outlabel= "Pull #phi";
    if initlabel=="PullC00": outlabel= "Pull C_{00}";
    if initlabel=="PullC01": outlabel= "Pull C_{01}";
    if initlabel=="PullC10": outlabel= "Pull C_{10}";
    if initlabel=="PullC11": outlabel= "Pull C_{11}";
    if initlabel=="FoxWolfram0": outlabel= "Fox Wolfram 0";
    if initlabel=="FoxWolfram1": outlabel= "Fox Wolfram 1";
    if initlabel=="FoxWolfram2": outlabel= "Fox Wolfram 2";
    if initlabel=="FoxWolfram3": outlabel= "Fox Wolfram 3";
    if initlabel=="FoxWolfram4": outlabel= "Fox Wolfram 4";
    if initlabel=="Aplanarity": outlabel= "Aplanarity";
    if initlabel=="Sphericity": outlabel= "Sphericity";
    if initlabel=="ThrustMin": outlabel= "Thrust_{Min}";
    if initlabel=="ThrustMaj": outlabel= "Thrust_{Major}";
    if initlabel=="MassDrop": outlabel= "#mu_{12}";
    if initlabel=="ysplit": outlabel= "#sqrt{y_{f}}";
    if initlabel=="QJetVol": outlabel= "#nu_{QJets}";
    if initlabel=="TJetVol": outlabel= "#nu_{TJets}";
    if initlabel=="EEC": outlabel= "Energy Energy Corr.";


    #replace the trailing comma with a blank

    return outlabel

def GetVariableSet( initlabel ):

    outlabel = initlabel

    #print "VaribaleSet=",outlabel

    #replace all of the variable names with nice latex formatted names
    outlabel = outlabel.replace("m,","Mass,")
    outlabel = outlabel.replace("SPLIT12,","#sqrt{d_{12}},")
    outlabel = outlabel.replace("Tau2Tau1,","#tau_{21},")
    outlabel = outlabel.replace("MassDropSplit,","#mu_{12},")
    outlabel = outlabel.replace("PlanarFlow,","P,")
    outlabel = outlabel.replace("ys12,","#sqrt{y_{f}},")

    #replace the trailing comma with a blank
    outlabel = outlabel[:-1]

    return outlabel

def GetAlgLabel( initlabel ):

    print "Getting alglabel: ",initlabel

    outlabel=""
    if initlabel[0:2]=="AK": outlabel+="anti-k_{T}"
    if initlabel[0:2]=="CA": outlabel+="C/A"

#     if initlabel[4:6]=="LC": outlabel+="_{LC}"
#     if initlabel[4:6]=="TR": outlabel+="_{TR}"

    if initlabel[2:4]=="06": outlabel+="^{R=0.6}"
    if initlabel[2:4]=="08": outlabel+="^{R=0.8}"
    if initlabel[2:4]=="10": outlabel+="^{R=1.0}"
    if initlabel[2:4]=="12": outlabel+="^{R=1.2}"


    if len(initlabel)>6:
        print initlabel[6:]
        if initlabel[6:]=="TRIMF5R20"      : outlabel+=" Trimmed (f=0.5,R=0.2)"
        if initlabel[6:]=="TRIMF5R30"      : outlabel+=" Trimmed (f=0.5,R=0.3)"
        if initlabel[6:]=="PRUNR50Z10"     : outlabel+=" Pruned (R=0.5,Z=0.10)"
        if initlabel[6:]=="PRUNR50Z15"     : outlabel+=" Pruned (R=0.5,Z=0.15)"
        if initlabel[6:]=="BDRSM100R30Y0"  : outlabel+=" BDRS (#mu=100,R=0.5,y=0.00)"
        if initlabel[6:]=="BDRSM100R30Y12" : outlabel+=" BDRS (#mu=100,R=0.5,y=0.12)"
        if initlabel[6:]=="BDRSM100R30Y15" : outlabel+=" BDRS (#mu=100,R=0.5,y=0.15)"
        if initlabel[6:]=="BDRSM100R30Y4"  : outlabel+=" BDRS (#mu=100,R=0.5,y=0.04)"
        if initlabel[6:]=="BDRSM100R30Y9"  : outlabel+=" BDRS (#mu=100,R=0.5,y=0.09)"
    else:
        outlabel+=" Ungroomed"

    print outlabel
    return outlabel

def FindOptimalGraph(graphs, targetseff):
    '''Find optimal choice from files'''
    print "getting optimal graph"
    bgrej=1.0
    maxBGRejection=0.0
    maxSigEff=0.0
    pathChoice=""
    for graph in graphs:
        tempgr=graph[4]

        seffMax, bgrejMax = GetBGRej(tempgr, targetseff)

        #print graph[0],seffMax,bgrejMax

        if bgrejMax>maxBGRejection:
            maxBGRejection = bgrejMax
            maxSigEff      = seffMax
            pathChoice     = graph[0]
            #print pathChoice,"  ",maxBGRejection



    #print "MaxFound:  ",pathChoice,"  ",maxBGRejection

    return pathChoice,maxSigEff,maxBGRejection

def FindOptimalFinalGraph(graphs, targetseff):
    '''Find optimal choice from files'''
    print "getting optimal graph"
    bgrej=1.0
    maxBGRejection=0.0
    maxSigEff=0.0
    pathChoice=""

    nbest=0
    i=0
    for graph in graphs:
        tempgr=graph

        seffMax, bgrejMax = GetBGRej(tempgr, targetseff)

        #print graph[0],seffMax,bgrejMax

        if bgrejMax>maxBGRejection:
            maxBGRejection = bgrejMax
            maxSigEff      = seffMax
            nbest     = i
            #print pathChoice,"  ",maxBGRejection



    #print "MaxFound:  ",pathChoice,"  ",maxBGRejection

    return nbest,maxSigEff,maxBGRejection

def GetBGRej( gr, targetseff ):

    n = gr.GetN();

    xinit       = Double()
    yinit       = Double()
    mindistance = 10.0
    foundseff   = 1.0
    foundbgeff=1.0
    itarget=1

    for i in range(n):
        gr.GetPoint(i,xinit,yinit)
        distance = abs(targetseff-xinit)
        #print "FindBR: ",i," ",xinit," ",yinit," ",distance,"   -   ",foundseff,"  ",foundbgeff


        #print distance,"   -  ",mindistance
        if distance<mindistance:
            #print "SWITCH"
            mindistance = distance
            #Tricky - need to cast them as floats or pyroot treats them as pointer references
            foundseff   = float(xinit)
            foundbgeff  = float(yinit)
            itarget     = i



    #print itarget,"  ",mindistance,"  ",foundseff,"  ",foundbgeff

    return foundseff,foundbgeff


def GetBGRejAndCut( gr, targetseff ):

    n = gr.GetN();

    xinit       = Double()
    yinit       = Double()
    mindistance = 10.0
    foundseff   = 1.0
    foundbgeff=1.0
    itarget=1

    for i in range(n):
        gr.GetPoint(i,xinit,yinit)
        distance = abs(targetseff-xinit)
        #print "FindBR: ",i," ",xinit," ",yinit," ",distance,"   -   ",foundseff,"  ",foundbgeff


        #print distance,"   -  ",mindistance
        if distance<mindistance:
            #print "SWITCH"
            mindistance = distance
            #Tricky - need to cast them as floats or pyroot treats them as pointer references
            foundseff   = float(xinit)
            foundbgeff  = float(yinit)
            itarget     = i



    #print itarget,"  ",mindistance,"  ",foundseff,"  ",foundbgeff

    return foundseff,foundbgeff


def GetHist1D(file, var, range, weight):
    print "Drawing: ",file,var
    ftemp    = TFile(file)
    ntuptemp = ftemp.Get("NTupVTag")
    ntuptemp.Draw(var+">>htemp("+range+")",weight,"e")
    htemp = gDirectory.Get("htemp")
    htemp.SetDirectory(0)
    
    #print htemp.Integral()

    ftemp.Close()
    return htemp


def GetHist2D(file, varx, vary, rangex, rangey, weight):
    print "Getting 2D Hist"
    print file, varx, vary, rangex, rangey, weight
    ftemp    = TFile(file)
    ntuptemp = ftemp.Get("NTupVTag")

    varstring = vary+":"+varx+">>htemp("+rangex+","+rangey+")"
    #print varstring

    ntuptemp.Draw(varstring,weight,"e")
    htemp = gDirectory.Get("htemp")

    htemp.SetDirectory(0)

    ftemp.Close()
    return htemp

def NormalizeHist( hist, norm=1.0):

    tot = hist.Integral()

    if tot<=0:
        print "Something is wrong with histogram"
        return hist
    else:
        hist.Scale(norm/tot)
        return hist

def GetMaxVal(bgh, sh):

    max = bgh.GetMaximum()

    if bgh.GetMaximum() > sh.GetMaximum():
        max = bgh.GetMaximum()
    else:
        max = sh.GetMaximum()

    print "MaxVal = ",max

    return max



def MinWindow( histo, frac):

  minWidth = 100000.0
  topEdge = 0.0
  
  histo.Scale(1.0/histo.Integral());

  Nbins = histo.GetNbinsX()

  for i in range(Nbins):
       tempFrac = 0.0
       imax = i

       while tempFrac<frac and imax != Nbins:
          #print "InCond: ",imax,"  ",tempFrac,"  ",imax
          #print histo.GetBinContent(imax),histo.Integral()
          tempFrac += histo.GetBinContent(imax)/histo.Integral()
          imax += 1
       #print i,"  ",tempFrac,"  ",imax
       
       width = histo.GetBinCenter(imax) - histo.GetBinCenter(i)

       top_edge = histo.GetBinCenter(imax)

       if imax != Nbins and width<minWidth:
          minWidth = width
          topEdge  = top_edge
          
       
        
  maxc = 0.0001
  bin_maxc=0
  Nbins = histo.GetNbinsX()
  for i in range(Nbins):
     tc = histo.GetBinContent(i)
     if tc > maxc:
        maxc=tc
        bin_maxc=i
        
  MPV = histo.GetBinCenter(bin_maxc);
  
  print "vals = ",MPV,minWidth,topEdge
  
  return MPV,minWidth,topEdge

def JetResponse(InputDir, alg, alglabel, outputdir, debug=0):
    
    fg1 = TF1("fg1","gaus",0.8,1.2)
    fg2 = TF1("fg2","gaus",0.8,1.2)
    
    #groomed TR
    algTruth=alg.replace("LC","TR");

    print "alg: ",alg
    print "algTruth: ",algTruth

    estr=alg+"_e";
    estrTruth=algTruth+"_e";
 
    etastr=alg+"_eta";
    
    cutlow="200"
    cuthigh="1000"

    cutlows  = ["200","350","500"]
    cuthighs = ["350","500","1000"]

    for cutlow,cuthigh in zip(cutlows,cuthighs):


        weight="weight_mc*weight_pu*PtReweighting(CA12TRLEAD_pt)*("+alg+"_TruthRecoMatch)*("+alg+"_eta>-1.2)*("+alg+"_eta<1.2)*("+estrTruth+">"+cutlow+")*("+estrTruth+"<"+cuthigh+")";
        print "Weight: ",weight

        mcal = TH1F();
        ecal = TH1F();

        mcal.Fill("e low",float(cutlow))
        mcal.Fill("e high",float(cuthigh))

        ecal.Fill("e low",float(cutlow))
        ecal.Fill("e high",float(cuthigh))

        #=======================
        #SIGNAL
        #=======================
    
        c = TCanvas("c","c",400,400);
    
        csig = TCanvas("csig","csig",900,300);
        csig.Divide(3,1);
        csig1 = csig.cd(1);
        csig2 = csig.cd(2);
        csig3 = csig.cd(3);

        #ereco average
        # used in the numerical inversion procedure
        print "ESig : ",estr
        hsige = GetHist1D(InputDir+"merged.LumpedSignal.root", estr, "1000,0,2000", weight+"*"+GWW)
        ereco = hsige.GetMean();
        print "EReco=",ereco
        mcal.Fill("sigereco",ereco);
        ecal.Fill("sigereco",ereco);


        #dR
        print "DR MATCHING TEST"
        csig1.cd();
        csig1.SetRightMargin(0.2);
        csig1.SetLogz();

        print alg,algTruth
        hsig = GetHist2D(InputDir+"merged.LumpedSignal.root", "("+alg+"_eta-"+algTruth+"_eta)", "("+alg+"_phi-"+algTruth+"_phi)", "40,-5,5", "40,-6,6", weight)
        hsig.GetXaxis().SetTitle("#Delta #eta (true-reco)");
        hsig.GetYaxis().SetTitle("#Delta #phi (true-reco)");
        hsig.Draw("colz");

        #cal1
        csig2.cd();
        var="("+alg+"_m)/("+algTruth+"_m)";
        print var
    
        hsigb = GetHist1D(InputDir+"merged.LumpedSignal.root", var, "50,0,2", weight+"*"+GWW)

        hsigb.Scale(1.0/hsigb.Integral());
        hsigb.GetXaxis().SetTitle("M_{groomed}^{reco}/M_{groomed}^{truth} [GeV]");
        hsigb.GetYaxis().SetTitle("Normalized Entries");
        hsigb.SetMaximum(hsigb.GetMaximum()*1.5);
        hsigb.Draw("hist");
        hsigb.Fit(fg1,"R+");
        chi2=fg1.GetChisquare();
        p1=fg1.GetParameter(1);
        str1="Mean(fit) = "+str(round(p1,3));
        p2=fg1.GetParameter(2);
        str2="Sigma(fit) = "+str(round(p2,3));
        p1b=hsigb.GetMean();
        str1b="Mean(hist) = "+str(round(p1b,3));
        p2b=hsigb.GetRMS();
        str2b="Sigma(hist) = "+str(round(p2b,3));
    
        mcal.Fill("sigmean",p1);
        mcal.Fill("sigres",p2);

        hsigb.Draw("hist");
        fg1.SetLineColor(2);
        fg1.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "W jets")
        myText(       0.20,0.70,1,0.03, "Mass Response")

    
        c.cd()
        hsigb.Draw("hist");
        fg1.SetLineColor(2);
        fg1.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "W jets")
        myText(       0.20,0.70,1,0.03, "Mass Response")
        c.SaveAs(outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+"_mresp_signal.eps");




        #cal2
        csig3.cd();
        var=estr+"/"+estrTruth;
        print var

        hsigb2 = GetHist1D(InputDir+"merged.LumpedSignal.root", var, "50,0.5,1.5", weight+"*"+GWW)

        hsigb2.Scale(1.0/hsigb2.Integral());
        hsigb2.GetXaxis().SetTitle("E_{groomed}^{reco}/E_{groomed}^{truth} [GeV]");
        hsigb2.GetYaxis().SetTitle("Normalized Entries");
        hsigb2.SetMaximum(hsigb.GetMaximum()*1.5);
        hsigb2.Draw("hist");
        hsigb2.Fit(fg2,"R+");
        chi2=fg2.GetChisquare();
        p1=fg2.GetParameter(1);
        str1="Mean(fit) = "+str(round(p1,3));
        p2=fg2.GetParameter(2);
        str2="Sigma(fit) = "+str(round(p2,3));
        p1b=hsigb2.GetMean();
        str1b="Mean(hist) = "+str(round(p1b,3));
        p2b=hsigb2.GetRMS();
        str2b="Sigma(hist) = "+str(round(p2b,3));
        ecal.Fill("sigmean",p1);
        ecal.Fill("sigres",p2);

        hsigb2.Draw("hist");
        fg2.SetLineColor(2);
        fg2.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "W jets")
        myText(       0.20,0.70,1,0.03, "Energy Response")

        c.cd()
        hsigb2.Draw("hist");
        fg2.SetLineColor(2);
        fg2.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "W jets")
        myText(       0.20,0.70,1,0.03, "Energy Response")
        c.SaveAs(outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+"_eresp_signal.eps");

        csig.SaveAs(outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+"_signal.eps");




        #=======================
        #BACKGROUND
        #=======================
    
        weight="weight_mc*weight_pu*("+alg+"_TruthRecoMatch)*("+alg+"_eta>-1.2)*("+alg+"_eta<1.2)*("+estrTruth+">"+cutlow+")*("+estrTruth+"<"+cuthigh+")";
    
        cbg = TCanvas("cbg","cbg",900,300);
        cbg.Divide(3,1);
        cbg1 = cbg.cd(1);
        cbg2 = cbg.cd(2);
        cbg3 = cbg.cd(3);



        #ereco average
        hbg1e = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", estr, "1000,0,2000", weight+"*"+JX3)
        hbg2e = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", estr, "1000,0,2000", weight+"*"+JX4)
        hbg3e = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", estr, "1000,0,2000", weight+"*"+JX5)
        hbg4e = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", estr, "1000,0,2000", weight+"*"+JX6)
        hbg1e.Add(hbg2e);
        hbg1e.Add(hbg3e);
        hbg1e.Add(hbg4e);
        ereco=hbg1e.GetMean();
        mcal.Fill("bgereco",ereco);
        ecal.Fill("bgereco",ereco);



        #dR
        cbg1.cd();
        cbg1.SetRightMargin(0.2);
        cbg1.SetLogz();

        hbg1a = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", "("+alg+"_eta-"+algTruth+"_eta)", "("+alg+"_phi-"+algTruth+"_phi)", "40,-5,5", "40,-6,6", weight+"*"+JX3)
        hbg2a = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", "("+alg+"_eta-"+algTruth+"_eta)", "("+alg+"_phi-"+algTruth+"_phi)", "40,-5,5", "40,-6,6", weight+"*"+JX4)
        hbg3a = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", "("+alg+"_eta-"+algTruth+"_eta)", "("+alg+"_phi-"+algTruth+"_phi)", "40,-5,5", "40,-6,6", weight+"*"+JX5)
        hbg4a = GetHist2D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", "("+alg+"_eta-"+algTruth+"_eta)", "("+alg+"_phi-"+algTruth+"_phi)", "40,-5,5", "40,-6,6", weight+"*"+JX6)
        hbg1a.Add(hbg2a);
        hbg1a.Add(hbg3a);
        hbg1a.Add(hbg4a);

        hbg1a.GetXaxis().SetTitle("#Delta #eta (true-reco)");
        hbg1a.GetYaxis().SetTitle("#Delta #phi (true-reco)");
        hbg1a.Draw("colz");


        #cal1
        cbg2.cd();
        var="("+alg+"_m)/("+algTruth+"_m)";
        print var
    
        hbg1d = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", var, "50,0,2", weight+"*"+JX3)
        hbg2d = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", var, "50,0,2", weight+"*"+JX4)
        hbg3d = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", var, "50,0,2", weight+"*"+JX5)
        hbg4d = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", var, "50,0,2", weight+"*"+JX6)
        hbg1d.Add(hbg2d);
        hbg1d.Add(hbg3d);
        hbg1d.Add(hbg4d);

        hbg1d.Scale(1.0/hbg1d.Integral());
        hbg1d.GetXaxis().SetTitle("M_{groomed}^{reco}/M_{groomed}^{truth} [GeV]");
        hbg1d.GetYaxis().SetTitle("Normalized Entries");
        hbg1d.SetMaximum(hbg1d.GetMaximum()*1.5);
        hbg1d.Draw("hist");
        hbg1d.Fit(fg1,"R+");
        chi2=fg1.GetChisquare();
        p1=fg1.GetParameter(1);
        str1="Mean(fit) = "+str(round(p1,3));
        p2=fg1.GetParameter(2);
        str2="Sigma(fit) = "+str(round(p2,3));
        p1b=hsigb.GetMean();
        str1b="Mean(hist) = "+str(round(p1b,3));
        p2b=hsigb.GetRMS();
        str2b="Sigma(hist) = "+str(round(p2b,3));
        mcal.Fill("bgmean",p1);
        mcal.Fill("bgres",p2);


        hbg1d.Draw("hist");
        fg1.SetLineColor(2);
        fg1.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "QCD jets")
        myText(       0.20,0.70,1,0.03, "Mass Response")

    
        c.cd()
        hbg1d.Draw("hist");
        fg1.SetLineColor(2);
        fg1.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "QCD jets")
        myText(       0.20,0.70,1,0.03, "Mass Response")
        c.SaveAs(outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+"_mresp_qcd.eps");



        #cal2
        cbg3.cd();
        var="("+alg+"_e)/("+algTruth+"_e)";
        print var
    
        hbg1d2 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ3W.root", var, "50,0,2", weight+"*"+JX3)
        hbg2d2 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ4W.root", var, "50,0,2", weight+"*"+JX4)
        hbg3d2 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ5W.root", var, "50,0,2", weight+"*"+JX5)
        hbg4d2 = GetHist1D(InputDir+"merged.Pythia8_AU2CT10_jetjet_JZ6W.root", var, "50,0,2", weight+"*"+JX6)
        hbg1d2.Add(hbg2d2);
        hbg1d2.Add(hbg3d2);
        hbg1d2.Add(hbg4d2);

        hbg1d2.Scale(1.0/hbg1d2.Integral());
        hbg1d2.GetXaxis().SetTitle("E_{groomed}^{reco}/E_{groomed}^{truth} [GeV]");
        hbg1d2.GetYaxis().SetTitle("Normalized Entries");
        hbg1d2.SetMaximum(hbg1d2.GetMaximum()*1.5);
        hbg1d2.Draw("hist");
        hbg1d2.Fit(fg2,"R+");
        chi2=fg2.GetChisquare();
        p1=fg2.GetParameter(1);
        str1="Mean(fit) = "+str(round(p1,3));
        p2=fg2.GetParameter(2);
        str2="Sigma(fit) = "+str(round(p2,3));
        p1b=hsigb.GetMean();
        str1b="Mean(hist) = "+str(round(p1b,3));
        p2b=hsigb.GetRMS();
        str2b="Sigma(hist) = "+str(round(p2b,3));
        ecal.Fill("bgmean",p1);
        ecal.Fill("bgres",p2);


        hbg1d2.Draw("hist");
        fg2.SetLineColor(2);
        fg2.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "QCD jets")
        myText(       0.20,0.70,1,0.03, "Energy Response")

    
        c.cd()
        hbg1d2.Draw("hist");
        fg2.SetLineColor(2);
        fg2.Draw("same");
        ATLASLabel(   0.20,0.90,1,0.1,0.03,"Internal")
        myText(       0.20,0.85,1,0.03,"#sqrt{s}=8 TeV")
        myText(       0.65,0.90,1,0.03, alg)
        myText(       0.65,0.85,1,0.03,"E^{true} = ["+cutlow+","+cuthigh+"] GeV");
        myText(       0.65,0.80,1,0.03, str1)
        myText(       0.65,0.75,1,0.03, str2)
        myText(       0.65,0.70,1,0.03, str1b)
        myText(       0.65,0.65,1,0.03, str2b)
        myLineBoxText( 0.25,0.80,0.05,2,1,3004,0.03,"Gaussian Fit");
        myText(       0.20,0.75,1,0.03, "QCD jets")
        myText(       0.20,0.70,1,0.03, "Energy Response")
        c.SaveAs(outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+"_eresp_qcd.eps");

        cbg.SaveAs(outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+"_qcd.eps");



        fname=outputdir+"Calibratability_"+alg+"_e"+cutlow+cuthigh+".root";
        fout = TFile(fname,"RECREATE");
        mcal.Write("mcal");
        ecal.Write("ecal");
        fout.Close();

def RejToPower( rej = 0.0):
    if rej==1.0:
        print "This is PERFECT with 100% rejection ... check this"
        power = -4
    else:
        power = 1.0/(1.0-rej)
    
    return power
    