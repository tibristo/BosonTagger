import sys
import ROOT
import os
ROOT.gROOT.SetBatch(True)
algorithmsdir = sys.argv[1]
algorithmsfile = sys.argv[2]

algorithms = open(algorithmsfile)

variables = ['Aplanarity','ThrustMin','Tau1','Sphericity','m','FoxWolfram20','Tau21','ThrustMaj','EEC_C2_1','pt','EEC_C2_2','Dip12','phi','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','Angularity','ZCUT12','Tau2','EEC_D2_2','eta','TauWTA1','PlanarFlow']

for a in algorithms:
    a = a.strip().replace('/','')
    rangefile = open('ranges/'+a+'.range','w')
    tc = ROOT.TChain("outputTree")
    filelist = os.listdir(algorithmsdir+'/'+a)
    print filelist
    sigfile = ''
    bkgfile = ''
    for f in filelist:
        if f.endswith('sig.root'):
            sigfile = f
        elif f.endswith('bkg.root'):
            bkgfile = f

    tc.Add(algorithmsdir+a+'/'+bkgfile)
    tc.Add(algorithmsdir+a+'/'+sigfile)

    algorithm = a[:a.find('_')]
    c = ROOT.TCanvas(a)
    for v in variables:

        varname = 'jet_'+algorithm+'_'+v
        histname = 'hist'+varname
        varexp = varname+">>" + histname
        tc.Draw(varexp,'('+varname+'>-998)')
        hist = ROOT.gDirectory.Get(histname)
        hist.Draw()
        c.SaveAs("ranges/"+algorithm+varname+".png")
        rangefile.write(varname+'\n')
        rangefile.write('max: ' + str(hist.GetXaxis().GetXmax())+'\n')
        rangefile.write('max: ' + str(hist.GetXaxis().GetXmin())+'\n')
        rangefile.write('*********************************************\n')
    rangefile.close()

