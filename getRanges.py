import sys
from ROOT import *
import os
import functions as fn
from AtlasStyle import *
gROOT.SetBatch(True)
algorithmsdir = sys.argv[1]
algorithmsfile = sys.argv[2]

algorithms = open(algorithmsfile)

variables = ['Aplanarity','ThrustMin','Tau1','Sphericity','m','FoxWolfram20','Tau21','ThrustMaj','EEC_C2_1','pt','EEC_C2_2','Dip12','phi','SPLIT12','TauWTA2TauWTA1','EEC_D2_1','YFilt','Mu12','TauWTA2','Angularity','ZCUT12','Tau2','EEC_D2_2','eta','TauWTA1','PlanarFlow']

pt  = [[200,350],[350,500],[500,1000],[1000,1500],[1500,2000],[2000,3000]]

class minmaxVals():
    pt_slices = {}
    def __init__(self, sample):
        self.pt_slices = {}
        self.sample = sample

    def setMax(self, pt_slice, variable, max_value, min_value):
        if pt_slice in self.pt_slices.keys():
            self.pt_slices[pt_slice][variable] = [max_value, min_value]
        else:
            self.pt_slices[pt_slice] = {}
            self.pt_slices[pt_slice][variable] = [max_value, min_value]

    def getMaxMin(self, pt_slice, variable):
        print self.pt_slices
        return self.pt_slices[pt_slice][variable]
        
SetAtlasStyle()
ROOT.gROOT.LoadMacro("SignalPtWeight3.C")

for a in algorithms:
    a = a.strip().replace('/','')


    filelist = os.listdir(algorithmsdir+'/'+a)
    print filelist
    sigfile = ''
    bkgfile = ''
    massWindowFiles = []
    for f in filelist:
        if f.endswith('sig.root'):
            sigfile = f
        elif f.endswith('bkg.root'):
            bkgfile = f
        elif f.endswith('masswindow.out'):
            massWindowFiles.append(f)

    #tc.Add(algorithmsdir+a+'/'+bkgfile)
    #tc.Add(algorithmsdir+a+'/'+sigfile)

    algorithm = a[:a.find('_')]
    
    c = TCanvas(a)


    # this is sorted
    signalValues = minmaxVals('signal')
    bkgValues = minmaxVals('background')

    # load the pt weighting histograms for this algorithm
    
    loadHistograms(algorithmsdir+a+'/'+sigfile, algorithmsdir+a+'/'+bkgfile)



    for sample in ['sig','bkg']:
        tc = TChain("outputTree")
        
        if sample == 'sig':
            tc.Add(algorithmsdir+a+'/'+sigfile)
        else:
            tc.Add(algorithmsdir+a+'/'+bkgfile)

        for ptrange in pt:
            rangefile = open('ranges/'+a+str(ptrange[0])+'_'+str(ptrange[1])+'_'+sample+'.range','w')

            # get the mass window for this range
            # look in the massWindowFiles list and find the correct pt range.
            massWinFile = ''
            for m in massWindowFiles:
                pt_rng = m[m.find('pt')+3:-len('masswindow.out')-1]
                # the pt range is always split by an underscore
                spl = pt_rng.split('_')
                pt_l = float(spl[0])
                pt_h = float(spl[1])
                # make sure we have the correct pt range mass window file
                if pt_l == float(ptrange[0]) and pt_h == float(ptrange[1]):
                    print 'mass window file: ' +m
                    massWinFile = algorithmsdir+a+'/'+m

            if massWinFile =='':
                print 'can not find the mass window files!!!! Exiting'
                sys.exit()

            mass_max, mass_min = fn.getMassWindow(massWinFile)

            for v in variables:

                cutstring = "(jet_CamKt12Truth_pt > "+str(ptrange[0]*1000)+") * (jet_CamKt12Truth_pt <= "+str(ptrange[1]*1000)+") * (jet_CamKt12Truth_eta > -1.2) * (jet_CamKt12Truth_eta < 1.2) "
                cutstringweight = '*mc_event_weight*1./evt_nEvts'
                if sample == 'bkg': 
                    # no longer apply the k factor
                    
                    cutstringweight += '*evt_filtereff*evt_xsec'#*evt_kfactor'

                elif sample == 'sig':# and ptreweight:
                    # we only apply pt rw now, so reset the cutstring
                    cutstringweight ='*SignalPtWeight3(jet_CamKt12Truth_pt)'

                cutstring = cutstring+ " * (jet_" +algorithm + "_m <= " +str(mass_max)+ ")" + " * (jet_" +algorithm + "_m > " +str(mass_min)+ ") " +cutstringweight

                varname = 'jet_'+algorithm+'_'+v
                histname = 'hist'+varname
                varexp = varname+">>" + histname
                tc.Draw(varexp, cutstring)
                hist = ROOT.gDirectory.Get(histname)
                hist.Draw()
                c.SaveAs("ranges/"+algorithm+'_'+varname+'_'+str(ptrange[0])+'_'+str(ptrange[1])+'_'+sample+".png")
                rangefile.write(varname+'\n')
                if (sample=='sig'):
                    signalValues.setMax(str(ptrange[0]), v, str(hist.GetXaxis().GetXmax()), str(hist.GetXaxis().GetXmin()))
                else:
                    bkgValues.setMax(str(ptrange[0]), v, str(hist.GetXaxis().GetXmax()), str(hist.GetXaxis().GetXmin()))
                rangefile.write('max: ' + str(hist.GetXaxis().GetXmax())+'\n')
                rangefile.write('max: ' + str(hist.GetXaxis().GetXmin())+'\n')
                rangefile.write('*********************************************\n')

    # now that we have the maximum values for the background and signal sep
    # we want to combine the two.  This is done for each pt range.
    rangefilecombined = open('ranges/'+a+'_combined.range','w')
    for ptrange in pt:        

        rangefilecombined.write("\nPT RANGE: " + str(ptrange[0]) + " - " +str(ptrange[1])+'\n')
        rangefilecombined.write('*********************************************\n')
        for v in variables:
            s_max, s_min = signalValues.getMaxMin(str(ptrange[0]), v)
            b_max, b_min = bkgValues.getMaxMin(str(ptrange[0]), v)
            rangefilecombined.write('variable: ' + v+'\n')
            rangefilecombined.write("max: " + str(max(s_max, b_max))+'\n')
            rangefilecombined.write("min: " + str(max(s_min, b_min))+'\n')

        rangefilecombined.write('*********************************************\n')

    rangefilecombined.close()

