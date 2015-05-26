#from ROOT import * 
from functions import setweights
from numpy import *
import root_numpy
from SignalPtWeight import SignalPtWeight
import pandas as pd
import matplotlib.pyplot as plt

InputDir = "/Users/wbhimji/Data/BoostedBosonNtuples/TopoSplitFilteredMu100SmallR30YCut4ca12_14tev/"
treename = 'physics'

trees,files,weights = ( {} for i in range(3) ) 

setweights(weights)

#Algorithm = 'CA12LCSF67'
Algorithm = 'CamKt12LCTopoSplitFilteredMu100SmallR30YCut4'

branches = ['mc_event_weight', 'jet_CamKt12Truth_pt', 'jet_CamKt12Truth_eta']
plotbranchstubs = ['_m','_Tau1', '_SPLIT12', '_PlanarFlow','_Tau21','_massdrop', '_yt']
branches = ['jet_' + Algorithm + branch for branch in plotbranchstubs]
cutstring = "(jet_CamKt12Truth_pt > 1000000) * (jet_CamKt12Truth_pt < 1500000) * (jet_CamKt12Truth_eta > -1.2) * (jet_CamKt12Truth_eta < 1.2) * (jet_" + Algorithm + "_yt<1.) * (jet_" + Algorithm + "_massdrop<1.)"
data={}
plt.figure()
for typename in ['sig','bkg']:
    print typename
    filename = InputDir + Algorithm.replace('CamKt12LC','') + "_2_inclusive_" + typename  + ".root"

    numpydata = root_numpy.root2array(filename,treename,branches,cutstring)
    numpydata = pd.DataFrame(numpydata)
    numpydata.rename(columns=lambda x: x.replace('jet_' + Algorithm,''), inplace=True)
    if typename == 'sig': 
        numpydata['label']=1 
        numpydata.to_csv(Algorithm+'14TeVmerged.csv')
    else: 
        numpydata['label']=0 
        numpydata.to_csv(Algorithm+'14TeVmerged.csv',mode='a',header=False)
    numpydata.hist(bins=20,grid=False,histtype='step',label=typename)
plt.savefig('Pandaplots.pdf')
