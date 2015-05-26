#from ROOT import * 
from functions import setweights
from numpy import *
import root_numpy
from SignalPtWeight import SignalPtWeight
import pandas as pd

InputDir = "/Users/wbhimji/Data/BosonTaggingMVA_ForJoeGregory/Data/v4/"
treename = 'NTupLepW'

trees,files,weights = ( {} for i in range(3) ) 

setweights(weights)

#Algorithm = 'CA12LCSF67'
Algorithm = 'AKT10LCTRIM530'
branches = ['weight_mc', 'CA12Truth_pt', 'CA12Truth_eta']
branchstubs = ['_m','_SPLIT12','_MassDropSplit','_Tau2Tau1','_ys12','_PlanarFlow']
branches.extend([Algorithm + branch for branch in branchstubs]) 

data={}
for index,typename in enumerate(weights):
    if typename == 'signal':
        filename = InputDir + typename + ".root"
        data[typename] = root_numpy.root2array(filename,treename,branches,'(CA12Truth_eta > -1.2) *( CA12Truth_eta < 1.2) * (CA12Truth_pt > 200) * (CA12Truth_pt < 350))')
        data[typename] = pd.DataFrame(data[typename])
        data[typename]['label']=1
        data[typename]['weight']=weights[typename]*data[typename]['weight_mc']
    else:
        filename = InputDir + "Sherpa_CT10_WmunuMassiveCBPt" + typename + ".root"
        data[typename] = root_numpy.root2array(filename,treename,branches,'(CA12Truth_eta > -1.2) *( CA12Truth_eta < 1.2) * (CA12Truth_pt > 200) * (CA12Truth_pt < 350))')
        data[typename] = pd.DataFrame(data[typename])
        data[typename]['label']=0
        data[typename]['weight']=weights[typename]*data[typename]['weight_mc']
        #    print data[typename] 
    if index==0:
        data[typename].to_csv('Merged.csv')
    else:
        data[typename].to_csv('Merged.csv',mode='a',header=False)
