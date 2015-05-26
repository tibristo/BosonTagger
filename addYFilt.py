
from ROOT import *
import sys
import math
# args: sys.argv[1] = input file, sys.argv[2] = treename, sys.argv[3] = algorithm
filename = sys.argv[1]
treename = sys.argv[2]
algo = sys.argv[3]

algo_prefix = algo[:algo.find('LC')]

f = TFile(filename, "READ")
t = f.Get(treename)


b_groom = std.vector(float)()
b_truth = std.vector(float)()
b_topo = std.vector(float)()

gROOT.ProcessLine("struct event_t{\
Float_t mass;\
Float_t mass_truth;\
Float_t mass_topo;\
Float_t split12;\
Float_t split12_truth;\
Float_t split12_topo;\
}")

event = event_t()

print algo
t.SetBranchAddress('jet_'+algo+'_m',AddressOf(event, "mass"))
t.SetBranchAddress('jet_CamKt12Truth_m',AddressOf(event, "mass_truth"))
t.SetBranchAddress('jet_'+algo_prefix+'LCTopo_m',AddressOf(event, "mass_topo"))

t.SetBranchAddress('jet_'+algo+'_SPLIT12',AddressOf(event, "split12"))
t.SetBranchAddress('jet_CamKt12Truth_SPLIT12',AddressOf(event, "split12_truth"))
t.SetBranchAddress('jet_'+algo_prefix+'LCTopo_SPLIT12',AddressOf(event, "split12_topo"))

f_clone = TFile(filename.replace('g.root','g2.root'),'RECREATE')
t_clone = t.CloneTree(0)

t_clone.Branch('jet_'+algo+'_YFilt',b_groom)
t_clone.Branch('jet_CamKt12Truth_YFilt', b_truth)
t_clone.Branch('jet_'+algo_prefix+'LCTopo_YFilt', b_topo)

nentries = t.GetEntries()

for x in xrange(0,nentries):
    t.GetEntry(x)
    if x%10000 == 0:
        print 'entry ' +str(x) + '/' + str(nentries)
    b_groom.clear()
    b_truth.clear()
    b_topo.clear()
    #print event.mass
    if event.mass != 0:
        yfilt = math.sqrt(event.split12/event.mass)
    else:
        #print '0 mass!'
        yfilt = -999
    b_groom.push_back(yfilt)
    if event.mass_truth != 0:
        yfilt = math.sqrt(event.split12_truth/event.mass_truth)
    else:
        yfilt = -999
    b_truth.push_back(yfilt)
    if event.mass_topo != 0:
        yfilt = math.sqrt(event.split12_topo/event.mass_topo)
    else:
        yfilt = -999
    b_topo.push_back(yfilt)

    t_clone.Fill()

t_clone.Write()

f_clone.Close()
