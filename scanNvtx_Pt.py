import subprocess
import sys
#pt_values = [[250,500],[500,750],[750,1000],[1000,1250],[1250,1500],[1500,1750],[1750,2000],[200,350],[350,500],[500,1000]]
#pt_values = [[200,350],[350,500],[500,1000]]
pt_values = [[350,500]]
#nvtx_cuts = [10,15,20,25,30]
#nvtx_cuts = [[0,15],[15,25],[25,99]]
nvtx_cuts = [[0,99]]

# run 14 TeV

for pt,nvtx in [(pt,nvtx) for pt in pt_values for nvtx in nvtx_cuts]:
    status = subprocess.call('python Tagger.py config/filter_mu100_cut4_14tev.xml -i /home/tim/boosted_samples/BoostedBosonMerging/sf_cuts_v3/TopoSplitFilteredMu100SmallR30YCut4ca12_14tev_mu100_cut4_v1/ -f 14tev_'+ str(pt[0]) +'_' +str(pt[1])+ '_vxp_'+str(nvtx[0])+ '_'+str(nvtx[1])+' --pthigh='+str(pt[1])+' --ptlow='+str(pt[0]) + ' --nvtx='+str(nvtx[1]) +' --nvtxlow='+str(nvtx[0]), shell=True)

sys.exit(0)

# run 8 TeV lily
for pt,nvtx in [(pt,nvtx) for pt in pt_values for nvtx in nvtx_cuts]:
    status = subprocess.call('python Tagger.py config/filter_mu100_cut4_8tev_l.xml -i /home/tim/boosted_samples/BoostedBosonMerging/sf_cuts_v3/TopoSplitFilteredMu100SmallR30YCut4lily_mu100_cut4_v1/ -f 8tev_l_'+ str(pt[0]) +'_' +str(pt[1])+ '_vxp_'+str(nvtx[0])+ '_'+str(nvtx[1])+' --pthigh='+str(pt[1])+' --ptlow='+str(pt[0]) + ' --nvtx='+str(nvtx[1])+' --nvtxlow='+str(nvtx[0]), shell=True)


# run Note 8 TeV
for pt,nvtx in [(pt,nvtx) for pt in pt_values for nvtx in nvtx_cuts]:
    status = subprocess.call('python Tagger.py config/filter_mu100_cut4_8tev.xml -i /home/tim/boosted_samples/BoostedBosonMerging/sf_cuts_v3/TopoSplitFilteredMu100SmallR30YCut4note_8TeV_mu100_cut4_v1/ -f 8tev_'+ str(pt[0]) +'_' +str(pt[1])+ '_vxp_'+str(nvtx[0])+ '_'+str(nvtx[1])+' --pthigh='+str(pt[1])+' --ptlow='+str(pt[0]) + ' --nvtx='+str(nvtx[1])+' --nvtxlow='+str(nvtx[0]), shell=True)

