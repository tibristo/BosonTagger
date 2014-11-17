import TaggerTim
import cPickle as pickle
import ROOT
import sys
import subprocess

base_dir = '/home/tim/boosted_samples/BoostedBosonMerging/optimisation_v1/'
paths = []
pathsf = open('files.txt','r')
for l in pathsf:
    l = l.strip()
    paths.append(base_dir+l)

configs = []
configsf = open('configs.txt','r')
for l in configsf:
    l = l.strip()
    configs.append(l)

fileids = []
for c in configs:
    l = c[7:-4]
    
    #l = c.replace('config/','').replace('.xml','')
    fileids.append(l)

maxrej = 0
maxrejvar = ''
maxrejm_min = 0
maxrejm_max = 0
maxalg = ''

rejectionmatrix = {}

for p,c,f in zip(paths,configs,fileids):
    #try:
    args = ['python','TaggerTim.py',c,'-i',p,'-f', f, '--pthigh=2000','--ptlow=350','--nvtx=99','--nvtxlow=0','--ptreweighting=true','--saveplots=true','--tree=physics']
    #print sys.argv
    #rej, var, m_min, m_max=#TaggerTim.main([c,'-i',p,'-f', f, '--pthigh=2000','--ptlow=350','--nvtx=99','--nvtxlow=0','--ptreweighting=true','--saveplots=true','--tree=physics','lumi=1'])
    p = subprocess.Popen(args,stdout=subprocess.PIPE)
    output = p.communicate()[0]
    #print output
    #print output.find("MAXREJSTART:")
    rejout= output[output.find("MAXREJSTART:")+12:output.find("MAXREJEND")]
    rejoutsplit = rejout.split(",")
    
    print 'Algorithm: ' + c
    print 'Rejection: ' + str(rejoutsplit[0])
    print 'Variable: ' + rejoutsplit[1]
    print 'Mass min: ' + str(rejoutsplit[2])
    print 'Mass max: ' + str(rejoutsplit[3])
    if float(rejoutsplit[0]) > maxrej:
        maxrej = float(rejoutsplit[0])
        maxrejvar = rejoutsplit[1]
        maxrejm_min = float(rejoutsplit[2])
        maxrejm_max = float(rejoutsplit[3])
        maxalg = c
    # load total background rejection matrix from pickle file
    totalrejection = pickle.load(open("tot_rej.p","rb"))
    rejectionmatrix[f] = totalrejection
    #except e:
     #   print 'Failed to analyse: ' + f
      #  print e

print '----------------------------------'
print 'Algorithm: ' + maxalg
print 'Rejection: ' + str(maxrej)
print 'Variable: ' + maxrejvar
print 'Mass min: ' + str(maxrejm_min)
print 'Mass max: ' + str(maxrejm_max)


# pickle output in case of crash
with open("rejectionmatrix.p","wb") as f:
    pickle.dump(rejectionmatrix, f)

import plotCorrelationMatrix as pm
pm.plotMatrix()

