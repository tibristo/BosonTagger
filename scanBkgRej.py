import TaggerTim
import cPickle as pickle
import ROOT
import sys
import subprocess
import argparse

def main(args):
    parser = argparse.ArgumentParser(description='scan bkg rejection')
    parser.add_argument('-b', '--basedir', help = 'The base directory of the input files')
    parser.add_argument('-f','--files', help = 'Plain text file containing names of folders')
    parser.add_argument('-c', '--configs', help = 'Plain text file containing names of config files')
    parser.add_argument('-v', '--version', help = 'Version to use in pickle file')

    args = parser.parse_args()

    base_dir = args.basedir
    print 'base_dir: ' + base_dir
    paths = []
    pathsf = open(args.files,'r')
    for l in pathsf:
        l = l.strip()
        paths.append(base_dir+l)

    configs = []
    configsf = open(args.configs,'r')
    for l in configsf:
        l = l.strip()    
        configs.append(l)

    fileids = []
    for c in configs:
        # take as a string the part after the final / and remove .xml from the end
        sl_idx = c.rfind('/')+1

        fileids.append(c[sl_idx:-4])

    maxrej = 0
    maxrejvar = ''
    maxrejm_min = 0
    maxrejm_max = 0
    maxalg = ''

    rejectionmatrix = {}

    for p,c,f in zip(paths,configs,fileids):
        #try:
        args_tag = ['python','TaggerTim.py',c,'-i',p,'-f', f, '--pthigh=350','--ptlow=200','--nvtx=99','--nvtxlow=0','--ptreweighting=true','--saveplots=true','--tree=physics', '--massWindowCut=True','-v',args.version]
        print args_tag
        #raw_input()
        #print sys.argv
        #rej, var, m_min, m_max=#TaggerTim.main([c,'-i',p,'-f', f, '--pthigh=2000','--ptlow=350','--nvtx=99','--nvtxlow=0','--ptreweighting=true','--saveplots=true','--tree=physics','lumi=1'])
        p = subprocess.Popen(args_tag,stdout=subprocess.PIPE)
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
        totalrejection = pickle.load(open("tot_rej_"+args.version+".p","rb"))
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
    with open("rejectionmatrix_"+args.version+".p","wb") as f:
        pickle.dump(rejectionmatrix, f)

    import plotCorrelationMatrix as pm
    pm.plotMatrix(args.version)

if __name__ == '__main__':
    main(sys.argv)
