import TaggerTim
import cPickle as pickle
import ROOT
import sys
import subprocess
import argparse
import os.path

def main(args):
    parser = argparse.ArgumentParser(description='scan bkg rejection')
    parser.add_argument('-b', '--basedir', help = 'The base directory of the input files')
    parser.add_argument('-f','--files', help = 'Plain text file containing names of folders')
    parser.add_argument('-c', '--configs', help = 'Plain text file containing names of config files')
    parser.add_argument('-v', '--version', help = 'Version to use in pickle file')
    parser.add_argument('-t', '--treename', help = 'Treename in input root files')
    parser.add_argument('-m', '--masswindow', help = 'Apply mass window cuts')

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
        args_tag = ['python','TaggerTim.py',c,'-i',p,'-f', f+'_'+args.version, '--pthigh=1000','--ptlow=500','--nvtx=99','--nvtxlow=0','--ptreweighting=true','--saveplots=true', '-v',args.version] 
        if args.treename:
            args_tag.append('--tree='+args.treename)
        else:
            args_tag.append('--tree=physics')
        if args.masswindow:
            if args.masswindow == 'true' or args.masswindow == 'True':
                args_tag.append('--massWindowCut=True')
        print args_tag
        #raw_input()
        #print sys.argv
        #rej, var, m_min, m_max=#TaggerTim.main([c,'-i',p,'-f', f, '--pthigh=2000','--ptlow=350','--nvtx=99','--nvtxlow=0','--ptreweighting=true','--saveplots=true','--tree=physics','lumi=1'])
        p = subprocess.Popen(args_tag,stdout=subprocess.PIPE)
        p.wait()
        #output = p.communicate()[0]
        # check output pickle exists - if not, raise error, set all values to 0
        rej = '0'
        var = ''
        mass_min = '0'
        mass_max = '10000'

        if os.path.isfile("temp.p"):
            output = pickle.load(open("temp.p","rb"))
            #print output
            #print output.find("MAXREJSTART:")
            rejout= output[output.find("MAXREJSTART:")+12:output.find("MAXREJEND")]
            rejoutsplit = rejout.split(",")
            rej = str(rejoutsplit[0])
            mass_min = str(rejoutsplit[2])
            mass_max = str(rejoutsplit[3])
            var = str(rejoutsplit[1])
        else:
            var = 'ERROR'
    
        print 'Algorithm: ' + c
        print 'Rejection: ' + rej
        print 'Variable: ' + var
        print 'Mass min: ' + mass_min
        print 'Mass max: ' + mass_max
        if float(rej) > maxrej:
            maxrej = float(rej)
            maxrejvar = var
            maxrejm_min = float(mass_min)
            maxrejm_max = float(mass_max)
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
