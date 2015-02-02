from os import listdir, getcwd
from os.path import join,isdir

f = [x for x in listdir('.') if isdir(x)]

outmk = open('mkdir.sh','w')
outidx = open('putidx.sh','w')
outidxm = open('putmidx.sh','w')
outf = open('putfiles.sh','w')

dirs = getcwd().split('/')[-1]+'/'
tag = './'

outmk.write('mkdir '+tag + '/\n'+ 'cd ' + tag+'\n')
outmk.write('mkdir '+dirs + '/\n'+ 'cd ' + dirs+'\n')
outidx.write('cd ' + tag+dirs+'\n')
outidxm.write('cd ' + tag+dirs+'\n')
outidxm.write('mput index.html\n')
outidxm.close()
outf.write('cd ' +tag+ dirs+'\n')

for l in f:
    outmk.write('mkdir '+l.strip()+'/\n')
    outidx.write('mput '+l.strip()+'/index.html\n')
    outf.write('mput '+l.strip()+'/*\n')

outmk.close()
outidx.close()
outf.close()
