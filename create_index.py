import sys
idx_start='<html>\n<head>\n<title>Boosted boson plots</title>\n</head>\n<body>\n<p>\n<img border="0" src="https://webservices.web.cern.ch/webservices/Images/cernlogo.jpg" alt="CERN Logo" align="bottom">\n<b><font size="6">&nbsp;&nbsp;&nbsp;&nbsp; Studies</font>\n</p>\n<hr>\n<font size="3">Back: </font><a href="http://tibristo.web.cern.ch/tibristo/">Home</a>\n<hr>\n<font size="3">\n'

f_name = '.'#20140423_NTup_v1.0'#'20140221_MET_20_Study_v0.0'
from os import listdir
from os.path import isdir

sample = sys.argv[1]

f = [x for x in listdir('.') if isdir(x) and x.find(sample)!=-1]

#f = open('folders.txt','r')
outf = open(sample+'.html','w')
count = 0
idx = idx_start
title = 'Boosted boson study - ' +sample + ' sample'
idx+='<h2>\n'+title+'\n</h2>\n'
idx+='<h3>\n plots \n</h3>\n'
tag = './'
for l in f:
    if True:
        ll = l.strip().replace("/","")
        idx+='<p>\n<a href=\"'+f_name+'/'+ll+'/index.html\">'+ll+'</a>\n</p>\n'
    else:

        count+=1

idx+='<h3>\n Background rejection \n</h3>\n'
rej = [x for x in listdir('.') if isfile(x) and x.find(sample)!=-1 and x.startswith('matrixinv')]
for r in rej:
    idx+='<p>\n<a href=\"'+r.strip()+'\">'+ll+'</a>\n</p>\n'

idx+='<hr>\n<p>Site created by Tim Bristow \n</p>\n</body>\n</html>\n'
outf.write(idx)
outf.close()
#f.close()
