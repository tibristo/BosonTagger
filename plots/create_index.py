import sys
idx_start='<html>\n<head>\n<title>Boosted boson plots</title>\n<!-- Latest compiled and minified CSS -->\n<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">\n<!-- Latest compiled and minified JavaScript -->\n<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>\n<script src="//code.jquery.com/jquery-1.11.2.min.js"></script>\n<script src="//code.jquery.com/jquery-migrate-1.2.1.min.js"></script>\n</head>\n<body>\n<p>\n<img border="0" src="https://webservices.web.cern.ch/webservices/Images/cernlogo.jpg" alt="CERN Logo" align="bottom">\n<b><font size="6">&nbsp;&nbsp;&nbsp;&nbsp; Studies</font>\n</p>\n<hr>\n<font size="3">Back: </font><a href="http://tibristo.web.cern.ch/tibristo/">Home</a>\n<hr>\n<font size="3">\n'

from os import listdir
from os.path import isdir, isfile

from operator import itemgetter

f_name = '.'
sample = sys.argv[1]

f = [x for x in listdir('.') if isdir(x) and x.find(sample)!=-1]

# sort alphabetically
f = sorted(f)

# sort the pt ranges
tosort = []
for i,x in enumerate (f):
    spl = x.split('_')
    tosort.append([x, spl[0], int(spl[-2])])

sorted_pt = sorted(tosort, key=itemgetter(1,2))

sorted_list = []
for p in sorted_pt:
    sorted_list.append(p[0])

#print f

#f = open('folders.txt','r')
outf = open(sample+'.html','w')
count = 0
idx = idx_start
title = 'Boosted boson study - ' +sample + ' sample'
idx+='<h2>\n'+title+'\n</h2>\n'
idx+='<h3>\n plots \n</h3>\n'
tag = './'
algorithmsAdded = []

for l in sorted_list:
    if True:

        ll = l.strip().replace("/","")
        spl = ll.strip().split('_')
        algname = spl[0]#ll.replace(sample+"_","")

        refname = algname+ ' ' + spl[-2] + ' - ' + spl[-1] + ' GeV'
        algorithmDirectory = l.strip().split('_')[0]
        
        if not algorithmDirectory in algorithmsAdded:
            idx+='<h3>\n'+algorithmDirectory+'\n</h3>\n'
            algorithmsAdded.append(algorithmDirectory)
        idx+='<p>\n<a href=\"'+f_name+'/'+algorithmDirectory+'/'+ll+'/index.html\">'+refname+'</a>\n</p>\n'
    else:

        count+=1

idx+='<h3>\n Background rejection \n</h3>\n'
rej = [x for x in listdir('.') if isfile(x) and x.find(sample)!=-1 and x.startswith('matrixinv')]

# sort by pt
tosortrej = []
for i,x in enumerate (rej):
    spl = x.split('_')
    tosortrej.append([x, int(spl[-2])])

sorted_rejtmp = sorted(tosortrej, key=itemgetter(1))

sorted_rej = []
for p in sorted_rejtmp:
    sorted_rej.append(p[0])



for r in sorted_rej:
    spl = r.split('_')
    
    if len(spl) > 2:
        rejname = spl[-2] + ' - '+ spl[-1].replace('.png','') +' GeV'
    else:
        rejname = 'background rejection: ' + r
    idx+='<p>\n<a href=\"'+r.strip()+'\">'+rejname+'</a>\n</p>\n'

idx+='<hr>\n<p>Site created by Tim Bristow \n</p>\n</body>\n</html>\n'
outf.write(idx)
outf.close()
#f.close()
