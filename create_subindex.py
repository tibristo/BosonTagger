import copy
from os import listdir, getcwd
from os.path import isdir, isfile, join


f = [x for x in listdir('.') if isdir(x)]#open('folders.txt','r')
onlyfiles = [ x for x in listdir('.') if isfile(x) ]
if 'index.html' in onlyfiles:
    onlyfiles.remove('index.html')
openTR = '<TR>\n'
closeTR = '</TR>\n'
print f
currdir = getcwd()

algorithm = currdir[currdir.rfind('/')+1:]
tag = algorithm[:algorithm.find('_')]
spl = algorithm.split('_')
print spl
# last two are the pt range, first one is beginning of the thing
sample = ''
for i, s in enumerate(spl):
    if i != 0 and i < len(spl)-2:
        sample+=s
        if i < len(spl)-3:
            sample+='_'
roc = algorithm+'-Tim2-ROCPlot.png'


idx_start = '<html>\n<head>\n<title>'

idx_part2 = '</title>\n</head>\n<body>\n<p>\n<img border="0" src="https://webservices.web.cern.ch/webservices/Images/cernlogo.jpg" alt="CERN Logo" align="bottom">\n<b><font size="6">&nbsp;&nbsp;&nbsp;&nbsp; Hbb Studies</font>\n</p>\n<hr>\n<font size="3">Back: </font><a href="https://dfs.cern.ch/dfs/Websites/t/tibristo/'#20140423_NTup_v1.0'
idx_part2 += sample+'/'+tag+'/'+algorithm
#idx_part2 += getcwd().split('/')[-1]
idx_part2 +='/'+sample+'.html">Study Home</a>\n<hr>\n<font size="3">\n'

idx_part2 += '<TABLE>\n<TR>\n<a href="../'+roc+'"><H3><font color="#0000FF">ROC Curves</font></H3></a>\n</TR>\n</TABLE>\n'


idx_part2 += '<H1><font color=\"#0000FF\">  </font></H1>\n<TABLE>\n'

idx_end = '<TR>\n</TR>\n</TABLE>\n<hr>\n<p>Site created by Tim Bristow \n</p>\n</body>\n</html>\n'

print algorithm

for l in onlyfiles:
    this_idx = idx_start
    #newidx = open(l.strip()+'/index.html','w')
    newidx = open('index.html','w')

    #onlyfiles.remove('index.html')
    this_idx+=algorithm
    this_idx+=idx_part2
    this_idx+='<TABLE>\n'
    this_idx+='<h2>\n'+algorithm+'\n</h2>\n'
    logv = False
    tr_is_open = False
    prev_var = ''
    for j in onlyfiles:
        if j.find('png') == -1:
            continue
        varname = j[0:j.find('.')]
        this_idx+=openTR
        this_idx+='<td align=\"center\">\n'
        this_idx+='<a href=\"'
        this_idx+=j
        this_idx+='\"> <img src=\"'+j+'\"></a>'
        this_idx+='<br> '+varname+' <br><a href=\"'+j+'\">[png]</a> <a href=\"'+varname+'.eps\">[eps]</a>\n'
        this_idx+='</td>\n'
        this_idx += closeTR

    this_idx+=idx_end
    newidx.write(this_idx)
    newidx.close()
#f.close()
