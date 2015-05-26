import copy
from os import listdir, getcwd
from os.path import isdir, isfile, join
from operator import itemgetter

# get all of the files from the directory, but don't include the noMW files
onlyfiles = [ x for x in listdir('.') if isfile(x) and x.find('noMW') == -1]
# remove the index file from the list
if 'index.html' in onlyfiles:
    onlyfiles.remove('index.html')

# sort the files by variable name, where generally we have algorithm_variablename.png
# this list has [filename, variablename]
listtosort = []
for x in onlyfiles:
    spl = x.split('_')
    # sometimes there is just variablename.png
    if len(spl) > 1:
        key = spl[-1].replace('.png','')#remove the file extension
    else:
        key = spl[0]
    listtosort.append([x,key])

# now sort on the key
sortedlist = sorted(listtosort, key=itemgetter(1))
# get the filenames without the key
sortedfiles = [x[0] for x in sortedlist]

# variables for opening and closing each row
openRow = '<div class=\"row\">\n'
closeRow = '</div>\n'

currdir = getcwd() # current directory

# the algorithm name is in the directory name
algorithm = currdir[currdir.rfind('/')+1:]
# tag refers to everything after the algorithm name
tag = algorithm[:algorithm.find('_')]
# variable names and algorithm name are sep by _
spl = algorithm.split('_')
print spl
# last two are the pt range, first one is beginning of the name of the sample
sample = ''
for i, s in enumerate(spl):
    if i != 0 and i < len(spl)-2:
        sample+=s
        if i < len(spl)-3:
            sample+='_'
# ROC curve filename
roc = algorithm+'-Tim2-ROCPlot.png'

# setup the html
idx_init = '<html>\n<head>\n<title>TITLEVAR'#replace titlevar later
idx_init += '</title>\n'
# bootstrap info so we can use the stylez
idx_init += '<!-- Latest compiled and minified CSS -->\n<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">\n<!-- Latest compiled and minified JavaScript -->\n<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>\n<script src="//code.jquery.com/jquery-1.11.2.min.js"></script>\n<script src="//code.jquery.com/jquery-migrate-1.2.1.min.js"></script>\n</head>\n<body>\n'
idx_init += '<div class=\"container\">\n'
idx_init += '<div class=\"row\">\n'
idx_init += '<div class=\"col-lg-12\">\n'
idx_init += '<p>\n<img border="0" src="https://webservices.web.cern.ch/webservices/Images/cernlogo.jpg" alt="CERN Logo" align="bottom">\n<b><font size="6">&nbsp;&nbsp;&nbsp;&nbsp; Boosted Boson Studies</font>\n</p>\n<hr>\n<font size="3">Back: </font><a href="https://dfs.cern.ch/dfs/Websites/t/tibristo/'
# sample will be something like 13_llqq_v3
idx_init += sample+'/'+sample+'.html">Study Home</a>\n<hr>\n<font size="3">\n'
idx_init += '<TABLE>\n<TR>\n<a href="../'+roc+'"><H3><font color="#0000FF">ROC Curves</font></H3></a>\n</TR>\n</TABLE>\n'
idx_init += '<H1><font color=\"#0000FF\">  </font></H1>\n<TABLE>\n'

idx_end = '\n<hr>\n<p>Site created by Tim Bristow \n</p>\n</div></div></div></body>\n</html>\n'
this_idx = idx_init.replace('TITLEVAR',algorithm)

idx_file = open('index.html','w')

#this_idx+=algorithm
#this_idx+=idx_init

this_idx+='<h2>\n'+tag+'\n</h2>\n'

# setup the column names
this_idx += openRow
this_idx += '<div class=\"col-sm-6 col-md-6 col-lg-6\" ><div style=\"text-align:center\"><b>Distributions within the 68% mass window</b></div></div>'
this_idx += '<div class=\"col-sm-6 col-md-6 col-lg-6\" ><div style=\"text-align:center\"><b>Distributions without the mass window cut</b></div></div>'
this_idx += closeRow

# loop through the sorted files and set them up in columns: mass window | no mass window
for f in sortedfiles:
    if f.find('png') == -1:
        continue
    varname = f[0:f.find('.')]
    this_idx+=openRow
    
    this_idx+='<div class=\"col-sm-6 col-md-6 col-lg-6\" >\n<a href=\"'+f
    this_idx+='\"> <img class=\"img-responsive\" src=\"'+f+'\"></a>'
    this_idx+='<br>  <p style=\"font-size:15px\">'+varname+'<a href=\"'+f+'\">[png]</a></p>\n'# <a href=\"'+varname+'.eps\">[eps]</a>\n'
    this_idx+='</div>\n'

    # add a space between the images
    

    # now add the no mass window on the right
    this_idx+='<div class=\"col-sm-6 col-md-5 col-lg-6\">\n<a href=\"'+f.replace('.png','_noMW.png')
    this_idx+='\"> <img class=\"img-responsive\" src=\"'+f.replace('.png','_noMW.png')+'\"></a>'
    #    this_idx+='<br> <p style=\"font-size:15px\">'+varname+' </p><br><a href=\"'+f.replace('.png','_noMW.png')+'\">[png]</a>\n'# <a href=\"'+varname+'.eps\">[eps]</a>\n'
    this_idx+='<br> <p style=\"font-size:15px\">'+varname+'<a href=\"'+f.replace('.png','_noMW.png')+'\">[png]</a></p>\n'# <a href=\"'+varname+'.eps\">[eps]</a>\n'
    this_idx+='</div>\n'
    this_idx += closeRow

# close the html
this_idx+=idx_end
idx_file.write(this_idx)
idx_file.close()

