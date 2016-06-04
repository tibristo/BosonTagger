from itertools import islice
import copy
import os.path
dnn_results = {}
bdt_results = {}

def dnn(infile):
    # get the top 5
    global dnn_results


    
    if infile in dnn_results.keys():
        return dnn_results[infile]

    # check if it is actually a file - if not it's a string and use that.
    if not os.path.isfile(infile):
        tmp = {}
        for x in xrange(0,5):
            tmp[str(x)] = 'DNN ID ' +str(x)+ ' ' +infile
        dnn_results[infile] = tmp
        return tmp
    
    with open(infile) as f:
        top5 = list(islice(f,6))
    results = {}
    for i, t in enumerate(top5):
        if i == 0:
            continue
        spl = t.split(",")
        # just easier to read as sep lines
        legend = 'm='+spl[1] + ', '
        # all of the best of the same regularisation factor....
        #legend += 'reg='+str('%.2E' % float(spl[2])) + ', '
        legend += 'lr='+str('%.2E' % float(spl[3])) + ', '
        legend += 'u='+spl[4] + ', '
        legend += 's='+spl[5]
        
        results[spl[0]] = 'DNN ID ' + spl[0] + ' ('+legend+')'
    dnn_results[infile] = copy.deepcopy(results)
    return results

def bdt(infile):
    global bdt_results
    if infile in bdt_results.keys():
        return bdt_results[infile]

    # check if it is actually a file - if not it's a string and use that.
    if not os.path.isfile(infile):
        tmp = {}
        for x in xrange(0,5):
            tmp[str(x)] = 'BDT ID ' +str(x)+ ' '+infile
        dnn_results[infile] = tmp
        return tmp

    
    with open(infile) as f:
        top5 = list(islice(f,6))
    results = {}
    for i,t in enumerate(top5):
        if i == 0:
            continue
        spl = t.split(",")
        # just easier to read as sep lines
        legend = 'dep='+spl[1] + ', '
        legend += 'lr='+spl[2] + ', '
        legend += 'est='+spl[3]
        
        results[spl[0]] = 'BDT ID ' + spl[0] + ' ('+legend+')'
    bdt_results[infile] = copy.deepcopy(results)
    return results
