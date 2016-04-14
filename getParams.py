from itertools import islice
import copy

dnn_results = {}
bdt_results = {}

def dnn(infile):
    # get the top 5
    global dnn_results
    
    if infile in dnn_results.keys():
        return dnn_results[infile]
    
    with open(infile) as f:
        top5 = list(islice(f,6))
    results = {}
    for i, t in enumerate(top5):
        if i == 0:
            continue
        spl = t.split(",")
        # just easier to read as sep lines
        legend = 'm='+spl[1] + ', '
        legend += 'reg='+spl[2] + ', '
        legend += 'lr='+spl[3] + ', '
        legend += 'ue='+spl[4] + ', '
        legend += 'se='+spl[5]
        
        results[spl[0]] = 'DNN ID ' + spl[0] + ' ('+legend+')'
    dnn_results[infile] = copy.deepcopy(results)
    return results

def bdt(infile):
    global bdt_results
    if infile in bdt_results.keys():
        return bdt_results[infile]
    
    with open(infile) as f:
        top5 = list(islice(f,6))
    results = {}
    for i,t in enumerate(top5):
        if i == 0:
            continue
        spl = t.split(",")
        # just easier to read as sep lines
        legend = 'depth='+spl[1] + ', '
        legend += 'lr='+spl[2] + ', '
        legend += 'est='+spl[3] + ', '
        
        results[spl[0]] = 'BDT ID ' + spl[0] + ' ('+legend+')'
    bdt_results[infile] = copy.deepcopy(results)
    return results
