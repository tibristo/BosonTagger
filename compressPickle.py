import bz2
import pickle
import os
from multiprocessing import Pool
import time
import sys


def printProgress(tasks):
    total = len(tasks)
    print total
    finished = 0.0
    for t in tasks:
        if t.ready():
            finished+=1
            t.get()
    print 'finished: ' + str(finished)
    return float(finished/total)



def compress(fname):
    print 'compressing file: ' + fname
    with open(fname,'r') as d:
        model = pickle.load(d)
    with bz2.BZ2File(fname.replace('.pickle','pbz2'),'w') as bz:
        pickle.dump(model,bz)

    return


files = ['evaluationObjects/'+f for f in os.listdir('evaluationObjects/') if f.endswith('pickle')]

#compress(files[0])
pool = Pool(8)
pool_results = []

print f

for f in files:
    pool_results.append(pool.apply_async(compress, [f]))
pool.close()


prog = printProgress(pool_results)
while prog < 1:
    time.sleep(10)
    prog = printProgress(pool_results)
    print "progress: " + str(prog)
