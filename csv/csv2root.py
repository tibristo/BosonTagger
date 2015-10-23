import sys
import numpy as np
from root_numpy import array2root
print sys.argv[1]
# need to find out how many columns are in the file

f = open(sys.argv[1])
l = f.readline()
colcount = l.count(',')
f.close()

cols = np.linspace(1,colcount,colcount,dtype=int)
data = np.genfromtxt(sys.argv[1],delimiter=',',names=True,usecols=cols)

array2root(data, sys.argv[1].replace('.csv','.root'),'outputTree')
