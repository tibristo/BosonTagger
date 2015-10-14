import sys
import numpy as np
from root_numpy import array2root
print sys.argv[1]
cols = np.linspace(1,41,41,dtype=int)
data = np.genfromtxt(sys.argv[1],delimiter=',',names=True,usecols=cols)

array2root(data, sys.argv[1].replace('.csv','.root'),'outputTree')
