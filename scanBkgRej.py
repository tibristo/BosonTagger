import TaggerTim
from cpickle import pickle
import ROOT
paths = []
configs = []
fileids = []


maxrej = 0
maxrejvar = ''
maxrejm_min = 0
maxrejm_max = 0
maxalg = ''

rejectionmatrix = {}

for p,c,f in zip(paths,configs,fileids):
    rej, var, m_min, m_max=TaggerTim.main([c,'-i',p,'-f', f, '--pthigh=2000','--ptlow=350','--nvtx=99','--nvtxlow=0'])
    print 'Algorithm: ' + c
    print 'Rejection: ' + str(rej)
    print 'Variable: ' + var
    print 'Mass min: ' + str(m_min)
    print 'Mass max: ' + str(m_max)
    if rej > maxrej:
        maxrej = rej
        maxrejvar = var
        maxrejm_min = m_min
        maxrejm_max = m_max
        maxalg = c
    # load total background rejection matrix from pickle file
    totalrejection = pickle.load(open("tot_rej.p","rb"))
    rejectionmatrix[fileids] = totalrejection

print '----------------------------------'
print 'Algorithm: ' + maxalg
print 'Rejection: ' + str(maxrej)
print 'Variable: ' + maxrejvar
print 'Mass min: ' + str(maxrejm_min)
print 'Mass max: ' + str(maxrejm_max)

tc = TCanvas()

maxlistvars = []
maxlistcounter = 0
# check that there is a common list of variables
# if one has more, add these to other rows
# this is not foolproof because the order could be incorrect... need to sort
for r in rejectionmatrix:
    if maxlistcounter > len (r):
        maxlistcounter = len(r)
        # set the variable name, which is from r[rej,var]
        maxlistvar = [x[1] for x in r]
# append extra rows
for r in rejectionmatrix:
    if len(r) < maxlistcounter:
        for l in range(len(r),maxlistcounter):
            r.append([maxlistvar[l], 0])


matrix = ROOT.TH2F("Correlation matrix","Correlation matrix",len(maxlistvar),1, len(maxlistvar), len(rejectionmatrix), 1, len(rejectionmatrix))

# fill the th2 with all of the values
for i, r in enumerate(rejectionmatrix.items()):
    # r will have [key, [rejection, variable name]]
    for x in range(1, len(r[1])):
        # add teh variables
        h.GetXaxis().SetBinLabel(x,r[1][1])
        # fill the values
        h.Fill(i, r[1][0])
    h.GetYaxis().SetBinLabel(i,r[0])

# turn on the colours
ROOT.gStyle.SetPalette(1)
matrix.Draw("COLZ")

matrix.SaveAs("matrix.pdf")

