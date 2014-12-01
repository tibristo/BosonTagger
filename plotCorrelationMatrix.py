import ROOT
import cPickle as pickle
import operator

def plotMatrix():

    # read input

    rejectionmatrix = pickle.load(open("rejectionmatrix_70_110_v3.p", "rb"))


    tc = ROOT.TCanvas()

    maxlistvars = []
    maxlistcounter = 0
    varsdict = {}
    # check that there is a common list of variables
    # if one has more, add these to other rows
    # this is not foolproof because the order could be incorrect... need to sort
    for r in rejectionmatrix.keys():
        for v in rejectionmatrix[r]:
            # count variables
            if v[0] in varsdict:
                varsdict[v[0]] += 1
            else:
                varsdict[v[0]] = 1

    # sort all of the variables. sortedvars is a tuple sorted by value of varsdict
    sortedvars = sorted(varsdict.items(), key=operator.itemgetter(0))
    
    # if there is a missing variable we append this to the end of the row
    for r in rejectionmatrix:
        rejvars = [x[0] for x in rejectionmatrix[r]]
        for v in varsdict:
            if v not in rejvars:
                rejectionmatrix[r].append([v,0])

    


    matrix = ROOT.TH2F("Correlation matrix","Correlation matrix",len(sortedvars),1, len(sortedvars), len(rejectionmatrix), 1, len(rejectionmatrix))

    # fill the th2 with all of the values
    for i, r in enumerate(rejectionmatrix):
        # r will have [key, [rejection, variable name]]
        # create dictionary from rejectionmatrix[r]
        rej = dict(rejectionmatrix[r])
        # we want to sort rej based on sorted vars.... Won't work yet
        for x, v in enumerate(sortedvars):
            # add teh variables
            # Just want to plot the variable name, not the rest of the name
            print v
            matrix.GetXaxis().SetBinLabel(x+1,v[0])
            matrix.SetBinContent(x+1,i+1, rej[v[0]])
        matrix.GetYaxis().SetBinLabel(i+1,r)

    # turn on the colours
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPadBorderSize(0)
    ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetLeftMargin(0.2)
    matrix.SetStats(0)
    matrix.Draw("COLZ")
    
    print rejectionmatrix
    tc.SaveAs("matrixinv_70_110_v2.pdf")


if __name__=='__main__':
    plotMatrix()
