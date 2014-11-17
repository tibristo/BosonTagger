import ROOT
import cPickle as pickle


def plotMatrix():

    # read input

    rejectionmatrix = pickle.load(open("rejectionmatrix.p", "rb"))


    tc = ROOT.TCanvas()

    maxlistvars = []
    maxlistcounter = 0
    # check that there is a common list of variables
    # if one has more, add these to other rows
    # this is not foolproof because the order could be incorrect... need to sort
    for r in rejectionmatrix.keys():
        if maxlistcounter < len (rejectionmatrix[r]):
            maxlistcounter = len(rejectionmatrix[r])
            # set the variable name, which is from r[rej,var]
            maxlistvars = [x[1] for x in rejectionmatrix[r]]
    # append extra rows
    for r in rejectionmatrix.keys():
        if len(rejectionmatrix[r]) < maxlistcounter:
            for l in range(len(rejectionmatrix[r]),maxlistcounter):
                rejectionmatrix[r].append([0,maxlistvars[l]])


    print len(maxlistvars)
    matrix = ROOT.TH2F("Correlation matrix","Correlation matrix",len(maxlistvars),1, len(maxlistvars), len(rejectionmatrix), 1, len(rejectionmatrix))

    # fill the th2 with all of the values
    for i, r in enumerate(rejectionmatrix):
        # r will have [key, [rejection, variable name]]
        for x, v in enumerate(rejectionmatrix[r]):
            # add teh variables
            print v[1]
            matrix.GetXaxis().SetBinLabel(x+1,v[1])
            matrix.SetBinContent(x+1,i+1, v[0])
        matrix.GetYaxis().SetBinLabel(i+1,r)

    # turn on the colours
    ROOT.gStyle.SetPalette(1)
    matrix.Draw("COLZ")
    
    tc.SaveAs("matrix.pdf")


if __name__=='__main__':
    plotMatrix()
