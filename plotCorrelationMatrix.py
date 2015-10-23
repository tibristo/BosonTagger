import ROOT
import cPickle as pickle
import operator
import sys
import os.path
ROOT.gStyle.SetPaintTextFormat("4.1f")

def abbreviate(in_str):
    '''
    Method to abbreviate an algorithm into something easier to fit on a plot.
    in_str is the full Algorithm string -> CamKt12LCTopoPrunedXYZ for example.
    '''
    out_str = ''
    rad_idx = 0
    # check for AntiKt or CamKt
    if in_str.find('AntiKt') != -1:
        out_str += 'AK'
        rad_idx = 6
    elif in_str.find('CamKt') != -1:
        out_str += 'CA'
        rad_idx = 5
    # add radius to the output string
    rad = in_str[rad_idx:in_str.find('LC')]
    if len(rad) == 1:
        rad = '0'+rad
    out_str+=rad
    out_str+='LC'

    # add abbrev for the type of grooming
    if in_str.find('SplitFilt') != -1:
        out_str+='BDRS'
        # example: CamKt6LCTopoSplitFilteredMu100SmallR30YCut0
        # want: CA6LCBDRSM100R30Y0
        out_str+='M'
        out_str+=in_str[in_str.find('Mu')+2:in_str.find('Small')]
        out_str+='R'
        out_str+=in_str[in_str.find('SmallR')+6:in_str.find('YCut')]
        out_str+='Y'
        out_str+=in_str[in_str.find('YCut')+4:]

    elif in_str.find('Trim') != -1:
        # example: PtFrac5SmallR20
        # want: TRIMF5R20
        out_str+='TRIM'
        out_str+='F'
        out_str+=in_str[in_str.find('Frac')+4:in_str.find('SmallR')]
        out_str+='R'
        out_str+=in_str[in_str.find('SmallR')+6:]

    elif in_str.find('Prun') != -1:
        out_str+='PRUN'
        # example: CamKt12LCTopoPrunedCaRcutFactor50Zcut10
        # want: CA12LCPRUNR50Z10
        out_str+='R'
        out_str+=in_str[in_str.find('Factor')+6:in_str.find('Zcut')]      
        out_str+='Z'
        out_str+=in_str[in_str.find('Zcut')+4:]

    return out_str

def plotMatrix(version):

    # read input

    rejectionmatrix = pickle.load(open("rejectionmatrix_"+version+".p", "rb"))



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

    # sort all of the variables. sortedvars is a tuple sorted by value of varsdict. order contains the desired order if these variables are present
    #order = ['Angularity','Aplanarity','EEC_C2_1','EEC_C2_2','EEC_D2_1','EEC_D2_2','FoxWolfram20','PlanarFlow','SPLIT12','Sphericity','Tau21','TauWTA2TauWTA1','ThrustMaj','ThrustMin','ZCUT12','Mu12','YFilt','m']
    order = ['Angularity','Aplanarity','EEC_C2_1','EEC_C2_2','EEC_D2_1','EEC_D2_2','FoxWolfram20','PlanarFlow','SPLIT12','Sphericity','Tau21','TauWTA2TauWTA1','ThrustMaj','ThrustMin','ZCUT12','Mu12','YFilt','m']    
    #sortedvars = sorted(varsdict.items(), key=operator.itemgetter(0))
    
    
    sortedvars = []
    # variables that are not in the "order" list
    toadd = []
    # check which variables are not in the order list
    for v in varsdict.items():
        if v[0] not in order:
            toadd.append(v)
    # add variables in the "order" list to the sorteredvars list
    for o in order:
        for v in varsdict.items():
            if o == v[0] and not (o == 'Dip12' or o == 'Angularity' or o == 'averageIntPerXing' or o == ''):
                sortedvars.append(v)
    # add the remaining variables to the sortedvars list
    for x in toadd:
        if not (x[0] == 'Dip12' or x[0] == 'Angularity' or x[0] == 'averageIntPerXing' or x[0] == ''):
            sortedvars.append(x)

    # if there is a missing variable we append this to the end of the row
    for r in rejectionmatrix:
        rejvars = [x[0] for x in rejectionmatrix[r]]
        for v in varsdict:
            if v not in rejvars:
                rejectionmatrix[r].append([v,0])
    
    print rejectionmatrix

    matrix = ROOT.TH2F("Background Rejection Matrix","Background Rejection Matrix",len(sortedvars),1, len(sortedvars)+1, len(rejectionmatrix), 1, len(rejectionmatrix)+1)
    matrix.GetZaxis().SetRangeUser(0.0,35.0)
    # fill the th2 with all of the values
    for i, r in enumerate(rejectionmatrix):
        # r will have [key, [rejection, variable name]]
        # create dictionary from rejectionmatrix[r]


        rej = {}
        var4 = False
        for variables in rejectionmatrix[r]:
            if len(variables) == 4:
                # nominal, error up, error down
                rej[variables[0]] = [ variables[1], variables[2], variables[3] ]
                var4 = True
            #if not var4:
            else:
                rej[variables[0]] = [variables[1],-1,-1]
            
        #print rej
        # we want to sort rej based on sorted vars.... Won't work yet
        for x, v in enumerate(sortedvars):
            # add teh variables
            # Just want to plot the variable name, not the rest of the name
            matrix.GetXaxis().SetBinLabel(x+1,v[0])
            if len(rej[v[0]]) == 1:
                value_nominal = rej[v[0]]
            else:
                value_nominal = rej[v[0]][0]
                value_errup = rej[v[0]][1]
                value_errdo = rej[v[0]][2]
                matrix.SetBinError(x+1,i+1, value_errup)
                #print 'setting bin error: ' + str(value_errup)
            matrix.SetBinContent(x+1,i+1, value_nominal)
            #print str(x+1) + ', ' + str(i+1) + ' : ' + str(value_nominal)
        matrix.GetYaxis().SetBinLabel(i+1,abbreviate(r))

    print 'matrix setup done'
    # turn on the colours
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPadBorderSize(0)
    ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetLeftMargin(0.2)
    matrix.SetStats(0)
    matrix.SetMarkerSize(1)
    matrix.Draw("TEXTCOLZ")
    # for errors to be printed in text
    #matrix.Draw("TEXTECOLZ")

    print 'matrix drawn'
    from ROOT import TLatex 
    texw = TLatex();
    texw.SetNDC();
    texw.SetTextSize(0.035);
    texw.SetTextFont(72);
    texw.DrawLatex(0.6,0.91,"ATLAS");
    p = TLatex();
    p.SetNDC();
    p.SetTextFont(42);
    p.SetTextSize(0.035);
    p.SetTextColor(ROOT.kBlack);
    p.DrawLatex(0.68,0.91,"Simulation Work in Progress");#"Internal Simulation");
    # check that the matrix folder exists, if not, create it
    if not os.path.exists('matrix'):
        os.makedirs('matrix')
    tc.SaveAs("matrix/matrixinv_"+version+".pdf")

    
    #print rejectionmatrix
    


if __name__=='__main__':
    plotMatrix(sys.argv[1])
