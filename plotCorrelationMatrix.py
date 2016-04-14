# -*- coding: iso-8859-15 -*-
import ROOT
import cPickle as pickle
import operator
import sys
import os.path
import math
ROOT.gStyle.SetPaintTextFormat("4.1f")
from matplotlib import rcParams
from matplotlib.afm import AFM
ROOT.gROOT.LoadMacro("getExtent.C")
from ROOT import getWidth
ROOT.gROOT.SetBatch(True)

def round_sigfigs(in_num, sig_figs):
    """Round to specified number of sigfigs.

    >>> round_sigfigs(0, sig_figs=4)
    0
    >>> int(round_sigfigs(12345, sig_figs=2))
    12000
    >>> int(round_sigfigs(-12345, sig_figs=2))
    -12000
    >>> int(round_sigfigs(1, sig_figs=2))
    1
    >>> '{0:.3}'.format(round_sigfigs(3.1415, sig_figs=2))
    '3.1'
    >>> '{0:.3}'.format(round_sigfigs(-3.1415, sig_figs=2))
    '-3.1'
    >>> '{0:.5}'.format(round_sigfigs(0.00098765, sig_figs=2))
    '0.00099'
    >>> '{0:.6}'.format(round_sigfigs(0.00098765, sig_figs=3))
    '0.000988'
    """
    num = float(in_num)
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0


def abbreviate(in_str):
    '''
    Method to abbreviate an algorithm into something easier to fit on a plot.
    in_str is the full Algorithm string -> CamKt12LCTopoPrunedXYZ for example.
    '''
    #afm_filename = os.path.join(rcParams['datapath'], 'fonts', 'afm', 'ptmr8a.afm')
    #afm = AFM(open(afm_filename))
    
    max_length = 0
    max_line = 0
    max_width = 0
    out_str = '#splitline{'
    rad_idx = 0
    # check for AntiKt or CamKt
    line = ''
    if in_str.find('AntiKt') != -1:
        line += 'anti-k_{t}'#'AK'
        rad_idx = 6
    elif in_str.find('CamKt') != -1:
        line += 'C/A'#'CA'
        rad_idx = 5
    # add radius to the output string
    rad = str(float(in_str[rad_idx:in_str.find('LC')])/10.0)
    if len(rad) == 1:
        rad = '0'+rad
    line += ' R='+str(rad) +' jets'
    out_str+=line+'}{#splitline{'#rad
    #print afm.string_width_height('blah ')
    max_length = len(out_str) - len('#splitline{}{#splitline{')
    max_width = getWidth(line)
    #out_str+='LC'

    # add abbrev for the type of grooming
    if in_str.find('BDRS') != -1:
        out_str+='Split-Filtered}{'
        max_length = max(len('Split-Filtered'), max_length)
        width = getWidth('Split-Filtered')
        if max_width < width:
            max_line = 1
            max_width = width
        # OLD example: CamKt6LCTopoSplitFilteredMu100SmallR30YCut0
        # NEW example: CamKt12LCTopoBDRSFilteredMU100Y4
        # want: CA6LCBDRSM100R30Y0
        config_str='#mu='#'M'
        # '{0:.3}'.format(round_sigfigs(3.1415, sig_figs=2))
        # old one
        # mu = float(in_str[in_str.find('MU')+2:in_str.find('Small')])/100.0
        # new one
        mu = float(in_str[in_str.find('MU')+2:in_str.find('Y')])/100.0
        config_str+='{0:.3}'.format(round_sigfigs(mu, sig_figs=2))+', '
        #out_str+='R'
        #out_str+=in_str[in_str.find('SmallR')+6:in_str.find('YCut')]
        config_str+='y_{filt}='
        # old one
        #out_str+=in_str[in_str.find('YCut')+4:]+'%'
        # new one
        config_str+=in_str[in_str.find('Y')+1:]+'%}}'
        max_length = max(len(config_str)-6, max_length)# the 6 comes from the _{}, # and }}
        width = getWidth(config_str)
        #print width
        if max_width < width:
            max_line = 2
            max_width = width
        out_str+=config_str

    elif in_str.find('Trim') != -1:
        # example: PtFrac5SmallR20
        # want: TRIMF5R20
        out_str+='Trimmed}{'
        max_length = max(len('Trimmed'), max_length)
        width = getWidth('Trimmed')
        if width > max_width:
            max_width = width
            max_line = 1
        config_str = 'f_{cut}='
        config_str+=in_str[in_str.find('Frac')+4:in_str.find('SmallR')]+'%,'
        config_str+=' R_{sub}='
        rsub = float(in_str[in_str.find('SmallR')+6:])/100.0
        config_str+=str('{0:.3}'.format(round_sigfigs(rsub, sig_figs=2)))+'}}'
        
        max_length = max(len(config_str)-8, max_length)# the 8 comes from the 2*_{} and }}
        width = getWidth(config_str)
        #print width
        if max_width < width:
            max_width = width
            max_line = 2
        out_str+=config_str
    elif in_str.find('Prun') != -1:
        out_str+='Pruned}{'
        max_length = max(max_length,len('Pruned'))
        width = getWidth('Pruned')
        if width > max_width:
            max_width = width
            max_line = 1
        # example: CamKt12LCTopoPrunedCaRcutFactor50Zcut10
        # want: CA12LCPRUNR50Z10
        config_str='R_{cut}='
        rcut=float(in_str[in_str.find('Factor')+6:in_str.find('Zcut')])/100.0
        config_str+=str('{0:.3}'.format(round_sigfigs(rcut, sig_figs=2)))+', '
        config_str+='z_{cut}='
        config_str+=in_str[in_str.find('Zcut')+4:]+'%}}'
        if max_length < len(config_str)-8:# the 8 comes from the 2*_{} and }}
            max_length = len(config_str)-8
        width = getWidth(config_str)
        #print width
        if max_width < width:
            max_line = 2
            max_width = width
        out_str+=config_str
    #line_width = getWidth(out_str)#blah.GetBBox().fWidth#LineWidth()
    return out_str, max_length, max_line, max_width#line_width

def plotMatrix(version, drawATLAS=False):

    # read input

    rejectionmatrix = pickle.load(open("rejectionmatrix_"+version+".p", "rb"))

    # get the pt range out of the version string
    pos_high = version.rfind("_")
    pt_high = version[pos_high+1:]
    pos_low = version.rfind("_",0,pos_high-1)
    pt_low = version[pos_low+1:pos_high]

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
    order = ['Angularity','Aplanarity','EEC_C2_1','EEC_C2_2','EEC_D2_1','EEC_D2_2','FoxWolfram20','PlanarFlow','SPLIT12','Sphericity','Tau21','TauWTA2','TauWTA2TauWTA1','ThrustMaj','ThrustMin','ZCUT12','Mu12','YFilt','m']
    latexVars = {'Angularity':'a_{3}','Aplanarity':'#it{A}','EEC_C2_1':'C_{2}^{(#beta=1)}','EEC_C2_2':'C_{2}^{(#beta=2)}','EEC_D2_1':'D_{2}^{(#beta=1)}','EEC_D2_2':'D_{2}^{(#beta=2)}','FoxWolfram20':'FoxWolfram20','PlanarFlow':'#it{P}','SPLIT12':'#sqrt{d_{12}}','Sphericity':'#it{S}','Tau21':'#tau_{21}','TauWTA2':'#tau^{WTA}_{2}','TauWTA2TauWTA1':'#tau^{WTA}_{2}/#tau^{WTA}_{1}','ThrustMaj':'#it{T}_{maj}','ThrustMin':'#it{T}_{min}','ZCUT12':'#sqrt{z_{12}}','Mu12':'#mu_{12}','YFilt':'#sqrt{y_{12}}','m':'m'}
    #sortedvars = sorted(varsdict.items(), key=operator.itemgetter(0))
    
    
    sortedvars = []
    sortedLatex = []
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
                if o in latexVars.keys():
                    sortedLatex.append(latexVars[o])
                else:
                    sortedLatex.append(o)
                # add the remaining variables to the sortedvars list
    for x in toadd:
        if not (x[0] == 'Dip12' or x[0] == 'Angularity' or x[0] == 'averageIntPerXing' or x[0] == ''):
            sortedvars.append(x)
            if x in latexVars.keys():
                sortedLatex.append(latexVars[x[0]])
            else:
                sortedLatex.append(x[0])

    # if there is a missing variable we append this to the end of the row
    # get abbreviations
    max_abbrev = 0
    max_width = 0
    abbreviations = []
    max_totwidth = 0
    for r in rejectionmatrix:
        rejvars = [x[0] for x in rejectionmatrix[r]]
        abb,lenabb,idx,width = abbreviate(r)
        if lenabb > max_abbrev:
            max_abbrev = lenabb
        if width > max_width:
            max_width = width
        abbreviations.append([abb,lenabb,idx,width])
        #print abb,lenabb,width
        max_totwidth = max(max_totwidth, getWidth(abb))
        for v in varsdict:
            if v not in rejvars:
                rejectionmatrix[r].append([v,0])
                
    for a in range(0,len(abbreviations)):
        #if len(abbreviations[a])
        add = ''
        #print abbreviations[a]
        
        if getWidth(abbreviations[a][0]) < max_totwidth:
            # sigh - need to know which line this belongs to!
            #print 'extending: ', abbreviations[a]
            #print getWidth(abbreviations[a][0])
            #for b in xrange(abbreviations[a][1], max_abbrev):
            #add = '#scale[0.1]{#color[0]{.}}'
            pos1 = abbreviations[a][0].find('{')+1
            pos2 = abbreviations[a][0].find('}{')
            add = abbreviations[a][0]#[pos1: pos2]
            while getWidth(add) <= max_totwidth:#-abbreviations[a][3]:
                add+='#scale[0.4]{#color[0]{.}}'#' '#kern[0.3]{}'
                #print 'added1'
            abbreviations[a][0] = add

    matrix = ROOT.TH2F("Background Rejection Matrix","Background Rejection Matrix",len(sortedvars),1, len(sortedvars)+1, len(rejectionmatrix), 1, len(rejectionmatrix)+1)
    matrix.SetTitle("")
    matrix.GetZaxis().SetRangeUser(0.0,35.0)
    matrix.GetZaxis().SetTitle("Bkg. rejection @ 50% signal eff.")


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
            matrix.GetXaxis().SetBinLabel(x+1,sortedLatex[x])#v[0])
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
        matrix.GetYaxis().SetBinLabel(i+1,abbreviations[i][0])#abbreviate(r))
        #print matrix.GetYaxis().Getlabel
        # create tpavetext object out of things
        #tpavetext = ROOT.TPaveText()
        #for line in abbrev:
        #    tpavetext.AddText(line)
        #tpavetexts.append()
    matrix.GetYaxis().LabelsOption('d')
    print 'matrix setup done'
    # turn on the colours
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPadBorderSize(0)
    ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetTopMargin(0.11)
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.15)
    ROOT.gStyle.SetTitleFont(matrix.GetZaxis().GetLabelFont())
    ROOT.gStyle.SetTitleAlign(23)
    ROOT.gStyle.SetTitleFontSize(matrix.GetZaxis().GetLabelSize())
    matrix.SetStats(0)
    matrix.SetMarkerSize(1)
    matrix.Draw("TEXTCOLZ")
    # for errors to be printed in text
    #matrix.Draw("TEXTECOLZ")

    print 'matrix drawn'
    from ROOT import TLatex
    title = TLatex();
    title.SetNDC();
    title.SetTextSize(0.03)
    title.SetTextFont(42)
    title.DrawLatex(0.15,0.91,"#sqrt{s}=13 TeV, "+pt_low+"<p_{T}^{Truth}<"+pt_high+" (GeV), 68% mass window")

    title2 = TLatex();
    title2.SetNDC();
    title2.SetTextSize(0.04)
    title2.SetTextFont(42)
    title2.DrawLatex(0.15,0.95,"Grooming and tagging combinations")

    if drawATLAS:
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
