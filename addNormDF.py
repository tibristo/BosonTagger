#
import sys
import os.path
import copy
import argparse

#
import ROOT as rt
import AtlasStyle as atlas
atlas.SetAtlasStyle()
#rt.gStyle.SetFillStyle(4000)
rt.gROOT.SetBatch(True)
import getParams as gp
mode = 'READ'

def createPlot(fname, plotType = 'discriminant', parameter_file = '', pt_range=[0,0], bdt=False, extraFiles=''):
    # values for the plotting of labels and stuff
    labelx = labelx_dnn = 0.45
    labely = labely_dnn = 0.88
    labelx_bdt = 0.2
    labely_bdt = 0.88
    legendx1 = legendx1_dnn = 0.65
    legendy1 = legendy1_dnn = 0.6
    legendx2 = legendx2_dnn = 0.9
    legendy2 = legendy2_dnn = 0.75
    xlow = 0.0
    xhigh = 1.0

    legendx1_bdt = 0.68
    legendy1_bdt = 0.75
    legendx2_bdt = 0.93
    legendy2_bdt = 0.95
    
    paramx = paramx_dnn = 0.4
    paramy = paramy_dnn = 0.8
    paramx_bdt = 0.2
    paramy_bdt = 0.8
    if bdt:
        labelx = labelx_bdt
        labely = labely_bdt
        legendx1 = legendx1_bdt
        legendx2 = legendx2_bdt
        legendy1 = legendy1_bdt
        legendy2 = legendy2_bdt
        paramx = paramx_bdt
        paramy = paramy_bdt
    #if plotType != 'discriminant':
    #    legendx1 = 0.65
    #    legendx2 = 0.9
    #    legendy1 = 0.68
    #    legendy2 = 0.88
    
    xtitle = 'Signal Probability'
    # need to get the tagger id from the file name
    pos = fname.find('paramID_')+8
    pos_end = copy.deepcopy(pos)
    while fname[pos_end].isdigit():
        pos_end+=1
    taggerid = fname[pos:pos_end]
    #print fname, taggerid
    cv_pos = fname.find('ID',pos_end+1,len(fname))+3
    cv = fname[cv_pos]
    print 'cv:',cv
    
    f = rt.TFile.Open(fname,mode)#'UPDATE'
    if plotType == 'discriminant':
        distrib = 'Discriminant'
        combined = 'Discriminant'
        file_flag = '_disc'
    else:
        distrib = 'Decision Function'
        combined = 'Decision Functions Norm'
        xtitle = 'Decision Function'
        file_flag = '_df'
        xlow = -1
    keys = f.GetListOfKeys()
    has_sigdf = False
    has_bkgdf = False
    has_df = False
    for k in keys:
        if k.GetName() == 'Signal '+distrib:
            has_sigdf = True
        if k.GetName() == 'Background '+distrib:
            has_bkgdf = True
        if k.GetName() == combined:
            has_df = True

    if not has_sigdf or not has_bkgdf:
        print 'keys do not exist'
        return
    if has_df and mode=='UPDATE':
        print 'already has the df canvas'
        return


    sigdf = f.Get('Signal '+distrib).Clone()
    sigdf.SetDirectory(0)
    sigdf.SetLineColor(rt.kRed)
    sigdf.SetMarkerSize(1)
    sigdf.SetTitle(xtitle)
    #sigdf.Rebin(5)
    sigdf.GetXaxis().SetTitle(xtitle)
    sigdf.GetYaxis().SetTitle('Normalised Entries')
    #sigdf.GetXaxis().SetRangeUser(xlow,xhigh)
    #if sigdf.Integral() != 0.0:
    #    sigdf.Scale(1./sigdf.Integral())
    bkgdf = f.Get('Background '+distrib).Clone()
    bkgdf.SetDirectory(0)
    bkgdf.SetLineColor(rt.kBlue)
    bkgdf.SetTitle(xtitle)
    bkgdf.SetMarkerSize(1)
    bkgdf.GetXaxis().SetTitle(xtitle)
    bkgdf.GetYaxis().SetTitle('Normalised Entries')
    #bkgdf.GetXaxis().SetRangeUser(xlow,xhigh)
    
    #if bkgdf.Integral() != 0.0:
    #    bkgdf.Scale(1./bkgdf.Integral())

    f.Close()
    print type(sigdf)
    # now get the rest of them from the other files!!!!
    #tlistsig = rt.TList()
    #tlistbkg = rt.TList()
    #tlistsig.Add(sigdf)
    #tlistbkg.Add(bkgdf)

    # find extra files!
    if bdt:
        folds = 10
    else:
        folds = 5
    ef = []
    for x in range(0,folds):
        ffff = fname[:cv_pos]+str(x)+fname[cv_pos+1:]
        ef.append(ffff)
    print ef

    print sigdf.Integral()
    for tf in ef:
        tmpfile = rt.TFile.Open(tf,mode)
        sigtmp = tmpfile.Get('Signal ' + distrib).Clone()
        sigtmp.SetDirectory(0)
        bkgtmp = tmpfile.Get('Background ' + distrib).Clone()
        bkgtmp.SetDirectory(0)
        sigdf.Add(sigtmp)
        bkgdf.Add(bkgtmp)
        tmpfile.Close()
    #ef.close()
    #sigdf2 = sigdf.Clone()
    #sigdf2.Reset()
    #sigdf2.Merge(tlistsig)
    #bkgdf2 = bkgdf.Clone()
    #bkgdf2.Reset()
    #bkgdf2.Merge(tlistbkg)
    print sigdf.Integral()
    if plotType != 'discriminant':
        sigdf.Rebin(4)
        bkgdf.Rebin(4)
    else:
        sigdf.Rebin(2)
        bkgdf.Rebin(2)

    if sigdf.Integral() != 0.0:
        sigdf.Scale(1./sigdf.Integral())
    if bkgdf.Integral() != 0.0:
        bkgdf.Scale(1./bkgdf.Integral())

    
    tc = rt.TCanvas(combined)
    tc.SetTitle(combined)
    legend = rt.TLegend(legendx1, legendy1, legendx2, legendy2);legend.SetFillColor(rt.kWhite)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(sigdf, 'Signal')
    legend.AddEntry(bkgdf, 'Background')

    max_v = max(sigdf.GetMaximum(), bkgdf.GetMaximum())
    sigdf.SetMaximum(max_v*1.4)
    bkgdf.SetMaximum(max_v*1.4)


    sigdf.Draw('hist')
    bkgdf.Draw('histsame')
    legend.Draw()


    label = rt.TLatex()
    label.SetTextSize(0.05)
    
    label.DrawLatexNDC(labelx,labely, '#sqrt{s}=13 TeV, '+str(pt_range[0])+'<p_{T}<'+str(pt_range[1])+' (GeV)')
        
    if fname.find('AGILE') != -1 or fname.find('DNN') != -1:
        param_label = gp.dnn(parameter_file)
    elif fname.find('BDT') != -1:
        param_label = gp.bdt(parameter_file)
    print param_label

    atlas.myText(paramx, paramy, 1, 0.04, param_label[taggerid])

    
    if mode == 'UPDATE':
        tc.Write()
    else:
        tc.SaveAs(fname.replace('.root',file_flag+'.pdf'))

    #f.Close()

def main(inargs):
    print inargs
    parser = argparse.ArgumentParser(description="Plot ROC curves from ROOT files")
    parser.add_argument('--key',required=True, help = 'key for the files to be read')
    parser.add_argument('--folder',required=True, help = 'folder for the input files')
    parser.add_argument('--paramfile',required=True, help = 'file that stores the sorted results of the models')
    parser.add_argument('--ptlow',required=True, type = int, help='pt low')
    parser.add_argument('--pthigh',required=True, type = int, help='pt high')
    parser.add_argument('--bdt',required=False,dest='bdt',action='store_true',help='is this a bdt file or dnn file?')
    parser.add_argument('--extraFiles',required=False, dest='extraFiles')
    #parser.add_argument('--df',required=False, dest='df',action='store_true',help='if the decision function and discriminant should be plotted')
    #parser.set_defaults(df=False)
    parser.set_defaults(bdt=False)
    parser.set_defaults(extraFiles='')
    args = parser.parse_args()
    

    
    #key = 'mc15_nTrk_v1_bkg_v3'
    #folder = 'DNN_mc15_nTrk_v3'
    rocfolder = 'ROC_root'
    if args.bdt:
        rocfolder = 'ROC'
    files = [args.folder+'/'+rocfolder+'/'+f for f in os.listdir('./'+args.folder+'/'+rocfolder+'/') if f.find(args.key) != -1 and f.endswith('.root')]
    #parameter_file = args.folder+'/data_mc15_nTrk_v1_bkg_v3_sortedresults.txt'
    parameter_file = args.folder+'/'+args.paramfile
    pt_range = [int(args.ptlow),int(args.pthigh)]

    for f in files:
        print f
        createPlot(f, 'discriminant', parameter_file, pt_range, bdt = args.bdt)
        if args.bdt:
            createPlot(f, 'df', parameter_file, pt_range, bdt = args.bdt)

if __name__=='__main__':
    print 'blah'
    main(sys.argv)

