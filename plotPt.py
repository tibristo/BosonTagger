import massPtFunctions

from ROOT import *

from AtlasStyle import *
SetAtlasStyle()


pthists = []
s = TFile('/Disk/ecdf-nfs-ppe/atlas/users/tibristo/BoostedBosonFiles/13tev_mc15_jz5_v1/AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_jz5_v1/TopoTrimmedPtFrac5SmallR20_inclusive_sig.root')
sighist = s.Get('pt_reweightsig')
sighist.SetDirectory(0)
s.Close()
b = TFile('/Disk/ecdf-nfs-ppe/atlas/users/tibristo/BoostedBosonFiles/13tev_mc15_jz5_v1/AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_jz5_v1/TopoTrimmedPtFrac5SmallR20_inclusive_bkg.root')

bkghist = b.Get('pt_reweightbkg')
bkghist.SetDirectory(0)
b.Close()


massPtFunctions.setPtWeightFile(sighist, bkghist, normalise=False)
ptweights = massPtFunctions.getPtWeightsFile()
# draw the pt histogram and reweighted histogram
ptweightBins = [200,250,300,350,400,450,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1800,2000,2200,2400,2600,2800,3000]
ptbinssmall = [200,250,300,350,400,450,500,600,700,800,900,1000]
from array import array

pt = {'sig':TH1F('pT sig','pT sig',24,array('d',ptweightBins)), 'bkg':TH1F('pT bkg','pT bkg',24,array('d',ptweightBins))}
pt_ca12 = {'sig': TH1F('pT ca12 sig','pT ca12 sig',24,array('d',ptweightBins)), 'bkg':TH1F('pT ca12 bkg','pT ca12 bkg',24,array('d',ptweightBins))}
pt_rw = {'sig': TH1F('pT_rw sig','pT_rw sig',24,array('d',ptweightBins)), 'bkg':TH1F('pT_rw bkg','pT_rw bkg',24,array('d',ptweightBins))}
#pt_rw = {'sig': TH1F('pT_rw sig','pT_rw sig',24,array('d',ptweightBins)), 'bkg':TH1F('pT_rw bkg','pT_rw bkg',24,array('d',ptweightBins))}
pt_rw_ca12 = {'sig':TH1F('pT_rw ca12 signal','pT_rw ca12 signal',24,array('d',ptweightBins)), 'bkg':TH1F('pT_rw ca12 bkg','pT_rw ca12 bkg',24,array('d',ptweightBins))}
#pt_rw_ca12 = {'sig':TH1F('pT_rw ca12 signal','pT_rw ca12 signal',11,array('d',ptbinssmall)), 'bkg':TH1F('pT_rw ca12 bkg','pT_rw ca12 bkg',11,array('d',ptbinssmall))}

# memory resident hists
for hist in [pt, pt_ca12, pt_rw, pt_rw_ca12]:
    hist['sig'].SetFillColor(4); hist['sig'].SetLineColor(4); hist['sig'].SetMarkerColor(4); hist['sig'].SetMarkerStyle(20);
    hist['bkg'].SetFillColor(2); hist['bkg'].SetLineColor(2); hist['bkg'].SetMarkerColor(2); hist['bkg'].SetMarkerStyle(24);
    for k in hist.keys():
        hist[k].Sumw2()
        hist[k].SetDirectory(0)
        hist[k].SetLineStyle(1); hist[k].SetFillStyle(0); hist[k].SetMarkerSize(1.5);
        hist[k].GetXaxis().SetTitle('#it{p}_{T} (GeV)')
        hist[k].GetYaxis().SetTitle('# entries normalised to 1')
        
fname = '/Disk/ecdf-nfs-ppe/atlas/users/tibristo/BoostedBosonFiles/13tev_mc15_jz5_v1/AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_jz5_v1/TopoTrimmedPtFrac5SmallR20_inclusive_FILE.root'

for datatype in ['sig', 'bkg']:
    f = TFile.Open(fname.replace('FILE',datatype),'read')
    tree = f.Get('outputTree')
    entries = tree.GetEntries()
    applyptrw = datatype == 'sig'
    bkg = datatype == 'bkg'
    for evt in xrange(entries):
        if evt % 10000:
            print evt
        tree.GetEntry(evt)
        ca12pt = tree.jet_CamKt12Truth_pt/1000.0
        topopt = tree.jet_AntiKt10LCTopoTrimmedPtFrac5SmallR20_pt/1000.0

        # bkg weight
        if bkg:
            weight = tree.evt_filtereff*tree.evt_xsec*tree.mc_event_weight/tree.evt_nEvts
        else:
            weight = ptweights.GetBinContent(ptweights.GetXaxis().FindBin(ca12pt))
        pt_rw_ca12[datatype].Fill(ca12pt, weight)
        pt_rw[datatype].Fill(topopt,weight)
        bkg_weight = weight if bkg else 1.0
        pt_ca12[datatype].Fill(ca12pt, bkg_weight)
        pt[datatype].Fill(topopt, bkg_weight)

    f.Close()


# normalise all of the hists
for hist in [pt, pt_ca12, pt_rw, pt_rw_ca12]:
    for k in hist.keys():
        hist[k].Scale(1./hist[k].Integral())
        
    max_val = max(hist['sig'].GetMaximum(), hist['bkg'].GetMaximum())
    hist['sig'].SetMaximum(max_val*1.1)
    hist['sig'].SetMinimum(5e-9)
    
    hist['bkg'].SetMaximum(max_val*1.1)
    hist['bkg'].SetMinimum(5e-8)
    
c = TCanvas('Pt distributions')

c.SetTitle('Unweighted and normalised CamKt12 truth pT distributions')

pt_ca12['sig'].Draw('hist')
pt_ca12['bkg'].Draw('histsame')
legend = TLegend(0.7,0.55,0.9,0.85);legend.SetFillColor(kWhite)
legend.AddEntry(pt['sig'], "Signal CamKt12Truth pT")
legend.AddEntry(pt['bkg'], "Background CamKt12Truth pT")
legend.Draw()

c.SaveAs('pt_unweighted_ca12.png')


c.Clear()
legend.Clear()
c.SetTitle('Unweighted and normalised groomed topo jet pT distributions')
pt_rw['sig'].Draw('hist')
pt_rw['bkg'].Draw('histsame')

legend.AddEntry(pt['sig'], "Signal AntiKt10LCTopoTrimmedPtFrac5SmallR20 pT")
legend.AddEntry(pt['bkg'], "Background AntiKt10LCTopoTrimmedPtFrac5SmallR20 pT")
legend.Draw()

c.SaveAs('pt_unweighted_topo.png')



c.Clear()
legend.Clear()
c.SetTitle('Weighted and normalised CamKt12 truth pT distributions')
c.SetLogy()
pt_rw_ca12['sig'].Draw('pe2')
pt_rw_ca12['bkg'].Draw('pe2same')

legend.AddEntry(pt['sig'], "Signal CamKt12Truth pT")
legend.AddEntry(pt['bkg'], "Background CamKt12Truth pT")
legend.Draw()

c.SaveAs('pt_weighted_ca12.png')



c.Clear()
legend.Clear()
c.SetTitle('weighted and normalised groomed topo jet pT distributions')
pt['sig'].Draw('pe2')
pt['bkg'].Draw('pe2same')

legend.AddEntry(pt['sig'], "Signal AntiKt10LCTopoTrimmedPtFrac5SmallR20 pT")
legend.AddEntry(pt['bkg'], "Background AntiKt10LCTopoTrimmedPtFrac5SmallR20 pT")
legend.Draw()

c.SaveAs('pt_weighted_topo.png')
