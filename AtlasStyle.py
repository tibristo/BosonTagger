##    gROOT.LoadMacro("atlasstyle-00-03-05/AtlasStyle.C")
##    gROOT.LoadMacro("atlasstyle-00-03-05/AtlasLabels.C")
##    gROOT.LoadMacro("atlasstyle-00-03-05/AtlasUtils.C")
##    SetAtlasStyle()

#from ROOT import *
import ROOT as rt

def AtlasStyle():
    atlasStyle = rt.TStyle("ATLAS","Atlas style")

    # use plain black on white colors
    icol=0
    atlasStyle.SetFrameBorderMode(icol)
    atlasStyle.SetFrameFillColor(icol)
    atlasStyle.SetCanvasBorderMode(icol)
    atlasStyle.SetCanvasColor(icol)
    atlasStyle.SetPadBorderMode(icol)
    atlasStyle.SetPadColor(icol)
    atlasStyle.SetStatColor(icol)
    #atlasStyle.SetFillColor(icol)  # don't use: white fill color for *all* objects

    # set the paper & margin sizes
    atlasStyle.SetPaperSize(20,26)

    # set margin sizes
    atlasStyle.SetPadTopMargin(0.05)
    atlasStyle.SetPadRightMargin(0.05)
    atlasStyle.SetPadBottomMargin(0.16)
    atlasStyle.SetPadLeftMargin(0.16)

    # set title offsets (for axis label)
    atlasStyle.SetTitleXOffset(1.4)
    atlasStyle.SetTitleYOffset(1.4)

    # use large fonts
    #Int_t font=72  # Helvetica italics
    font=42  # Helvetica
    tsize=0.05
    atlasStyle.SetTextFont(font)

    atlasStyle.SetTextSize(tsize)
    atlasStyle.SetLabelFont(font,"x")
    atlasStyle.SetTitleFont(font,"x")
    atlasStyle.SetLabelFont(font,"y")
    atlasStyle.SetTitleFont(font,"y")
    atlasStyle.SetLabelFont(font,"z")
    atlasStyle.SetTitleFont(font,"z")

    atlasStyle.SetLabelSize(tsize,"x")
    atlasStyle.SetTitleSize(tsize,"x")
    atlasStyle.SetLabelSize(tsize,"y")
    atlasStyle.SetTitleSize(tsize,"y")
    atlasStyle.SetLabelSize(tsize,"z")
    atlasStyle.SetTitleSize(tsize,"z")

    # use bold lines and markers
    atlasStyle.SetMarkerStyle(20)
    atlasStyle.SetMarkerSize(1.2)
    atlasStyle.SetHistLineWidth(2)
    atlasStyle.SetLineStyleString(2,"[12 12]")  # postscript dashes

    # get rid of X error bars
    #atlasStyle.SetErrorX(0.001)
    # get rid of error bar caps
    atlasStyle.SetEndErrorSize(0.)

    # do not display any of the standard histogram decorations
    atlasStyle.SetOptTitle(0)
    #atlasStyle.SetOptStat(1111)
    atlasStyle.SetOptStat(0)
    #atlasStyle.SetOptFit(1111)
    atlasStyle.SetOptFit(0)

    # put tick marks on top and RHS of plots
    atlasStyle.SetPadTickX(1)
    atlasStyle.SetPadTickY(1)

    return atlasStyle


def SetAtlasStyle():
    print "Applying ATLAS style settings...\n"
    atlasStyle = AtlasStyle()
    rt.gROOT.SetStyle("ATLAS")
    rt.gROOT.ForceStyle()


def ATLASLabel(x, y, color, dR, tsize, text):
    l = rt.TLatex()
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextColor(color)
    l.SetTextSize(tsize)
    l.DrawLatex(x,y,"ATLAS")

    p = rt.TLatex()
    p.SetNDC()
    p.SetTextFont(42)
    p.SetTextColor(color)
    p.SetTextSize(tsize)
    p.DrawLatex(x+dR,y,text)


def myText(x, y, color, tsize, text):

    l = rt.TLatex()
    l.SetNDC()
    l.SetTextColor(color)
    l.SetTextSize(tsize)
    l.DrawLatex(x,y,text)


def myLineText(x, y, lsize, lcolor, lstyle, tsize, text):

    l = rt.TLatex()
    l.SetTextAlign(12)
    l.SetTextSize(tsize)
    l.SetNDC()
    l.DrawLatex(x,y,text)

    y1 = y-0.25*lsize
    y2 = y+0.25*lsize
    x2 = x-0.3*lsize
    x1 = x2-lsize

    mline = rt.TLine()
    mline.SetLineWidth(2)
    mline.SetLineColor(lcolor)
    mline.SetLineStyle(lstyle)
    y_new=(y1+y2)/2.
    mline.DrawLineNDC(x1,y_new,x2,y_new)

def myLineBoxText(x, y, lcolor, lstyle, bcolor, bstyle, size, text):

    l = rt.TLatex()
    l.SetTextAlign(12)
    l.SetTextSize(size/2.5)
    l.SetNDC()
    l.DrawLatex(x,y,text)

    yl1 = y-0.25*size
    yl2 = y+0.25*size
    xl1 = x-0.25*size-0.4*size
    xl2 = x+0.25*size-0.4*size


    #print "line = ",xl1,yl1,xl2,yl2
    mline = rt.TLine(xl1,yl1,xl2,yl2)
    mline.SetLineColor(lcolor)
    mline.SetLineStyle(lstyle)
    mline.SetLineWidth(2)
    y_new=(yl1+yl2)/2.
    mline.DrawLineNDC(xl1,y_new,xl2,y_new)

    yb1 = y-0.15*size
    yb2 = y+0.15*size
    xb1 = x-0.25*size-0.4*size
    xb2 = x+0.25*size-0.4*size


    #print "box  = ",xb1,yb1,xb2,yb2
    mbox = rt.TPave(xb1,yb1,xb2,yb2,0,"NDC")
    mbox.SetLineWidth(0)
    mbox.SetFillColor(bcolor)
    mbox.SetFillStyle(bstyle)
    mbox.Draw()
    
    return mbox

    

def myMarkerTextSmall(x, y, lcolor, lstyle, mcolor, mstyle, size, text):

    l = rt.TLatex()
    l.SetTextAlign(12)
    l.SetTextSize(size/2.5)
    l.SetNDC()
    l.DrawLatex(x,y,text)

    yl1 = y-0.25*size
    yl2 = y+0.25*size
    xl1 = x-0.25*size-0.4*size
    xl2 = x+0.25*size-0.4*size


    print "line = ",xl1,yl1,xl2,yl2
    mline = rt.TLine(xl1,yl1,xl2,yl2)
    mline.SetLineColor(lcolor)
    mline.SetLineStyle(lstyle)
    mline.SetLineWidth(2)
    y_new=(yl1+yl2)/2.
    mline.DrawLineNDC(xl1,y_new,xl2,y_new)

    yb1 = y-0.15*size
    yb2 = y+0.15*size
    xb1 = x-0.25*size-0.4*size
    xb2 = x+0.25*size-0.4*size

    print "box  = ",xb1,yb1,xb2,yb2

    marker = rt.TMarker((xb1+xb2)/2.0,y,mstyle)
    marker.SetNDC()
    marker.SetMarkerColor(mcolor)
    marker.SetMarkerStyle(mstyle)
    marker.Draw()
    
    return marker




