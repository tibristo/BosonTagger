#include "TROOT.h"

double getWidth(std::string text)
{
  
  TCanvas * tc = new TCanvas("blah","blah", 500,500);
  TLatex tl;// = new TLatex();
  //std::cout << text << std::endl;
  tl.SetTextSize(0.5);
  TLatex * tmp = tl.DrawLatex(0.1,0.1,text.c_str());
  UInt_t w=0,h=0;
  double width = tmp->GetBBox().fWidth;
  //tl.SetBBoxX1(0);
  //tl.SetBBoxX2(0);
  //tl.SetBBoxY1(0);
  //tl.SetBBoxY2(0);
  //std::cout << width <<std::endl;
  //tl.DrawLatex(0.1,0.1,'');
  delete tc;
  
  return width;//
  //tl.GetTextExtent(w,h,text.c_str());
  //return w;
}
