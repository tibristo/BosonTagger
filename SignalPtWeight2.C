#include <string>
#include <iostream>

double SignalPtWeight2(double pt = 0)
{
  double weight = ptweights->GetBinContent(ptweights->GetXaxis()->FindBin(pt/1000));
  return weight;
}

TH1F * ptweights;
TH1F * ptweights_sig;
TH1F * ptweights_bkg;

//void loadweights(std::string filename = "", int numbins = 200, std::vector<float> * bins = 0)
void loadweights(std::string filename = "", int numbins = 200, Float_t * bins = 0)
{
  //if (ptweights)
  //  return;
  ifstream f(filename.c_str());
  std::string line;
  float step_size = 3000/numbins;
  int counter = 1;
  std::string edge, bkg, sig;
  float running_bkg, running_sig;
  float current_edge = 0, next_edge = 0;
  float * binsarr;
  if (numbins!=-1)//bins->empty())
    {
      //ptweights = new TH1F("ptreweight","ptreweight",numbins,0,3000)
      ptweights_sig = new TH1F("ptreweight_sig","ptreweight_sig",numbins,0,3000);
      ptweights_bkg = new TH1F("ptreweight_bkg","ptreweight_bkg",numbins,0,3000);
      next_edge = step_size;
    }
  else
    {
      //binsarr = &(*bins)[0];
      //ptweights = new TH1F("ptreweight","ptreweight",sizeof(bins)/*->size()*/-1,bins);//binsarr);
      ptweights_sig = new TH1F("ptreweight_sig","ptreweight_sig",sizeof(bins)-1,bins);
      ptweights_bkg = new TH1F("ptreweight_bkg","ptreweight_bkg",sizeof(bins)-1,bins);
      //next_edge = (*bins)[1];
    }
  // normally 200 bins would be for 0 - 3500.  Now we are going
  // to say 3500/numbins instead.
  current_edge = 200;
  if (numbins!=-1)//bins->empty())
    next_edge = current_edge+step_size;
  else
    next_edge = ptweights_bkg->GetXaxis()->GetBinUpEdge(1);

  while(getline(f, line))
    {
      std::stringstream ss(line);
      getline(ss, edge, ',');
      getline(ss, bkg, ',');
      getline(ss, sig, ',');
      //if (atof(edge.c_str()) > minedge)
      //continue;
      
      if (atof(edge.c_str()) > next_edge)
	{
	  if (numbins!=-1)//bins->empty())
	    {
	      next_edge += step_size;
	    }
	  else
	    {
	      next_edge = ptweights_bkg->GetXaxis()->GetBinUpEdge(counter+1);

	    }
	  //if (running_sig == 0)
	  //{
	      //ptweights->SetBinContent( counter, 0 );
	      ptweights_sig->SetBinContent( counter, running_sig );
	      //}
	      //else
	      //{
	      //ptweights->SetBinContent( counter, running_bkg/running_sig );
	      ptweights_bkg->SetBinContent( counter, running_bkg );
	      //}
	  running_bkg = 0;
	  running_sig = 0;
	  counter++;
	}
      //else
      //{
	  running_bkg += atof(bkg.c_str());
	  running_sig += atof(sig.c_str());
	  //}
      
    }
  if (ptweights_sig->Integral()!=0)
    ptweights_sig->Scale(1./ptweights_sig->Integral());
  if (ptweights_bkg->Integral()!=0)
    ptweights_bkg->Scale(1./ptweights_bkg->Integral());
  ptweights = (TH1F*)ptweights_bkg->Clone();
  ptweights->Divide(ptweights_sig);
  f.close();
}

