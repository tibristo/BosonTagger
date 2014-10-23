#include <string>
#include <iostream>

double SignalPtWeight2(double pt = 0)
{
  double weight = ptweights->GetBinContent(ptweights->GetXaxis()->FindBin(pt/1000));
  return weight;
}

TH1F * ptweights;

void loadweights(std::string filename = "", int numbins = 200)
{
  if (ptweights)
    return;
  ifstream f(filename.c_str());
  std::string line;
  ptweights = new TH1F("ptreweight","ptreweight",200,0,3500);
  int counter = 1;

  while(getline(f, line))
    {
      try
	{
	  ptweights->SetBinContent(counter, atof(line.c_str()));
	}
      catch (exception &e)
	{
	  std::cout << "added more than 200 bins to ptweights, or couldn't convert line to float: " << e << std::endl;
	}
      counter++;
    }

  ptweights->Rebin(200/numbins);
  f.close();
}

void variableBins(std::vector<float> & bins)
{
  float * rebin  = &bins[0];// create array out of vector
  ptweights = ptweights->Rebin(bins.size(), "", rebin);
}
