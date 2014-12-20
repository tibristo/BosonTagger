#include <string>
#include <iostream>

double SignalPtWeight2(double pt = 0)
{
  double weight = ptweights->GetBinContent(ptweights->GetXaxis()->FindBin(pt/1000));
  return weight;
}

TH1F * ptweights;

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
      ptweights = new TH1F("ptreweight","ptreweight",numbins,0,3000);
      next_edge = step_size;
    }
  else
    {
      //binsarr = &(*bins)[0];
      ptweights = new TH1F("ptreweight","ptreweight",sizeof(bins)/*->size()*/-1,bins);//binsarr);
      //next_edge = (*bins)[1];
    }
  // normally 200 bins would be for 0 - 3500.  Now we are going
  // to say 3500/numbins instead.
  current_edge = 200;
  if (numbins!=-1)//bins->empty())
    next_edge = current_edge+step_size;
  else
    next_edge = ptweights->GetXaxis()->GetBinUpEdge(1);

  while(getline(f, line))
    {
      std::stringstream ss(line);
      getline(ss, edge, ',');
      getline(ss, bkg, ',');
      getline(ss, sig, ',');
      //if (atof(edge.c_str()) > minedge)
      //continue;
      running_bkg += atof(bkg.c_str());
      running_sig += atof(sig.c_str());
      
      if (atof(edge.c_str()) > next_edge)
	{
	  if (numbins!=-1)//bins->empty())
	    {
	      next_edge += step_size;
	    }
	  else
	    {
	      next_edge = ptweights->GetXaxis()->GetBinUpEdge(counter+1);
	      /*
	      if (counter >= bins.size())
		next_edge += atof(edge.c_str())*2;
	      else
	      next_edge = bins[counter+1];*/
	    }
	  if (running_sig == 0)
	    ptweights->SetBinContent( counter, 0 );
	  else
	    ptweights->SetBinContent( counter, running_bkg/running_sig );
	  running_bkg = 0;
	  running_sig = 0;
	  counter++;
	}
      
    }

  f.close();
}

