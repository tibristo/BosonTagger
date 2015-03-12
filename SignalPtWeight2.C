#include <string>
#include <iostream>
#include "TROOT.h"
// global histogram for the pt weights ratio (bkg/sig) and the signal and background pt spectrum
TH1F * ptweights;
TH1F * ptweights_sig;
TH1F * ptweights_bkg;

/*
 * Return the ratio of background to signal for a given pT.
 *
 * pt --- pT to get weight for.
 * @returns the ratio or weight to apply.
 */
double SignalPtWeight2(double pt = 0)
{
  double weight = ptweights->GetBinContent(ptweights->GetXaxis()->FindBin(pt/1000));
  return weight;
}

/*
 * Load the weights file and create the ptweighting histogram.
 *
 * @args
 * filename --- input csv file with signal and background event counts
 * numbins --- Number of bins to use for the pt weighting histogram.
 * bins --- An array containing the bin edges so we can use variable bin sizes.
 */
void loadweights(std::string filename = "", int numbins = 100, Float_t * bins = 0)
{
  // input file
  ifstream f(filename.c_str());
  // counters for bins
  float step_size = 3000/numbins;
  int counter = 1;
  // variables used for reading the lines of the input file
  std::string line;
  std::string edge, bkg, sig;
  // running total to be added to a pt bin
  float running_bkg, running_sig;
  // the lower and upper edges for the current bin
  float current_edge = 200, next_edge = 0;

  // if we are not using a variable size binning
  if (numbins!=-1)
    {
      // don't go beyond 3 TeV or below 200 GeV
      ptweights_sig = new TH1F("ptreweight_sig","ptreweight_sig",numbins,200,3000);
      ptweights_bkg = new TH1F("ptreweight_bkg","ptreweight_bkg",numbins,200,3000);
      next_edge = current_edge+step_size;
    }
  // using variable size bins
  else
    {
      ptweights_sig = new TH1F("ptreweight_sig","ptreweight_sig",sizeof(bins)-1,bins);
      ptweights_bkg = new TH1F("ptreweight_bkg","ptreweight_bkg",sizeof(bins)-1,bins);
      // the beginning of the next edge will be the up edge of the current bin
      next_edge = ptweights_bkg->GetXaxis()->GetBinUpEdge(1);
    }

  // loop through input file and fill in the ptweights histogram
  while(getline(f, line))
    {
      // csv, split into tokens
      std::stringstream ss(line);
      getline(ss, edge, ',');
      getline(ss, bkg, ',');
      getline(ss, sig, ',');

      // check we are still in the same bin
      if (atof(edge.c_str()) > next_edge)
	{
	  // fixed bins
	  if (numbins!=-1)
	    {
	      next_edge += step_size;
	    }
	  else
	    {
	      // get the edge for the next bin
	      next_edge = ptweights_bkg->GetXaxis()->GetBinUpEdge(counter+1);

	    }
	  // record the current signal and background counts before resetting
	  ptweights_sig->SetBinContent( counter, running_sig );
	  ptweights_bkg->SetBinContent( counter, running_bkg );
	  running_bkg = 0;
	  running_sig = 0;
	  // increment counter for bin index
	  counter++;
	}
      // count the background and signal
      running_bkg += atof(bkg.c_str());
      running_sig += atof(sig.c_str());
    }

  // normalise to 1
  if (ptweights_sig->Integral()!=0)
    ptweights_sig->Scale(1./ptweights_sig->Integral());
  if (ptweights_bkg->Integral()!=0)
    ptweights_bkg->Scale(1./ptweights_bkg->Integral());
  // clone the background and divide
  ptweights = (TH1F*)ptweights_bkg->Clone();
  ptweights->Divide(ptweights_sig);
  // close input file
  f.close();
}

