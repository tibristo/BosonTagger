
std::map<long, float> nevents;

double NEvents(long runNumber)
{
  if (nevents.count(runNumber) > 0)
    return nevents[runNumber];
  else
    return 1;
}

void loadEvents(std::string filename)
{
  ifstream f(filename.c_str());
  std::string line;
  while (getline(f,line))
    {
      std::stringstream ss(line);
      std::string run;
      getline(ss,run,',');
      std::string ev;
      getline(ss,ev,',');
      nevents[atol(run.c_str())] = atof(ev.c_str());
    }
  f.close();
}

