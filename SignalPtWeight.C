double SignalPtWeight(double pt_for_binning = 0){

    float scalept=1.0;

    if     (pt_for_binning <= 225.0) scalept = 0.27;
    else if(pt_for_binning <= 250.0) scalept = 0.32;
    else if(pt_for_binning <= 275.0) scalept = 0.73;
    else if(pt_for_binning <= 300.0) scalept = 1.72;
    else if(pt_for_binning <= 335.0) scalept = 2.09;
    else if(pt_for_binning <= 350.0) scalept = 2.34;
    else if(pt_for_binning <= 400.0) scalept = 0.42;
    else if(pt_for_binning <= 450.0) scalept = 0.73;
    else if(pt_for_binning <= 500.0) scalept = 1.33;
    else if(pt_for_binning <= 600.0) scalept = 0.35;
    else if(pt_for_binning <= 700.0) scalept = 1.00;
    else if(pt_for_binning <= 800.0) scalept = 2.55;
    else if(pt_for_binning <= 900.0) scalept = 5.16;
    else if(pt_for_binning <= 1000.0) scalept = 15.15;
    return 1.0/scalept;
}
