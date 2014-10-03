def SignalPtWeight(pt_for_binning):

    scalept=1.0

    if  (pt_for_binning <= 225.0): scalept = 0.27
    elif(pt_for_binning <= 250.0): scalept = 0.32
    elif(pt_for_binning <= 275.0): scalept = 0.73
    elif(pt_for_binning <= 300.0): scalept = 1.72
    elif(pt_for_binning <= 335.0): scalept = 2.09
    elif(pt_for_binning <= 350.0): scalept = 2.34
    elif(pt_for_binning <= 400.0): scalept = 0.42
    elif(pt_for_binning <= 450.0): scalept = 0.73
    elif(pt_for_binning <= 500.0): scalept = 1.33
    elif(pt_for_binning <= 600.0): scalept = 0.35
    elif(pt_for_binning <= 700.0): scalept = 1.00
    elif(pt_for_binning <= 800.0): scalept = 2.55
    elif(pt_for_binning <= 900.0): scalept = 5.16
    elif(pt_for_binning <= 1000.0): scalept = 15.15
    return 1.0/scalept

