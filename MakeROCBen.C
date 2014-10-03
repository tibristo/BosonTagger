#include "TROOT.h"


//void MakeROCBen(int type, TH1D *&S, TH1D *&B, TGraph &curve, TGraph &curve_up, TGraph &curve_do){
void MakeROCBen(int type, TH1D *&S, TH1D *&B, TGraph &curve){
    //Usage:
    //TH1D *S = new TH1D();
    //TH1D *B = new TH1D();
    //...Fill S and B with signal (S) and background (B)
    //Need S and B to have the same number of bins!
    //TH1D *curve = new TH1D("","",M,0,1); where M is however fine you want the ROC curve binning to be.
    //ROC(S,B,curve);

    const int n = S->GetNbinsX();
    vector<double> s;
    vector<double> b;
    vector<double> r;
    vector<double> serr;
    vector<double> berr;

    //cout<<"Getting init ROC"<<endl;
    for (int i=0;i<n;i++){
        s.push_back(S->GetBinContent(i+1));
        b.push_back(B->GetBinContent(i+1));
        serr.push_back(S->GetBinError(i+1));
        berr.push_back(B->GetBinError(i+1));
        if (B->GetBinContent(i+1)>0){
            r.push_back(S->GetBinContent(i+1)/B->GetBinContent(i+1));
        }
        else{
            r.push_back(-1);
        }
    }

//     for (int i=0;i<n;i++){
//         cout<<s.at(i)<<"  "<<serr.at(i)<<"  "<<b.at(i)<<"  "<<berr.at(i)<<endl;
//     }

    //sort by ascending order
    float temp_s=1;
    float temp_b=1;
    float temp_r=1;
    float temp_serr=1;
    float temp_berr=1;
    int sizes = s.size();
    for(int isort=sizes; isort>1; isort=isort-1){
        for(int i=0; i<isort-1; i++){
            if( r.at(i)<r.at(i+1) ){
                temp_s  = s.at(i);
                temp_b  = b.at(i);
                temp_serr  = serr.at(i);
                temp_berr  = berr.at(i);
                temp_r  = r.at(i);

                s.at(i) = s.at(i+1);
                b.at(i) = b.at(i+1);
                serr.at(i) = serr.at(i+1);
                berr.at(i) = berr.at(i+1);
                r.at(i) = r.at(i+1);

                s.at(i+1) = temp_s;
                b.at(i+1) = temp_b;
                serr.at(i+1) = temp_serr;
                berr.at(i+1) = temp_berr;
                r.at(i+1) = temp_r;
            }
        }
    }

//     cout<<"Reordered ROC"<<endl;
//     for (int i=0;i<n;i++){
//         cout<<i<<"  "<<s.at(i)<<"  "<<b.at(i)<<"  "<<r.at(i)<<endl;
//     }

    double totalB = B->Integral();
    double totalS = S->Integral();


    //put into graph
    //cout<<"NPoints: "<<n<<endl;
    TGraph gr(n);
    TGraph gr_up(n);
    TGraph gr_do(n);
    for (int i=0; i<n; i++){
        double myS = 0.;
        double myB = 0.;
        double mySerr = 0.;
        double myBerr = 0.;
        for (int j=0; j<i; j++){
            myS += s.at(j)/totalS;
            myB += b.at(j)/totalB;

//             mySerr += serr.at(j)/totalS;
//             myBerr += berr.at(j)/totalB;

            //cout<<"ROCa: "<<i<<"  "<<myS<<"  "<<mySerr<<"  "<<myB<<"  "<<myBerr<<endl;

            mySerr += pow(pow(mySerr, 2.0) + pow((serr.at(j)/totalS), 2.0), 0.5);
            myBerr += pow(pow(myBerr, 2.0) + pow((berr.at(j)/totalB), 2.0), 0.5);

//            cout<<"ROCb: "<<i<<"  "<<myS<<"  "<<mySerr<<"  "<<myB<<"  "<<myBerr<<endl;

        }
        if(type==1){
            gr.SetPoint(i, myS, (1-myB));
            gr_up.SetPoint(i, myS+mySerr, (1-(myB-myBerr)));
            gr_do.SetPoint(i, myS-mySerr, (1-(myB+myBerr)));
        }
        if(type==2){
            if(myB==1)
                gr.SetPoint(i, myS, 100000);
            else{
                gr.SetPoint(i, myS, 1/(1-myB));
            }
        }
    }

    curve=gr;
    // curve_up=gr_up;
    // curve_do=gr_do;

    return;


}
