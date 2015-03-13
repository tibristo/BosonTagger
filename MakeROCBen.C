#include "TROOT.h"

void MakeROCBen(int type, TH1D *&S, TH1D *&B, TGraph &curve, TGraph &bkgRejPower, double sigeff, double bkgeff){//, TGraph &errUp, TGraph &errDo){

  if(S->Integral()!=0) S->Scale(1.0/S->Integral());
  if(B->Integral()!=0) B->Scale(1.0/B->Integral());


    const int n = S->GetNbinsX();
    vector<double> s;
    vector<double> b;
    vector<double> r;
    vector<double> serr;
    vector<double> berr;

    //cout<<"Getting init ROC"<<endl;
    for (int i=1;i<=n;i++){
        s.push_back(S->GetBinContent(i));
        b.push_back(B->GetBinContent(i));
        serr.push_back(S->GetBinError(i));
        berr.push_back(B->GetBinError(i));
        if (B->GetBinContent(i)>0){
            r.push_back(S->GetBinContent(i)/B->GetBinContent(i));
        }
        else{
            r.push_back(-1);
        }
    }

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

    double totalB = B->Integral();
    double totalS = S->Integral();


    //put into graph
    TGraph gr(n);
    TGraph gr_pow(n);
    TGraph gr_up(n);
    TGraph gr_do(n);

    for (int i=0; i<n; i++){
        double myS = 0.;
        double myB = 0.;
        double mySerr = 0;
        double myBerr = 0;
        for (int j=0; j<i; j++){
            myS += s.at(j)/totalS;
            myB += b.at(j)/totalB;
	    
            mySerr += pow(pow(mySerr, 2.0) + pow((serr.at(j)/totalS), 2.0), 0.5);
            myBerr += pow(pow(myBerr, 2.0) + pow((berr.at(j)/totalB), 2.0), 0.5);

        }
	gr.SetPoint(i, myS*sigeff, (1-myB*bkgeff));
	gr.SetPointError(i, mySerr*sigeff, myBerr*bkgeff)
	//gr.SetPoint(i, myS*sigeff, 1/((1-myB)*(1-bkgeff)));
        if(type==1){
	  gr_pow.SetPoint(i, myS*sigeff, (1-myB*bkgeff));
	  gr_up.SetPoint(i, (myS+mySerr)*sigeff, (1-(myB+myBerr)*bkgeff));
	  gr_do.SetPoint(i, (myS-mySerr)*sigeff, (1-(myB-myBerr)*bkgeff));
        }
        else if(type==2){
            if(myB==0)
	      gr_pow.SetPoint(i, myS*sigeff, 10000);//100000);
            else{
	      //gr_pow.SetPoint(i, myS*sigeff, 1/((1-myB)*(1-bkgeff)) );
	      gr_pow.SetPoint(i, myS*sigeff, 1/(1-(1-myB*bkgeff)) );
            }
        }
	else if(type==3){
	  if(myB==1)
	      gr_pow.SetPoint(i, myS*sigeff, 100000);
	  else{
	      //gr_pow.SetPoint(i, myS*sigeff, 1/(myB*bkgeff));
	      gr_pow.SetPoint(i, myS*sigeff, 1/(1-myB*bkgeff));
	      }
        }
    }

    curve=gr;
    bkgRejPower = gr_pow;
    //errUp = gr_up;
    //errDo = gr_do;

    return;


}
