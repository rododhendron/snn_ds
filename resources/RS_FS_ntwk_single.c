#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nrutil.h"
#include <time.h>

#define dim 1000
#define dimmexc 800
#define diminh 200


#define Pi 3.1415


double par(double []);
double par1(double [],int [],int);
double ran2(long *idum);
void indexx(unsigned long n, double arr[]);
double media( double []);
void funcd(double tau, double *fvalue, double *fderivative);
void funcderivata(double tau, double *fvalue, double *fderivative);
void func(double tau, double *fvalue, double *fderivative);
double rtsafe(void (*funcd)(double, double *, double *), double x1, double x2,double xacc); 
double abss(double);
int sgn(double);
double tetaaa(double);
float modu(float);
int abssint(int);
int scelta(int);
double gausdev();
double gausdevdet(double ind);
double doubgauss(double t,double t0,double T1,double T2,double amp);
double heaviside(double x);

double avver(double []);

FILE *timecourse,*rasterexc,*rasterinh,*rateavv,*distrib,*fp1;

double BIN=.001;



int t,i,r,j,s;
double gij[dim][dim];
int gijint[dim][dim];

int gijintEE[dimmexc][dimmexc];
int gijintEI[dimmexc][diminh];
int gijintIE[diminh][dimmexc];
int gijintII[diminh][diminh];



int conn[dim][2];
double dimdoubll=dim;
double fase[dim];

int main()
{

    double afferent,T1,T2,amp,t0;
    t0=6;
    T1=0.05;
    T2=0.05;
    amp=1.;

	/*VARIABLES DEFINITION*/
		
	double v[dim],vold[dim],hetEl[dim],tolsp[dim],isyn[dim],phase[dim],isi[dim],GEinh[dimmexc],GEexc[dimmexc],GIinh[diminh],GIexc[diminh],ww[dimmexc],rexc,rinh;
	double indicator[dim];
    int orderexc[dimmexc];
	double stepp=0.0001;
	double g=10.5;
	double vthr=100;
	double vres=-100;
	double tauu=0.001;
	double vav,rate,vavnumb,ratebinn;
    double vave,vavi,vavnumbe,vavnumbi,vavtot,vavnumbtot;
	double tbinold=0;
	double deltabin=0.02;
	double timet;
	double aleakav=-5;
    double adeltaa=0.;
	double inpext;
	double ratetheory,ratetheoryold;
	double vavtheory,vavtheoryold,kuratheory,kuratheory0,kuratheory1,kuratheory2;
	double campo,campoold;
	double kuramoto=0;
	double ksparse=4;
	double variance=0;
	double variance0=0;
	double variance1=0;
	double contavariance=0;
	double kuravariance=0;
	double kuravariance0=0;
	double kuravariance1=0;
	double ratetoc=0;
	double kuratheav=0;
	double noise[dim];
	double differences[dim];
	double avdifferences[dim];
	double avfdifferences[dim];
	double countdifferences[dim];
	double ampnoise=0;
	double randnumb;
	double globvarr,mediaglobii;
	double delay=0;
	double  timedecay=3;
	double fieldrec[dim];
	double U=1;
	double globfi,globvar,variancee,coeff;
	double contacoeff,noiseterm;
	double kmedio,kvariance;
	double amposc,maxosc,minosc,maxoscr,minoscr,process,rateglobvarr,ratemediaglobii;
	double rateun[dim];
	double varproc=0;
	int chispara[dim];
	double iss[dim];
	double  issqu[dim];
	double nmisi[dim];
	double numya,numyb,ya,yb,oo,yyglob;
	double yy[dim];
	process=0;
	coeff=0;
	contacoeff=0;
	double vrappr,tolsprappr;
tolsprappr=0;
double gammazero;
double gzzero;
double izzero;
double balancefactor;
double rstar;
double vstar;
double omega;
double kmedioee;
double kmedioii;
double fracin=0.2;
int ni=floor(fracin*dim);
int ne=floor(dim*(1-fracin));
    
    int kvalue0,kvalue1;
        double  kee,kii,kei,kie;

     double  avvee,avvii,avvei,avvie;


double avvre,avvri,sdre,sdri,countrates;
    
double nee=ne;
double nii=ni;
double gei;
double gee;
double gii;
double gie;
double ieav;
double iiav;
double g0;
double gei0;
double gee0=g0*0.5;
double gie0;
double gii0;
    double epsilon;
double ie0,ii0,alpha,beta,nubar,transientt;
    double tstop;
    double pconn;
    
    double glRS;
    double CmRS;
    double bRS;
    double aRS;
    double twRS;
    double deltaRS;
    
    
    double glFS;
    double CmFS;
    double bFS;
    double aFS;
    double twFS;
    double deltaFS;
    
    double ElRS;
    double ElFS;
    
    double Ee;
    double Ei;
    
    double collad;
    double Trefr=5;
    
    double tausynexc=5;
    double tausyninh=5;
    
    double thr=-50;
    double tauw;
    double told;
 
    double sigmaexc_El,Elavvexc,sigmainh_El,Elavvinh;
    int hui,contu;
    double factor=1.;

    

    
    
    /*dimulation duration*/
    
    tstop=5.;
    transientt=0.01;
    stepp=0.004*1.e-3; /*until 0.002*/
    
    
    /*connection parameters*/
    
    pconn=0.05;
    pconn=pconn*10000/dim;
    

    
    factor=1.05;
    
    gee=1.5*1.e-9*factor;
    gie=1.5*1.e-9;
    gii=5.*1.e-9;
    gei=5.*1.e-9;
    
    Ee=0*1.e-3;
    Ei=-80*1.e-3;
    
    

    /*neurons parameters*/
    
    
    glRS=15.*1.e-9;
    CmRS=200*1.e-12;
    bRS=10*1.e-12;
    aRS=0.e-9;
    twRS=1000;
    deltaRS=2*1.e-3;
    ElRS=-65*1.e-3;
    sigmaexc_El=0.2*0;
    Elavvexc=ElRS;
    
    ElFS=-65*1.e-3;
    glFS=15*1.e-9;
    CmFS=200*1.e-12;
    bFS=0*1.e-12;
    aFS=0*1.e-9;
    twFS=500;
    deltaFS=0.5*1.e-3;

    sigmainh_El=0.1*0.;
    Elavvinh=ElFS;
    
    
    
    
    tausynexc=5*1.e-3;
    tausyninh=5*1.e-3;
    tauw=500*1.e-3;
    Trefr=5*1.e-3;
    

    
    
    
    
    
    
    vres=-65*1.e-3;
    thr=-50*1.e-3;
    double vspike=-30*1.e-3;

    
    double timescalefactor=1;
    tausynexc*=timescalefactor;
    tausyninh*=timescalefactor;
    /*
    gee=gee/timescalefactor;
    gii=gii/timescalefactor;
    gei=gei/timescalefactor;
    gie=gie/timescalefactor;
     */
    
    
    
    /*gee*=1.05;*/

    printf("Simulation starts!!\n");
    double inputrateonexc=2.5;
    double inputrateoninh=0.;
    double inputrateonexc0=inputrateonexc;

    

    rasterexc=fopen("rasterplot_EXc_het_1,05.dat","w");
    rasterinh=fopen("rasterplot_Inh_het_1,05.dat","w");


    
    
    char filename1[100]="________time_traces_het_1,05.dat";
    

    int lll,ooo;
    double ggg;
    
    /*EE submatrix*/
    
    
    contu=0;
    for(i=0;i<ne;i++){
        for(r=0;r<ne;r++){
            randnumb=rand()/(RAND_MAX+1.0);
            gijintEE[i][r]=0;
            if(randnumb<pconn*1.){
                gijintEE[i][r]=1;
                
            }
            
            
            
        }
    }
    
    /*II submatrix*/
    
    for(i=0;i<ni;i++){
        for(r=0;r<ni;r++){
            
            randnumb=rand()/(RAND_MAX+1.0);
            gijintII[i][r]=0;
            if(randnumb<pconn){
                gijintII[i][r]=1;
                
            }
            
            
        }
    }
    
    /*EI submatrix*/
    for(i=0;i<ne;i++){
        for(r=0;r<ni;r++){
            
            
            randnumb=rand()/(RAND_MAX+1.0);
            gijintEI[i][r]=0;
            if(randnumb<pconn){
                gijintEI[i][r]=1;
                
            }
            
            
        }
    }
    
    
    /*IE submatrix*/
    for(i=0;i<ni;i++){
        for(r=0;r<ne;r++){
            randnumb=rand()/(RAND_MAX+1.0);
            gijintIE[i][r]=0;
            if(randnumb<pconn){
                gijintIE[i][r]=1;
                
            }
            
            
        }
    }
    
    
    
    

    
    int huj=0;
    



    
    
    inputrateonexc0=1.5;
    hui=0;



inputrateonexc0=1.5;



    
    for(hui=0;hui<1;hui++)
    {

        
        //Elavvinh+=0.1*1.e-3; from 0 to 30
        Elavvinh+=0.3*1.e-3; // from 30 to 50
            for(huj=0;huj<1;huj++)
    {
        srand(time(NULL));
        
        
        Elavvinh=-70*1.e-3;
        sigmainh_El=0.15;
        sigmaexc_El=0.;
        /*printf("%lf	%lf\n",sigmaexc_El,sigmainh_El);*/
        
        
        kvalue0=floor(hui/10)+'0';
        kvalue1=(hui-10*floor(hui/10))+'0';
        

        
        filename1[4]=floor(hui/10)+'0';
        filename1[5]=(hui-10*floor(hui/10))+'0';
        filename1[6]=floor(huj/10)+'0';
        filename1[7]=(huj-10*floor(huj/10))+'0';
        

        /*printf("ee",filename1);*/
  
    
 
    for(r=0;r<dim;r++)
    {
        
        
        
        for(s=0;s<dim;s++){
           
            
            if(r<ne && s<ne){
                
                gij[r][s]=gee*gijintEE[r][s];
                gijint[r][s]=gijintEE[r][s];
                
            }
            
            
            
            if(r>=ne && s<ne){
                
                    gij[r][s]=gie*gijintIE[r-ne][s];
                    gijint[r][s]=gijintIE[r-ne][s];
                
                
            }
            
            if(r<ne && s>=ne){
                
                
                
                  gij[r][s]=gei*gijintEI[r][s-ne];
                    gijint[r][s]=gijintEI[r][s-ne];
                
            }
            
            if(r>ne && s>=ne){
                
                
                gij[r][s]=gii*gijintII[r-ne][s-ne];
                gijint[r][s]=gijintII[r-ne][s-ne];
                
            }
            
            
        }
    }

    
 

	oo=0;
	for(j=0;j<ne;j++)
	{

        v[j]=-80+10*(0.5-rand()/(RAND_MAX+1.0));
        v[j]=v[j]*1.e-3;
        
        hetEl[j]=(1+sigmaexc_El*gausdev())*Elavvexc;
     
        /*hetEl[j]=(1+sigmaexc_El*gausdevdet(1.*j/ne))*Elavvexc;*/
        
        ww[j]=0.;
        
        
        fase[0]=0;
		yy[j]=0.5-rand()/(RAND_MAX+1.0);
		rateun[j]=0;
				isyn[j]=0;
		tolsp[j]=0;
		
		randnumb=rand()/(RAND_MAX+1.0);
		phase[j]=0;
		isi[j]=1;
		indicator[j]=10000000;
		fieldrec[j]=0.5*rand()/(RAND_MAX+1.0);
		iss[j]=0;
		issqu[j]=0;
		nmisi[j]=0;
		chispara[j]=0;
		differences[j]=0;
		countdifferences[j]=0;
		avdifferences[j]=0;

		}
    
    
    
    
    oo=0;
    for(j=ne;j<dim;j++)
    {
      
        v[j]=-80+10*(0.5-rand()/(RAND_MAX+1.0));
        
        v[j]=v[j]*1.e-3;

        hetEl[j]=(1+sigmainh_El*gausdev())*Elavvinh;
 
        
        
        
        fase[0]=0;
        yy[j]=0.5-rand()/(RAND_MAX+1.0);
        rateun[j]=0;
        
        isyn[j]=0;
        tolsp[j]=0;
        randnumb=rand()/(RAND_MAX+1.0);
        phase[j]=0;
        isi[j]=1;
        indicator[j]=10000000;
        fieldrec[j]=0.5*rand()/(RAND_MAX+1.0);
        iss[j]=0;
        issqu[j]=0;
        nmisi[j]=0;
        chispara[j]=0;
        differences[j]=0;
        countdifferences[j]=0;
        avdifferences[j]=0;
        
    }

        
        /*plot distribution*/
        

        
        double sdvElexc=-ElRS*0.2;
        double x=ElRS-2.5*sdvElexc;
        int lenghtintegral=50;
        double dx=5*sdvElexc/lenghtintegral;
        
        double auss=0;
        for(i=0;i<lenghtintegral;i++)
        {
            
            x+=dx;
            auss=0;
            for(j=0;j<ne;j++)
            {
                
                
                if((hetEl[j]>x-.5*dx)&&(hetEl[j]<x+.5*dx)){
                    told=timet;
                    
                    auss+=1.;
                }
                
            }
            

            
            
            
        }

        
        
/*TIME SIMULATION*/
        
        
        
        
        timet=0;
        tbinold=0;
        told=0;
        double ratesing=0;
        double ratesingInh=0;
        double gec=0;
        double gevv=0;
        double gic=0;
        double givv=0;
        double freqsin=20;
        collad=0;

avvre=0;
avvri=0;
sdre=0;
sdri=0;
countrates=0;

        
	/*while (time<tstop) {*/
    tstop=t0+1;
        double timetold=0;
        
    while (timet<tstop) {
        
        if(timet>timetold+1){
        printf("advancelment    %lf sec\n",timet);
            timetold=timet;
        }
        afferent=doubgauss(timet,t0,T1,T2,amp);
    //afferent=amp*sin((time-t0)*freqsin)*heaviside(time-t0);
        
        inputrateonexc=inputrateonexc0+afferent;

        if(timet>told+BIN){
            told=timet;
            
            //fprintf(timecourse,"%lf %lf %lf    %lf\n",timet,(rexc/dimmexc)/BIN,(rinh/diminh)/BIN,inputrateonexc);
            collad=bRS*tauw*((rexc/dimmexc)/BIN);
		        if(timet>2.){
				avvre+=((rexc/dimmexc)/BIN);
				avvri+=((rinh/diminh)/BIN);
				sdre+=((rexc/dimmexc)/BIN)*((rexc/dimmexc)/BIN);
				sdri+=((rinh/diminh)/BIN)*((rinh/diminh)/BIN);
				countrates+=1;
				}
            rexc=0;
            rinh=0;

        }
        
        
	
		timet+=stepp;
	
        /*fprintf(timecourse,"%lf %lf %lf %E %E %E %E %E\n",time,v[dimmexc+1],v[dim-1],GEexc[0],GIexc[0],GEinh[0],GIinh[0],collad);*/
        
       
        

for (i=0; i<dimmexc; i++) {
    if(i==0){
        gevv+=GEexc[0];
        gec+=1;
        givv+=GEinh[0];
        gic+=1;
    }
    
        randnumb=rand()/(RAND_MAX+1.0);
    
        if(randnumb<stepp*inputrateonexc*pconn*dimmexc){
            GEexc[i]+=gee;
           
            }
    
    randnumb=rand()/(RAND_MAX+1.0);
    
    if(randnumb<stepp*inputrateoninh*pconn*diminh){
        GEinh[i]+=gei;
        
    }
    
    
			
            v[i]+=(glRS*(hetEl[i]-v[i])/CmRS+glRS*deltaRS*exp(((v[i]-thr)/(deltaRS)))/CmRS+GEexc[i]*(Ee-v[i])/CmRS+GEinh[i]*(Ei-v[i])/CmRS-ww[i]/CmRS)*stepp*tetaaa(timet-tolsp[i]-Trefr);
    
    
            ww[i]+=(-ww[i]+aRS*(v[i]-hetEl[i]))*stepp/tauw;
            /*ww[i]=collad;*/


    
    
    
            GEexc[i]+=stepp*(-GEexc[i]/tausynexc);
            GEinh[i]+=stepp*(-GEinh[i]/tausyninh);
			phase[i]=(timet-tolsp[i])/(isi[i]);
			
	
		}
       
        
        
			
        
				for (i=0; i<dimmexc; i++) {
				
			vold[i]= v[i];
			if(v[i]>vspike){
                if(i==0){
                ratesing+=1;
                }

                rexc+=1;
			chispara[i]=1;
			isi[i]=timet-tolsp[i];
			if(timet>transientt){

                iss[i]+=isi[i];
                issqu[i]+=isi[i]*isi[i];
                nmisi[i]+=1;


            }
			
			tolsp[i]=timet;
			indicator[i]=timet;
                
            fprintf(rasterexc,"%lf	%d\n",timet,i);
         

						
			rateun[i]+=1;
			ratebinn+=1;
			ratetoc+=1;
            v[i]=vres;
            ww[i]+=bRS;
		
}
}
         
       
       
        
        
        /*INHIBITORY*/
        
        for (i=dimmexc; i<dim; i++) {
            
            randnumb=rand()/(RAND_MAX+1.0);
            
            if(randnumb<stepp*inputrateonexc*pconn*dimmexc){
                GIexc[i-dimmexc]+=gie;
                
            
            }
            
            randnumb=rand()/(RAND_MAX+1.0);
            
            if(randnumb<stepp*inputrateoninh*pconn*diminh){
                GIinh[i-dimmexc]+=gii;
                
            }
            
            
            
            
            
            v[i]+=(glFS*(hetEl[i]-v[i])/CmFS+glFS*deltaFS*exp(((v[i]-thr)/(deltaFS)))/CmFS+GIexc[i-dimmexc]*(Ee-v[i])/CmFS+GIinh[i-dimmexc]*(Ei-v[i])/CmFS)*stepp*tetaaa(timet-tolsp[i]-Trefr);
            
            
            
            
            
            GIexc[i-dimmexc]+=stepp*(-GIexc[i-dimmexc]/tausynexc);
            GIinh[i-dimmexc]+=stepp*(-GIinh[i-dimmexc]/tausyninh);
            phase[i]=(timet-tolsp[i])/(isi[i]);
            
            
        }
        
        
        
        
        
        for (i=dimmexc; i<dim; i++) {
            
            vold[i]= v[i];
            if(v[i]>vspike){
                if(i==dimmexc){
                    ratesingInh+=1;
                }
                 rinh+=1;
                chispara[i]=1;
                isi[i]=timet-tolsp[i];
                if(timet>transientt){
                    
                    iss[i]+=isi[i];
                    issqu[i]+=isi[i]*isi[i];
                    nmisi[i]+=1;
                    
                    
                }
                
                tolsp[i]=timet;
                indicator[i]=timet;
                fprintf(rasterinh,"%lf	%d\n",timet,i);
                
                
                
                rateun[i]+=1;
                ratebinn+=1;
                ratetoc+=1;
                v[i]=vres;
                
            }
        }
        
    

        
        


        
        
        
    
        
        
	for (i=0; i<dim; i++) {
				
		


if((timet-indicator[i]-delay)>=0){
    
   
				indicator[i]=1000000;
                    kee=0;
                    kei=0;
					for (j=0; j<dimmexc; j++) {

							if(i<ne){
							GEexc[j]+=(gij[j][i]);
                                if(gij[j][i]>0){
                                kee+=1;
                                    }
                            }
                        else{
                            GEinh[j]+=(gij[j][i]);
                           
                            if(gij[j][i]>0){
                                kei+=1;
                            }

                            
                        }
                        }
                        kii=0;
                        kie=0;
                        for (j=0; j<diminh; j++) {
                            
                            if(i<ne){
                                GIexc[j]+=(gij[j+dimmexc][i]);
                               
                                if(gij[j][i]>0){
                                    kie+=1;
                                }

                                
                            }
                            else{
                                GIinh[j]+=(gij[j+dimmexc][i]);
                                if(gij[j][i]>0){
                                    kii+=1;
                                }

                               
                                
                            }
                            
                        }
    
  /* printf(" %d  %lf	%lf %lf %lf\n",i,kee,kei,kii,kie);*/
                    }


 }
 
        
        
        
        
        
        
    }




        fclose(timecourse);
        }
     }
    

    
   
}

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran2(long *idum)
{
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	double temp;
	
	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		idum2=(*idum);
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ1;
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1;
	k=idum2/IQ2;
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX




#define MAXIT 100

double rtsafe(void (*funcd)(double, double *, double *), double x1, double x2,
			  double xacc)
{
	
	void nrerror(char error_text[]);
	int j;
	double df,dx,dxold,f,fh,fl;
	double temp,xh,xl,rts;
	
	(*funcd)(x1,&fl,&df);
	(*funcd)(x2,&fh,&df);
	if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0))
	{
		
		return 1000;
		/*nrerror("Root must be bracketed in rtsafe");*/
		
	
	}
	if (fl == 0.0) return x1;
	if (fh == 0.0) return x2;
	if (fl < 0.0) {
		xl=x1;
		xh=x2;
	} else {
		xh=x1;
		xl=x2;
	}
	rts=0.5*(x1+x2);
	dxold=fabs(x2-x1);
	dx=dxold;
	(*funcd)(rts,&f,&df);
	for (j=1;j<=MAXIT;j++) {
		if ((((rts-xh)*df-f)*((rts-xl)*df-f) >= 0.0)
			|| (fabs(2.0*f) > fabs(dxold*df))) {
			dxold=dx;
			dx=0.5*(xh-xl);
			rts=xl+dx;
			if (xl == rts) return rts;
		} else {
			dxold=dx;
			dx=f/df;
			temp=rts;
			rts -= dx;
			if (temp == rts) return rts;
		}
		if (fabs(dx) < xacc) return rts;
		(*funcd)(rts,&f,&df);
		if (f < 0.0)
			xl=rts;
		else
			xh=rts;
	}
	nrerror("Maximum number of iterations exceeded in rtsafe");
	return 0.0;
}
#undef MAXIT









double media( double y[])
{
	double Y=0;
	double c;
	int q;
	for(q=0;q<dim;q++){
		Y=Y + y[q];
	}
	c=Y/dim;
	return c;
}

int delta(int a,int b){
	int c;
	if(a==b){
		return 1;
	}
	else{
		return 0;
	}
}



double par(double v[]){
	double R,A,B,teta;
	int t;
	A=0;
	B=0;
	for(t=0;t<dim;t++){
		teta=2*Pi*v[t];
		A +=sin(teta);
		B +=cos(teta);
	}
	R=sqrt(A*A+B*B);
	return R/dim;
}



double abss(double v){
	if(v<0)
		return -v;
	else
		return v;
}

int abssint(int v){
    if(v<0)
        return -v;
    else
        return v;
}








int sgn(double g){
	if (g>0) {
		return 1;
	}
	else {
		return -1;
	}
	
	
	
	
}




double par1(double v[],int indici[],int n){
	double R,A,B,teta;
	int t;
	
	A=0;
	B=0;
	for(t=dim;t>=dim-n+1;t--){
		teta=2*Pi*v[indici[t]-1];
		A +=sin(teta);
		B +=cos(teta);
	}
	R=sqrt(A*A+B*B);
	return R/dim;
}



double avver(double vec[]){
  
    int t;
    int ll=sizeof(*vec)/sizeof(vec[0]);
    double lld=ll;
    
    double A=0;
  
    for(t=0;t<ll;t++){
     
        A +=vec[t];

    }

    return A/lld;
}







double tetaaa(double juy){
		
if (juy>=0) {
		return 1;
	}
	else {
		return 0;
	}


	}





int scelta(piii){
    double x,randomi;
    
    randomi=rand()/(RAND_MAX+1.0);
    int i,s;
    x=(piii-1)*randomi;
    s=floor(x);
    return s;
    
}

    
double gausdevdet(double ind) {
        double v1,v2,r2,fac;
        
        /*srand(time(NULL));*/
        do {
            
            v1=2.0*(ind)-1.0;
            v2=2.0*(ind)-1.0;
            r2=v1*v1+v2*v2;
        }
        
        while (r2>=1.0 || r2 == 0.0);
        fac=sqrt( -2.0*log(r2)/r2);
        return v2*fac;
}




double gausdev () {
    double v1,v2,r2,fac;
    
    /*srand(time(NULL));*/
    do {
        
        v1=2.0*(rand()/(RAND_MAX+1.0))-1.0;
        v2=2.0*(rand()/(RAND_MAX+1.0))-1.0;
        r2=v1*v1+v2*v2;
        
        
    }
    
    while (r2>=1.0 || r2 == 0.0);
    fac=sqrt( -2.0*log(r2)/r2);
    return v2*fac;
}



double heaviside(double x){
    if(x>=0){
        return 1;
        
    }
    else{
        return 0;
    }
    
}
double doubgauss(double t,double t0,double T1,double T2,double amp){
    return amp*(exp(-((t-t0)*(t-t0))/(2*T1*T1))*heaviside(-(t-t0))+exp(-((t-t0)*(t-t0))/(2*T2*T2))*heaviside(t-t0));
}









