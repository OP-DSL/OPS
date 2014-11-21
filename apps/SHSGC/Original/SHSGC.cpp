
// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#ifndef MAX
	#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
double dx,dy,gam, gam1 ;


void ops_timers_core( double * cpu, double * et )
{
  (void)cpu;
  struct timeval t;

  gettimeofday ( &t, ( struct timezone * ) 0 );
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
}


/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  int nxp, nyp ,xhalo, yhalo,niter;
  double xmin, ymin, xmax, ymax;
  double pl, pr, ul, ur, rhol, rhor, eps, lambda ;

   // Initialisation
  nxp = 204;
  nyp = 5;
  xhalo = 2;
  yhalo = 2;
  xmin = -5.0;
  ymin = 0;
  xmax = 5.0;
  ymax = 0.5;
  dx = (xmax-xmin)/(nxp-(1 + 2*xhalo));
  dy = (ymax-ymin)/(nyp-1); 
  pl = 10.333f;
  pr = 1.0f;
  rhol = 3.857143;
  rhor = 1.0f;
  ul = 2.6293690 ;
  ur = 0.0f;
  gam = 1.4;
  gam1=gam - 1.0;
  eps = 0.2;
  lambda = 5.0;

  double *x = (double *)malloc(nxp*sizeof(double*));
//  double *y = (double *)malloc(nyp*sizeof(double*));

// Conservative variables definition
  double *rho_old = (double *)malloc(nxp*sizeof(double*));
  double *rho_new = (double *)malloc(nxp*sizeof(double*));
  double *rho_res = (double *)malloc(nxp*sizeof(double*));

  double *rhou_old = (double *)malloc(nxp*sizeof(double*));
  double *rhou_new = (double *)malloc(nxp*sizeof(double*));
  double *rhou_res = (double *)malloc(nxp*sizeof(double*));

  double *rhov_old = (double *)malloc(nxp*sizeof(double*));
  double *rhov_new = (double *)malloc(nxp*sizeof(double*));

  double *rhoE_old = (double *)malloc(nxp*sizeof(double*));
  double *rhoE_new = (double *)malloc(nxp*sizeof(double*));
  double *rhoE_res = (double *)malloc(nxp*sizeof(double*));

  // halo variables definition
  double *rho_halo = (double *)malloc(2*xhalo*sizeof(double*));
  double *rhou_halo = (double *)malloc(2*xhalo*sizeof(double*));
  double *rhoE_halo = (double *)malloc(2*xhalo*sizeof(double*));

  double fni, fnim1,fnim2,fnip1,fnip2, deriv, p,dt, totaltime ;

  // rk3 co-efficient's
  double *a1 = (double *)malloc(3*sizeof(double*));
  double *a2 = (double *)malloc(3*sizeof(double*));
  

  // TVD scheme variables
  double r[nxp][3][3], al[nxp][3], alam[nxp][3], gt[nxp][3], eff[nxp][3], s[nxp][3], tht[nxp][3];
  double cmp[nxp][3], cf[nxp][3], ep2[nxp][3];
  double akap2 = 0.40;
  double tvdsmu = 0.25f;
  printf("%lf\t %lf",akap2, tvdsmu);
  
  // Initialize rk3 co-efficient's
  a1[0] = 2.0/3.0;
  a1[1] = 5.0/12.0;
  a1[2] = 3.0/5.0;
  a2[0] = 1.0/4.0;
  a2[1] = 3.0/20.0;
  a2[2] = 3.0/5.0;
  dt=0.0002;
  totaltime =0.0f;
  niter = 9005;
// Initialize with the test case
  for (int i = 0; i < nxp; i++) {
	  x[i] = xmin + (i-2) * dx;

  }
  
  for (int i = 0; i < nxp; i++) {
	if (x[i] >= -4.0){
		rho_new[i] = 1.0 + eps * sin(lambda *x[i] );
		rhou_new[i] = ur * rho_new[i];
		rhoE_new[i] = (pr / gam1) + 0.5 * pow(rhou_new[i],2)/rho_new[i];
	}
	else {
		rho_new[i] = rhol;
		rhou_new[i] = ul * rho_new[i];
		rhoE_new[i] = (pl / gam1) + 0.5 * pow(rhou_new[i],2)/rho_new[i];
	}
	

  
  
	FILE *fp;
  fp = fopen("rhoin.txt", "w");
  for (int i=0; i<nxp; i++) 
	  fprintf(fp, "%3.10lf\n", gam1 * (rhoE_new[i] - 0.5 * rhou_new[i] * rhou_new[i] / rho_new[i] ));
  fclose(fp);
  }
  
 
   // boundary conditions as of no it is just an idea to implement however for the test case the grid points  nx+4 are given as input
//   for (int i = 0; i < 2*xhalo; i++) {
// 	  if(i < xhalo){
// 		  rho_halo[i] = rhol;
// 		  rhou_halo[i] = ul*rho_halo[i];
// 		  rhoE_halo[i] = (pl/gam1) + 0.5 * pow(rhou_halo[i],2)/rho_halo[i];
// 		  //printf("left is %d\n",i);
// 	  }
// 	  else{
// 		  rho_halo[i] = 1.0 + eps * sin(lambda *x[i] );
// 		  rhou_halo[i] = ur*rho_halo[i];
// 		  rhoE_halo[i] = (pr/gam1) + 0.5 * pow(rhou_halo[i],2)/rho_halo[i];
// 		  //printf("right is %d\n",i);
// 	  }
//   }

  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);
  
   
  //main iterative loop
  for (int iter = 0; iter <niter;  iter++){
	
    // Save previous data arguments
    for (int i = 0; i < nxp; i++) {
      rho_old[i]=rho_new[i];
      rhou_old[i]=rhou_new[i];
      rhoE_old[i]=rhoE_new[i];
    }
    
	// rk3 loop
	for (int nrk=0; nrk <3; nrk++){
		// make residue equal to zero
		for (int i = 0; i < nxp; i++) {
			rho_res[i]=0.0;
			rhou_res[i]=0.0;
			rhoE_res[i]=0.0;
		}
		// computations of convective derivatives


		//conv( &nxp, &rho_res, &rhou_res, &rhoE_res, &rho_new, &rhou_new, &rhoE_new)
		// calculate drhou/dx
		double p;
		for (int i=2; i < nxp-2; i++) {
			fni = rhou_new[i];
			fnim1 = rhou_new[i-1];
			fnim2 = rhou_new[i-2];
			fnip1 = rhou_new[i+1];
			fnip2 = rhou_new[i+2];

			deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rho_res[i] = deriv;
		}
		

   
   
  
		// calculate d(rhouu + p)/dx
		for (int i=2; i < nxp-2; i++) {
			// cal pressure 
			fni = rhou_new[i] * rhou_new[i] / rho_new[i] ;
			p = gam1 * (rhoE_new[i] - 0.5 * fni);
			fni = fni + p;
			fnim1 = rhou_new[i-1] * rhou_new[i-1] / rho_new[i-1];
			p = gam1 * (rhoE_new[i-1] - 0.5 * fnim1);
			fnim1 = fnim1 + p;
			fnim2 = rhou_new[i-2] * rhou_new[i-2] / rho_new[i-2];
			p = gam1 * (rhoE_new[i-2] - 0.5 * fnim2);
			fnim2 = fnim2 + p;
			fnip1 = rhou_new[i+1] * rhou_new[i+1] / rho_new[i+1];
			p = gam1 * (rhoE_new[i+1] - 0.5 * fnip1);
			fnip1 = fnip1 + p;
			fnip2 = rhou_new[i+2] * rhou_new[i+2] / rho_new[i+2];
			p = gam1 * (rhoE_new[i+2] - 0.5 * fnip2);
			fnip2 = fnip2 + p;

			deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rhou_res[i] = deriv;
		}


		
		// Energy equation derivative d(rhoE+p)u/dx
		for (int i=2; i < nxp-2; i++) {
			
			fni = rhou_new[i] * rhou_new[i] / rho_new[i] ;
			p = gam1 * (rhoE_new[i] - 0.5 * fni);
			fni = (rhoE_new[i] + p) * rhou_new[i] / rho_new[i] ;
			
			fnim1 = rhou_new[i-1] * rhou_new[i-1] / rho_new[i-1];
			p = gam1 * (rhoE_new[i-1] - 0.5 * fnim1);
			fnim1 = (rhoE_new[i-1] + p) * rhou_new[i-1] / rho_new[i-1];
			
			fnim2 = rhou_new[i-2] * rhou_new[i-2] / rho_new[i-2];
			p = gam1 * (rhoE_new[i-2] - 0.5 * fnim2);
			fnim2 = (rhoE_new[i-2] + p ) * rhou_new[i-2] / rho_new[i-2];
			
			fnip1 = rhou_new[i+1] * rhou_new[i+1] / rho_new[i+1];
			p = gam1 * (rhoE_new[i+1] - 0.5 * fnip1);
			fnip1 = (rhoE_new[i+1] + p) * rhou_new[i+1] / rho_new[i+1];
			
			fnip2 = rhou_new[i+2] * rhou_new[i+2] / rho_new[i+2];
			p = gam1 * (rhoE_new[i+2] - 0.5 * fnip2);
			fnip2 = (rhoE_new[i+2] + p) * rhou_new[i+2] / rho_new[i+2];
			
// 			fni = rhoE_new[i] * gam - 0.5 * gam1 * (rhou_new[i] * rhou_new[i] / rho_new[i]) ; // not reqd for first der
// 			fnim1 = rhoE_new[i-1] * gam - 0.5 * gam1 * (rhou_new[i-1] * rhou_new[i-1] / rho_new[i-1]);
// 			fnim1 = fnim1 * rhou_new[i-1] / rho_new[i-1];
// 			fnim2 = rhoE_new[i-2] * gam - 0.5 * gam1 * (rhou_new[i-2] * rhou_new[i-2] / rho_new[i-2]);
// 			fnim2 = fnim2 * rhou_new[i-2] / rho_new[i-2];
// 			fnip1 = rhoE_new[i+1] * gam - 0.5 * gam1 * (rhou_new[i+1] * rhou_new[i+1] / rho_new[i+1]);
// 			fnip1 = fnip1 * rhou_new[i+1] / rho_new[i+1];
// 			fnip2 = rhoE_new[i+2] * gam - 0.5 * gam1 * (rhou_new[i+2] * rhou_new[i+2] / rho_new[i+2]);
// 			fnip2 = fnip2 * rhou_new[i+2] / rho_new[i+2];

			deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rhoE_res[i] = deriv;
		}
		


		// boundary derivatives

		// update use rk3 co-efficient's

		for (int i=3; i < nxp-2; i++) {
			rho_new[i] = rho_old[i] + dt * a1[nrk] * (-rho_res[i]);
			rhou_new[i] = rhou_old[i] + dt * a1[nrk] * (-rhou_res[i]);
			rhoE_new[i] = rhoE_old[i] + dt * a1[nrk] * (-rhoE_res[i]);
			// update old state
			rho_old[i] = rho_old[i] + dt * a2[nrk] * (-rho_res[i]);
			rhou_old[i] = rhou_old[i] + dt * a2[nrk] * (-rhou_res[i]);
			rhoE_old[i] = rhoE_old[i] + dt * a2[nrk] * (-rhoE_res[i]);
		}
    
	}

	// TVD scheme

// 	// Riemann invariants
	for (int i=0; i < nxp-1; i++) {
		double rl, rr, rho, leftu, rightu, u, hl, hr, h, Vsq, csq, c, g;
		double dw1, dw2, dw3, delpc2, rdeluc;
		double ri[3][3];

		rl = sqrt(rho_new[i]);
		rr = sqrt(rho_new[i+1]);
		rho = rl + rr;
		u = ((rhou_new[i] / rl) + (rhou_new[i+1] / rr)) / rho ;
		fni = rhou_new[i] * rhou_new[i] / rho_new[i] ;
		p = gam1 * (rhoE_new[i] - 0.5 * fni);
		hl = (rhoE_new[i] + p)  / rl ;
		 //= (rhoE_new[i] * gam - 0.5 * gam1 * (rhou_new[i] * rhou_new[i] / rho_new[i])) / rl;
		//hr = (rhoE_new[i+1] * gam - 0.5 * gam1 * (rhou_new[i+1] * rhou_new[i+1] / rho_new[i+1])) / rr;
		fni = rhou_new[i+1] * rhou_new[i+1] / rho_new[i+1] ;
		p = gam1 * (rhoE_new[i+1] - 0.5 * fni);
		hr = (rhoE_new[i+1] + p)  / rr ;
		h = (hl + hr)/rho;
		Vsq = u*u;
		csq = gam1 * (h - 0.5 * Vsq);
		g = gam1 / csq;
		c = sqrt(csq);
		alam[i][0] = u - c;
		alam[i][1] = u;
		alam[i][2] = u + c;

		r[i][0][0] = 1.0;
		r[i][0][1] = 1.0;
		r[i][0][2] = 1.0;

		r[i][1][0] = u - c;
		r[i][1][1] = u;
		r[i][1][2] = u + c;

		r[i][2][0] = h - u * c;
		r[i][2][1] = 0.5 * Vsq;
		r[i][2][2] = h + u * c;

		for (int m=0; m<3; m++)
			for (int n=0; n<3; n++) 
				r[i][m][n] = r[i][m][n] / csq;
		dw1 = rho_new[i+1] - rho_new[i]; 
		dw2 = rhou_new[i+1] - rhou_new[i]; 
		dw3 = rhoE_new[i+1] - rhoE_new[i]; 
		
		
		delpc2 = gam1 * ( dw3 + 0.50 * Vsq * dw1  - u * dw2) / csq;
		rdeluc = ( dw2 - u * dw1) / c ;
		al[i][0] = 0.5 * (delpc2 - rdeluc);
		al[i][1] = dw1 - delpc2 ;
		al[i][2] = 0.5 * ( delpc2 + rdeluc );
// 		
// 		ri[0][0] = 0.5 * (0.50 *g * Vsq + u /c);
// 		ri[0][1] = -0.5 * (g * u + 1.0/c);
// 		ri[0][2] = 0.50 * g;
// 
// 		ri[1][0] = 1.00 - 0.5 * g * Vsq;
// 		ri[1][1] = g * u;
// 		ri[1][2] = -g;
// 
// 		ri[2][0] = 0.50 * (0.50 * g * Vsq - u /c);
// 		ri[2][1] = -0.50 * (g * u - 1.00 / c);
// 		ri[2][2] = 0.50 * g;
// // 		
// 		al[i][0] = 0.00;
//         al[i][0] = al[i][0] + ri[0][0] * dw1 + ri[0][1] * dw2 + ri[0][2] * dw3 ;
// 		
// 		al[i][1] = 0.00;
//         al[i][1] = al[i][1] + ri[1][0] * dw1 + ri[1][1] * dw2 + ri[1][2] * dw3;
// 		
// 		al[i][2] = 0.00;
//         al[i][2] = al[i][2] + ri[2][0] * dw1 + ri[2][1] * dw2 + ri[2][2] * dw3;
      
		
 		for (int m=0; m<3; m++) 
			al[i][m] = al[i][m] * csq;
	}
	
	

	
	
	double del2 = 1e-8;
	
	// limiter function
	double aalm, aal, all, ar, gtt;
	for (int i=1; i < nxp; i++) {
		for (int m=0; m < 3 ;m++) {
			aalm = fabs(al[i-1][m]);
			aal = fabs(al[i][m]);
			tht[i][m] = fabs (aal - aalm) / (aal + aalm + del2);
			all = al[i-1][m];
			ar = al[i][m];
			gtt = all * ( ar * ar + del2 ) + ar * (all * all + del2);
			gt[i][m]= gtt / (ar * ar + all * all + 2.00 * del2);
		}
	}
   
  
	double maxim;
	// Second order tvd dissipation
	for (int i=0; i < nxp-1; i++) {
		for (int m=0; m < 3 ;m++) {
			if (tht[i][m] > tht[i+1][m]) 
				maxim = tht[i][m];
			else
				maxim = tht[i+1][m];
			ep2[i][m] = akap2 * maxim;
		}
	}
	 
  
	// vars
	double  anu, aaa, ga, qf, con, ww;
	con = pow (tvdsmu,2.f);
	for (int i=0; i < nxp-1; i++) {
		for (int m=0; m < 3 ;m++) {
			anu = alam[i][m];
			aaa = al[i][m];
			ga = aaa * ( gt[i+1][m] - gt[i][m]) / (pow(aaa,2.f) + del2);
			qf = sqrt ( con + pow(anu,2.f));
			cmp[i][m] = 0.50 * qf;
			ww = anu + cmp[i][m] * ga; 
			qf = sqrt(con + pow(ww,2.f));
			cf[i][m] = qf;
		}
	}
	
	
  
  
	// cal upwind eff
	double e1, e2, e3;
	for (int i=0; i < nxp-1; i++) {
		e1 = (cmp[i][0] * (gt[i][0] + gt[i+1][0]) - cf[i][0] * al[i][0]) * ep2[i][0];
		e2 = (cmp[i][1] * (gt[i][1] + gt[i+1][1]) - cf[i][1] * al[i][1]) * ep2[i][1];
		e3 = (cmp[i][2] * (gt[i][2] + gt[i+1][2]) - cf[i][2] * al[i][2]) * ep2[i][2];
		
		eff[i][0]=e1 * r[i][0][0] + e2 * r[i][0][1] + e3 * r[i][0][2];
		eff[i][1]=e1 * r[i][1][0] + e2 * r[i][1][1] + e3 * r[i][1][2];
		eff[i][2]=e1 * r[i][2][0] + e2 * r[i][2][1] + e3 * r[i][2][2];
	}
 
	
	//fact
	double fact;
	for (int i=1; i < nxp; i++) {
		for (int m=0; m < 3 ;m++) {
			fact= 0.50 * dt / dx ;
			s[i][m] = -fact * (eff[i][m] - eff[i-1][m]);
		}
	}
  
  
  
	// update loop
	for (int i=3; i < nxp-3; i++) {
		rho_new[i] = rho_new[i] + s[i][0];
		rhou_new[i] = rhou_new[i] + s[i][1];
		rhoE_new[i] = rhoE_new[i] + s[i][2];
	}
	
	
	totaltime = totaltime + dt;
	printf("%d \t %f\n", iter, totaltime);
  }
  
  ops_timers_core(&ct1, &et1);
  printf("\nOriginal Application Total Wall time %lf\n",et1-et0);
  
  FILE *test_fp;
  test_fp = fopen("shsgc.txt", "w");
  //for (int i=0; i<nxp; i++) 
  //  fprintf(test_fp, "%3.10lf\n",x[i]);
  for (int i=0; i<nxp; i++) 
    fprintf(test_fp, "%3.10lf\n",rho_new[i]);
  //for (int i=0; i<nxp; i++) 
  //  fprintf(test_fp, "%3.10lf\n",rhou_new[i]);
  /*for (int i=0; i<nxp; i++) {
    for (int j = 0; j<3;j++)
      fprintf(test_fp, "%3.10lf ",alam[i][j]);
    fprintf(test_fp, "\n");
  }*/
  /*or (int i=0; i<nxp; i++) {
    for (int j = 0; j<3;j++)
      for (int k = 0; k<3;k++)
        fprintf(test_fp, "%3.10lf ",r[i][j][k]);
    fprintf(test_fp, "\n");
  }*/
  /*for (int i=0; i<nxp; i++) {
    for (int j = 0; j<3;j++)
      //fprintf(test_fp, "%3.10lf ",ep2[i][j]);
      fprintf(test_fp, "%3.10lf ",s[i][j]);
    fprintf(test_fp, "\n");
  }*/
  fclose(test_fp);
  exit(0);
  
  
  FILE *fp;
  fp = fopen("rho.txt", "w");
  for (int i=0; i<nxp; i++) 
	  fprintf(fp, "%lf\n",rho_new[i]);
  fclose(fp);
  
  fp = fopen("U.txt", "w");
  for (int i=0; i<nxp; i++) 
	  fprintf(fp, "%lf\n",rhou_new[i]/rho_new[i]);
  fclose(fp);
  double readvar = 0.0;
  double rms = 0.0;
  fp = fopen("Rho", "r");
  for (int i=0; i<nxp; i++){
	  fscanf(fp,"%lf",&readvar);
	  rms = rms + pow ((rho_new[i] - readvar), 2);
  }
  printf("\nthe RMS between C and Fortran is %lf\n" , sqrt(rms)/nxp);
  // end time loop
}