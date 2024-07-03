#pragma once

void rtm_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, float *rho, float *mu, float *yy, float* yy_sum) {
  float x = 1.0*((float)(idx[0]-nx/2)/nx);
  float y = 1.0*((float)(idx[1]-ny/2)/ny);
  float z = 1.0*((float)(idx[2]-nz/2)/nz);
  //printf("x,y,z = %f %f %f\n",x,y,z);
  const float C = 1.0f;
  const float r0 = 0.001f;
  rho[OPS_ACC4(0,0,0)] = 1000.0f; /* density */
  mu[OPS_ACC5(0,0,0)] = 0.001f; /* bulk modulus */

  yy[OPS_ACC_MD6(0,0,0,0)] = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0); //idx[0] + idx[1] + idx[2];//
  yy_sum[OPS_ACC_MD7(0,0,0,0)] = 0;


}



void fd3d_pml_kernel1(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const float *rho, const float *mu, const float* yy, float* dyy, float* sum) {
  
// #include "../coeffs/coeffs8.h"
//  float* c = &coeffs[half+half*(order+1)];
  const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
  float invdx = 1.0 / dx;
  float invdy = 1.0 / dy;
  float invdz = 1.0 / dz;
  int xbeg=half;
  int xend=nx-half;
  int ybeg=half;
  int yend=ny-half;
  int zbeg=half;
  int zend=nz-half;
  int xpmlbeg=xbeg+pml_width;
  int ypmlbeg=ybeg+pml_width;
  int zpmlbeg=zbeg+pml_width;
  int xpmlend=xend-pml_width;
  int ypmlend=yend-pml_width;
  int zpmlend=zend-pml_width;

  float sigma = mu[OPS_ACC5(0,0,0)]/rho[OPS_ACC4(0,0,0)];
  float sigmax=0.0;
  float sigmay=0.0;
  float sigmaz=0.0;
  if(idx[0]<=xbeg+pml_width){
    sigmax = (xbeg+pml_width-idx[0])*sigma * 0.1f;///pml_width;
  }
  if(idx[0]>=xend-pml_width){
    sigmax=(idx[0]-(xend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[1]<=ybeg+pml_width){
    sigmay=(ybeg+pml_width-idx[1])*sigma * 0.1f;///pml_width;
  }
  if(idx[1]>=yend-pml_width){
    sigmay=(idx[1]-(yend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[2]<=zbeg+pml_width){
    sigmaz=(zbeg+pml_width-idx[2])*sigma * 0.1f;///pml_width;
  }
  if(idx[2]>=zend-pml_width){
    sigmaz=(idx[2]-(zend-pml_width))*sigma * 0.1f;///pml_width;
  }

					//sigmax=0.0;
					//sigmay=0.0;
  
  float px = yy[OPS_ACC_MD6(0,0,0,0)];
  float py = yy[OPS_ACC_MD6(1,0,0,0)];
  float pz = yy[OPS_ACC_MD6(2,0,0,0)];
  
  float vx = yy[OPS_ACC_MD6(3,0,0,0)];
  float vy = yy[OPS_ACC_MD6(4,0,0,0)];
  float vz = yy[OPS_ACC_MD6(5,0,0,0)];
  
  float vxx=0.0;
  float vxy=0.0;
  float vxz=0.0;
  
  float vyx=0.0;
  float vyy=0.0;
  float vyz=0.0;

  float vzx=0.0;
  float vzy=0.0;
  float vzz=0.0;
  
  float pxx=0.0;
  float pxy=0.0;
  float pxz=0.0;
  
  float pyx=0.0;
  float pyy=0.0;
  float pyz=0.0;

  float pzx=0.0;
  float pzy=0.0;
  float pzz=0.0;

  for(int i=-half;i<=half;i++){
    pxx += yy[OPS_ACC_MD6(0,i,0,0)]*c[i+half];
    pyx += yy[OPS_ACC_MD6(1,i,0,0)]*c[i+half];
    pzx += yy[OPS_ACC_MD6(2,i,0,0)]*c[i+half];
    
    vxx += yy[OPS_ACC_MD6(3,i,0,0)]*c[i+half];
    vyx += yy[OPS_ACC_MD6(4,i,0,0)]*c[i+half];
    vzx += yy[OPS_ACC_MD6(5,i,0,0)]*c[i+half];
    
    pxy += yy[OPS_ACC_MD6(0,0,i,0)]*c[i+half];
    pyy += yy[OPS_ACC_MD6(1,0,i,0)]*c[i+half];
    pzy += yy[OPS_ACC_MD6(2,0,i,0)]*c[i+half];
    
    vxy += yy[OPS_ACC_MD6(3,0,i,0)]*c[i+half];
    vyy += yy[OPS_ACC_MD6(4,0,i,0)]*c[i+half];
    vzy += yy[OPS_ACC_MD6(5,0,i,0)]*c[i+half];
    
    pxz += yy[OPS_ACC_MD6(0,0,0,i)]*c[i+half];
    pyz += yy[OPS_ACC_MD6(1,0,0,i)]*c[i+half];
    pzz += yy[OPS_ACC_MD6(2,0,0,i)]*c[i+half];
    
    vxz += yy[OPS_ACC_MD6(3,0,0,i)]*c[i+half];
    vyz += yy[OPS_ACC_MD6(4,0,0,i)]*c[i+half];
    vzz += yy[OPS_ACC_MD6(5,0,0,i)]*c[i+half];
  }


  pxx *= invdx;
  pyx *= invdx;
  pzx *= invdx;

  vxx *= invdx;
  vyx *= invdx;
  vzx *= invdx;

  pxy *= invdy;
  pyy *= invdy;
  pzy *= invdy;

  vxy *= invdy;
  vyy *= invdy;
  vzy *= invdy;

  pxz *= invdz;
  pyz *= invdz;
  pzz *= invdz;

  vxz *= invdz;
  vyz *= invdz;
  vzz *= invdz;
  
  float ytemp0 =(vxx/rho[OPS_ACC4(0,0,0)] - sigmax*px) * *dt;
  float ytemp3 =((pxx+pyx+pxz)*mu[OPS_ACC5(0,0,0)] - sigmax*vx)* *dt;
  
  float ytemp1 =(vyy/rho[OPS_ACC4(0,0,0)] - sigmay*py)* *dt;
  float ytemp4 =((pxy+pyy+pyz)*mu[OPS_ACC5(0,0,0)] - sigmay*vy)* *dt;
  
  float ytemp2 =(vzz/rho[OPS_ACC4(0,0,0)] - sigmaz*pz)* *dt;
  float ytemp5 =((pxz+pyz+pzz)*mu[OPS_ACC5(0,0,0)] - sigmaz*vz)* *dt;



  dyy[OPS_ACC_MD7(0,0,0,0)] = yy[OPS_ACC_MD6(0,0,0,0)] + ytemp0* *scale1;
  dyy[OPS_ACC_MD7(3,0,0,0)] = yy[OPS_ACC_MD6(3,0,0,0)] + ytemp3* *scale1;
  dyy[OPS_ACC_MD7(1,0,0,0)] = yy[OPS_ACC_MD6(1,0,0,0)] + ytemp1* *scale1;
  dyy[OPS_ACC_MD7(4,0,0,0)] = yy[OPS_ACC_MD6(4,0,0,0)] + ytemp4* *scale1;
  dyy[OPS_ACC_MD7(2,0,0,0)] = yy[OPS_ACC_MD6(2,0,0,0)] + ytemp2* *scale1;
  dyy[OPS_ACC_MD7(5,0,0,0)] = yy[OPS_ACC_MD6(5,0,0,0)] + ytemp5* *scale1;

  sum[OPS_ACC_MD8(0,0,0,0)] += ytemp0 * *scale2;
  sum[OPS_ACC_MD8(3,0,0,0)] += ytemp3 * *scale2;
  sum[OPS_ACC_MD8(1,0,0,0)] += ytemp1 * *scale2;
  sum[OPS_ACC_MD8(4,0,0,0)] += ytemp4 * *scale2;
  sum[OPS_ACC_MD8(2,0,0,0)] += ytemp2 * *scale2;
  sum[OPS_ACC_MD8(5,0,0,0)] += ytemp5 * *scale2;
   
}

void fd3d_pml_kernel2(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const float *rho, const float *mu, const float* yy, const float* dyyIn,  float* dyyOut, float* sum) {
  
// #include "../coeffs/coeffs8.h"
//  float* c = &coeffs[half+half*(order+1)];
  const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
  float invdx = 1.0 / dx;
  float invdy = 1.0 / dy;
  float invdz = 1.0 / dz;
  int xbeg=half;
  int xend=nx-half;
  int ybeg=half;
  int yend=ny-half;
  int zbeg=half;
  int zend=nz-half;
  int xpmlbeg=xbeg+pml_width;
  int ypmlbeg=ybeg+pml_width;
  int zpmlbeg=zbeg+pml_width;
  int xpmlend=xend-pml_width;
  int ypmlend=yend-pml_width;
  int zpmlend=zend-pml_width;

  float sigma = mu[OPS_ACC5(0,0,0)]/rho[OPS_ACC4(0,0,0)];
  float sigmax=0.0;
  float sigmay=0.0;
  float sigmaz=0.0;
  if(idx[0]<=xbeg+pml_width){
    sigmax = (xbeg+pml_width-idx[0])*sigma * 0.1f;///pml_width;
  }
  if(idx[0]>=xend-pml_width){
    sigmax=(idx[0]-(xend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[1]<=ybeg+pml_width){
    sigmay=(ybeg+pml_width-idx[1])*sigma * 0.1f;///pml_width;
  }
  if(idx[1]>=yend-pml_width){
    sigmay=(idx[1]-(yend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[2]<=zbeg+pml_width){
    sigmaz=(zbeg+pml_width-idx[2])*sigma * 0.1f;///pml_width;
  }
  if(idx[2]>=zend-pml_width){
    sigmaz=(idx[2]-(zend-pml_width))*sigma * 0.1f;///pml_width;
  }

          //sigmax=0.0;
          //sigmay=0.0;
  
  float px = dyyIn[OPS_ACC_MD7(0,0,0,0)];
  float py = dyyIn[OPS_ACC_MD7(1,0,0,0)];
  float pz = dyyIn[OPS_ACC_MD7(2,0,0,0)];
  
  float vx = dyyIn[OPS_ACC_MD7(3,0,0,0)];
  float vy = dyyIn[OPS_ACC_MD7(4,0,0,0)];
  float vz = dyyIn[OPS_ACC_MD7(5,0,0,0)];
  
  float vxx=0.0;
  float vxy=0.0;
  float vxz=0.0;
  
  float vyx=0.0;
  float vyy=0.0;
  float vyz=0.0;

  float vzx=0.0;
  float vzy=0.0;
  float vzz=0.0;
  
  float pxx=0.0;
  float pxy=0.0;
  float pxz=0.0;
  
  float pyx=0.0;
  float pyy=0.0;
  float pyz=0.0;

  float pzx=0.0;
  float pzy=0.0;
  float pzz=0.0;

  for(int i=-half;i<=half;i++){
    pxx += dyyIn[OPS_ACC_MD7(0,i,0,0)]*c[i+half];
    pyx += dyyIn[OPS_ACC_MD7(1,i,0,0)]*c[i+half];
    pzx += dyyIn[OPS_ACC_MD7(2,i,0,0)]*c[i+half];
    
    vxx += dyyIn[OPS_ACC_MD7(3,i,0,0)]*c[i+half];
    vyx += dyyIn[OPS_ACC_MD7(4,i,0,0)]*c[i+half];
    vzx += dyyIn[OPS_ACC_MD7(5,i,0,0)]*c[i+half];
    
    pxy += dyyIn[OPS_ACC_MD7(0,0,i,0)]*c[i+half];
    pyy += dyyIn[OPS_ACC_MD7(1,0,i,0)]*c[i+half];
    pzy += dyyIn[OPS_ACC_MD7(2,0,i,0)]*c[i+half];
    
    vxy += dyyIn[OPS_ACC_MD7(3,0,i,0)]*c[i+half];
    vyy += dyyIn[OPS_ACC_MD7(4,0,i,0)]*c[i+half];
    vzy += dyyIn[OPS_ACC_MD7(5,0,i,0)]*c[i+half];
    
    pxz += dyyIn[OPS_ACC_MD7(0,0,0,i)]*c[i+half];
    pyz += dyyIn[OPS_ACC_MD7(1,0,0,i)]*c[i+half];
    pzz += dyyIn[OPS_ACC_MD7(2,0,0,i)]*c[i+half];
    
    vxz += dyyIn[OPS_ACC_MD7(3,0,0,i)]*c[i+half];
    vyz += dyyIn[OPS_ACC_MD7(4,0,0,i)]*c[i+half];
    vzz += dyyIn[OPS_ACC_MD7(5,0,0,i)]*c[i+half];
  }


  pxx *= invdx;
  pyx *= invdx;
  pzx *= invdx;

  vxx *= invdx;
  vyx *= invdx;
  vzx *= invdx;

  pxy *= invdy;
  pyy *= invdy;
  pzy *= invdy;

  vxy *= invdy;
  vyy *= invdy;
  vzy *= invdy;

  pxz *= invdz;
  pyz *= invdz;
  pzz *= invdz;

  vxz *= invdz;
  vyz *= invdz;
  vzz *= invdz;
  
  float ytemp0 =(vxx/rho[OPS_ACC4(0,0,0)] - sigmax*px) * *dt;
  float ytemp3 =((pxx+pyx+pxz)*mu[OPS_ACC5(0,0,0)] - sigmax*vx)* *dt;
  
  float ytemp1 =(vyy/rho[OPS_ACC4(0,0,0)] - sigmay*py)* *dt;
  float ytemp4 =((pxy+pyy+pyz)*mu[OPS_ACC5(0,0,0)] - sigmay*vy)* *dt;
  
  float ytemp2 =(vzz/rho[OPS_ACC4(0,0,0)] - sigmaz*pz)* *dt;
  float ytemp5 =((pxz+pyz+pzz)*mu[OPS_ACC5(0,0,0)] - sigmaz*vz)* *dt;



  dyyOut[OPS_ACC_MD8(0,0,0,0)] = yy[OPS_ACC_MD6(0,0,0,0)] + ytemp0* *scale1;
  dyyOut[OPS_ACC_MD8(3,0,0,0)] = yy[OPS_ACC_MD6(3,0,0,0)] + ytemp3* *scale1;
  dyyOut[OPS_ACC_MD8(1,0,0,0)] = yy[OPS_ACC_MD6(1,0,0,0)] + ytemp1* *scale1;
  dyyOut[OPS_ACC_MD8(4,0,0,0)] = yy[OPS_ACC_MD6(4,0,0,0)] + ytemp4* *scale1;
  dyyOut[OPS_ACC_MD8(2,0,0,0)] = yy[OPS_ACC_MD6(2,0,0,0)] + ytemp2* *scale1;
  dyyOut[OPS_ACC_MD8(5,0,0,0)] = yy[OPS_ACC_MD6(5,0,0,0)] + ytemp5* *scale1;

  sum[OPS_ACC_MD9(0,0,0,0)] += ytemp0 * *scale2;
  sum[OPS_ACC_MD9(3,0,0,0)] += ytemp3 * *scale2;
  sum[OPS_ACC_MD9(1,0,0,0)] += ytemp1 * *scale2;
  sum[OPS_ACC_MD9(4,0,0,0)] += ytemp4 * *scale2;
  sum[OPS_ACC_MD9(2,0,0,0)] += ytemp2 * *scale2;
  sum[OPS_ACC_MD9(5,0,0,0)] += ytemp5 * *scale2;
   
}

void fd3d_pml_kernel3(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const float *rho, const float *mu, const float* yy, const float* dyyIn, float* sum) {
  
// #include "../coeffs/coeffs8.h"
//  float* c = &coeffs[half+half*(order+1)];
  const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
  float invdx = 1.0 / dx;
  float invdy = 1.0 / dy;
  float invdz = 1.0 / dz;
  int xbeg=half;
  int xend=nx-half;
  int ybeg=half;
  int yend=ny-half;
  int zbeg=half;
  int zend=nz-half;
  int xpmlbeg=xbeg+pml_width;
  int ypmlbeg=ybeg+pml_width;
  int zpmlbeg=zbeg+pml_width;
  int xpmlend=xend-pml_width;
  int ypmlend=yend-pml_width;
  int zpmlend=zend-pml_width;

  float sigma = mu[OPS_ACC5(0,0,0)]/rho[OPS_ACC4(0,0,0)];
  float sigmax=0.0;
  float sigmay=0.0;
  float sigmaz=0.0;
  if(idx[0]<=xbeg+pml_width){
    sigmax = (xbeg+pml_width-idx[0])*sigma * 0.1f;///pml_width;
  }
  if(idx[0]>=xend-pml_width){
    sigmax=(idx[0]-(xend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[1]<=ybeg+pml_width){
    sigmay=(ybeg+pml_width-idx[1])*sigma * 0.1f;///pml_width;
  }
  if(idx[1]>=yend-pml_width){
    sigmay=(idx[1]-(yend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[2]<=zbeg+pml_width){
    sigmaz=(zbeg+pml_width-idx[2])*sigma * 0.1f;///pml_width;
  }
  if(idx[2]>=zend-pml_width){
    sigmaz=(idx[2]-(zend-pml_width))*sigma * 0.1f;///pml_width;
  }

          //sigmax=0.0;
          //sigmay=0.0;
  
  float px = dyyIn[OPS_ACC_MD7(0,0,0,0)];
  float py = dyyIn[OPS_ACC_MD7(1,0,0,0)];
  float pz = dyyIn[OPS_ACC_MD7(2,0,0,0)];
  
  float vx = dyyIn[OPS_ACC_MD7(3,0,0,0)];
  float vy = dyyIn[OPS_ACC_MD7(4,0,0,0)];
  float vz = dyyIn[OPS_ACC_MD7(5,0,0,0)];
  
  float vxx=0.0;
  float vxy=0.0;
  float vxz=0.0;
  
  float vyx=0.0;
  float vyy=0.0;
  float vyz=0.0;

  float vzx=0.0;
  float vzy=0.0;
  float vzz=0.0;
  
  float pxx=0.0;
  float pxy=0.0;
  float pxz=0.0;
  
  float pyx=0.0;
  float pyy=0.0;
  float pyz=0.0;

  float pzx=0.0;
  float pzy=0.0;
  float pzz=0.0;

  for(int i=-half;i<=half;i++){
    pxx += dyyIn[OPS_ACC_MD7(0,i,0,0)]*c[i+half];
    pyx += dyyIn[OPS_ACC_MD7(1,i,0,0)]*c[i+half];
    pzx += dyyIn[OPS_ACC_MD7(2,i,0,0)]*c[i+half];
    
    vxx += dyyIn[OPS_ACC_MD7(3,i,0,0)]*c[i+half];
    vyx += dyyIn[OPS_ACC_MD7(4,i,0,0)]*c[i+half];
    vzx += dyyIn[OPS_ACC_MD7(5,i,0,0)]*c[i+half];
    
    pxy += dyyIn[OPS_ACC_MD7(0,0,i,0)]*c[i+half];
    pyy += dyyIn[OPS_ACC_MD7(1,0,i,0)]*c[i+half];
    pzy += dyyIn[OPS_ACC_MD7(2,0,i,0)]*c[i+half];
    
    vxy += dyyIn[OPS_ACC_MD7(3,0,i,0)]*c[i+half];
    vyy += dyyIn[OPS_ACC_MD7(4,0,i,0)]*c[i+half];
    vzy += dyyIn[OPS_ACC_MD7(5,0,i,0)]*c[i+half];
    
    pxz += dyyIn[OPS_ACC_MD7(0,0,0,i)]*c[i+half];
    pyz += dyyIn[OPS_ACC_MD7(1,0,0,i)]*c[i+half];
    pzz += dyyIn[OPS_ACC_MD7(2,0,0,i)]*c[i+half];
    
    vxz += dyyIn[OPS_ACC_MD7(3,0,0,i)]*c[i+half];
    vyz += dyyIn[OPS_ACC_MD7(4,0,0,i)]*c[i+half];
    vzz += dyyIn[OPS_ACC_MD7(5,0,0,i)]*c[i+half];
  }


  pxx *= invdx;
  pyx *= invdx;
  pzx *= invdx;

  vxx *= invdx;
  vyx *= invdx;
  vzx *= invdx;

  pxy *= invdy;
  pyy *= invdy;
  pzy *= invdy;

  vxy *= invdy;
  vyy *= invdy;
  vzy *= invdy;

  pxz *= invdz;
  pyz *= invdz;
  pzz *= invdz;

  vxz *= invdz;
  vyz *= invdz;
  vzz *= invdz;
  
  float ytemp0 =(vxx/rho[OPS_ACC4(0,0,0)] - sigmax*px) * *dt;
  float ytemp3 =((pxx+pyx+pxz)*mu[OPS_ACC5(0,0,0)] - sigmax*vx)* *dt;
  
  float ytemp1 =(vyy/rho[OPS_ACC4(0,0,0)] - sigmay*py)* *dt;
  float ytemp4 =((pxy+pyy+pyz)*mu[OPS_ACC5(0,0,0)] - sigmay*vy)* *dt;
  
  float ytemp2 =(vzz/rho[OPS_ACC4(0,0,0)] - sigmaz*pz)* *dt;
  float ytemp5 =((pxz+pyz+pzz)*mu[OPS_ACC5(0,0,0)] - sigmaz*vz)* *dt;


  yy[OPS_ACC_MD6(0,0,0,0)] += sum[OPS_ACC_MD8(0,0,0,0)] + ytemp0 * *scale2;
  yy[OPS_ACC_MD6(3,0,0,0)] += sum[OPS_ACC_MD8(3,0,0,0)] + ytemp3 * *scale2;
  yy[OPS_ACC_MD6(1,0,0,0)] += sum[OPS_ACC_MD8(1,0,0,0)] + ytemp1 * *scale2;
  yy[OPS_ACC_MD6(4,0,0,0)] += sum[OPS_ACC_MD8(4,0,0,0)] + ytemp4 * *scale2;
  yy[OPS_ACC_MD6(2,0,0,0)] += sum[OPS_ACC_MD8(2,0,0,0)] + ytemp2 * *scale2;
  yy[OPS_ACC_MD6(5,0,0,0)] += sum[OPS_ACC_MD8(5,0,0,0)] + ytemp5 * *scale2;
   
}
