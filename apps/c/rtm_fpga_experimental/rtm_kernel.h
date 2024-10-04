zdfd

#pragma once

void rtm_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, ACC<float>& rho, ACC<float>& mu, 
    ACC<float>& yy_0, ACC<float>& yy_sum_0) {
  float x = 1.0*((float)(idx[0]-nx/2)/nx);
  float y = 1.0*((float)(idx[1]-ny/2)/ny);
  float z = 1.0*((float)(idx[2]-nz/2)/nz);
  //printf("x,y,z = %f %f %f\n",x,y,z);
  const float C = 1.0f;
  const float r0 = 0.001f;
  rho(0,0,0) = 1000.0f; /* density */
  mu(0,0,0) = 0.001f; /* bulk modulus */

  yy_0(0,0,0) = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0); //idx[0] + idx[1] + idx[2];//
  yy_sum_0(0,0,0) = 0;
}



void fd3d_pml_kernel1(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
    const ACC<float>& yy_0, const ACC<float>& yy_1, const ACC<float>& yy_2, const ACC<float>& yy_3, const ACC<float>& yy_4, const ACC<float>& yy_5,
    ACC<float>& dyy_0, ACC<float>& dyy_1, ACC<float>& dyy_2, ACC<float>& dyy_3, ACC<float>& dyy_4, ACC<float>& dyy_5, 
    ACC<float>& sum_0, ACC<float>& sum_1, ACC<float>& sum_2, ACC<float>& sum_3, ACC<float>& sum_4, ACC<float>& sum_5) {
  
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

  float sigma = mu(0,0,0)]/rho(0,0,0)];
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
  
  float px = yy_0(0,0,0);
  float py = yy_1(0,0,0);
  float pz = yy_2(0,0,0);
  
  float vx = yy_3(0,0,0);
  float vy = yy_4(0,0,0);
  float vz = yy_5(0,0,0);
  
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
    pxx += yy_0(i,0,0)*c[i+half];
    pyx += yy_1(i,0,0)*c[i+half];
    pzx += yy_2(i,0,0)*c[i+half];
    
    vxx += yy_3(i,0,0)*c[i+half];
    vyx += yy_4(i,0,0)*c[i+half];
    vzx += yy_5(i,0,0)*c[i+half];
    
    pxy += yy_0(0,i,0)*c[i+half];
    pyy += yy_1(0,i,0)*c[i+half];
    pzy += yy_2(0,i,0)*c[i+half];
    
    vxy += yy_3(0,i,0)*c[i+half];
    vyy += yy_4(0,i,0)*c[i+half];
    vzy += yy_5(0,i,0)*c[i+half];
    
    pxz += yy_0(0,0,i)*c[i+half];
    pyz += yy_1(0,0,i)*c[i+half];
    pzz += yy_2(0,0,i)*c[i+half];
    
    vxz += yy_3(0,0,i)*c[i+half];
    vyz += yy_4(0,0,i)*c[i+half];
    vzz += yy_5(0,0,i)*c[i+half];
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
  
  float ytemp0 =(vxx/rho(0,0,0) - sigmax*px) * *dt;
  float ytemp3 =((pxx+pyx+pxz)*mu(0,0,0) - sigmax*vx)* *dt;
  
  float ytemp1 =(vyy/rho(0,0,0) - sigmay*py)* *dt;
  float ytemp4 =((pxy+pyy+pyz)*mu(0,0,0) - sigmay*vy)* *dt;
  
  float ytemp2 =(vzz/rho(0,0,0) - sigmaz*pz)* *dt;
  float ytemp5 =((pxz+pyz+pzz)*mu(0,0,0) - sigmaz*vz)* *dt;



  dyy_0(0,0,0) = yy_0(0,0,0) + ytemp0* *scale1;
  dyy_3(0,0,0) = yy_3(0,0,0) + ytemp3* *scale1;
  dyy_1(0,0,0) = yy_1(0,0,0) + ytemp1* *scale1;
  dyy_4(0,0,0) = yy_4(0,0,0) + ytemp4* *scale1;
  dyy_2(0,0,0) = yy_2(0,0,0) + ytemp2* *scale1;
  dyy_5(0,0,0) = yy_5(0,0,0) + ytemp5* *scale1;

  sum_0(0,0,0) += ytemp0 * *scale2;
  sum_3(0,0,0) += ytemp3 * *scale2;
  sum_1(0,0,0) += ytemp1 * *scale2;
  sum_4(0,0,0) += ytemp4 * *scale2;
  sum_2(0,0,0) += ytemp2 * *scale2;
  sum_5(0,0,0) += ytemp5 * *scale2;
   
}

void fd3d_pml_kernel2(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
        const ACC<float>& yy_0, const ACC<float>& yy_1, const ACC<float>& yy_2, const ACC<float>& yy_3, const ACC<float>& yy_4, const ACC<float>& yy_5, 
        const ACC<float>& dyyIn_0, const ACC<float>& dyyIn_1, const ACC<float>& dyyIn_2, const ACC<float>& dyyIn_3, const ACC<float>& dyyIn_4, const ACC<float>& dyyIn_5, 
        const ACC<float>& dyyOut_0, const ACC<float>& dyyOut_1, const ACC<float>& dyyOut_2, const ACC<float>& dyyOut_3, const ACC<float>& dyyOut_4, const ACC<float>& dyyOut_5, 
        const ACC<float>& sum_0, const ACC<float>& sum_1, const ACC<float>& sum_2, const ACC<float>& sum_3, const ACC<float>& sum_4, const ACC<float>& sum_5) {
  
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

  float sigma = mu(0,0,0)/rho(0,0,0);
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
  
  float px = dyyIn_0(0,0,0);
  float py = dyyIn_1(0,0,0);
  float pz = dyyIn_2(0,0,0);
  
  float vx = dyyIn_3(0,0,0);
  float vy = dyyIn_4(0,0,0);
  float vz = dyyIn_5(0,0,0);
  
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
    pxx += dyyIn_0(i,0,0)*c[i+half];
    pyx += dyyIn_1(i,0,0)*c[i+half];
    pzx += dyyIn_2(i,0,0)*c[i+half];
    
    vxx += dyyIn_3(i,0,0)*c[i+half];
    vyx += dyyIn_4(i,0,0)*c[i+half];
    vzx += dyyIn_5(i,0,0)*c[i+half];
    
    pxy += dyyIn_0(0,i,0)*c[i+half];
    pyy += dyyIn_1(0,i,0)*c[i+half];
    pzy += dyyIn_2(0,i,0)*c[i+half];
    
    vxy += dyyIn_3(0,i,0)*c[i+half];
    vyy += dyyIn_4(0,i,0)*c[i+half];
    vzy += dyyIn_5(0,i,0)*c[i+half];
    
    pxz += dyyIn_0(0,0,i)*c[i+half];
    pyz += dyyIn_1(0,0,i)*c[i+half];
    pzz += dyyIn_2(0,0,i)*c[i+half];
    
    vxz += dyyIn_3(0,0,i)*c[i+half];
    vyz += dyyIn_4(0,0,i)*c[i+half];
    vzz += dyyIn_5(0,0,i)*c[i+half];
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
  
  float ytemp0 =(vxx/rho(0,0,0) - sigmax*px) * *dt;
  float ytemp3 =((pxx+pyx+pxz)*mu(0,0,0) - sigmax*vx)* *dt;
  
  float ytemp1 =(vyy/rho(0,0,0) - sigmay*py)* *dt;
  float ytemp4 =((pxy+pyy+pyz)*mu(0,0,0) - sigmay*vy)* *dt;
  
  float ytemp2 =(vzz/rho(0,0,0) - sigmaz*pz)* *dt;
  float ytemp5 =((pxz+pyz+pzz)*mu(0,0,0) - sigmaz*vz)* *dt;



  dyyOut_0(0,0,0) = yy_0(0,0,0) + ytemp0* *scale1;
  dyyOut_3(0,0,0) = yy_3(0,0,0) + ytemp3* *scale1;
  dyyOut_1(0,0,0) = yy_1(0,0,0) + ytemp1* *scale1;
  dyyOut_4(0,0,0) = yy_4(0,0,0) + ytemp4* *scale1;
  dyyOut_2(0,0,0) = yy_2(0,0,0) + ytemp2* *scale1;
  dyyOut_5(0,0,0) = yy_5(0,0,0) + ytemp5* *scale1;

  sum_0(0,0,0) += ytemp0 * *scale2;
  sum_3(0,0,0) += ytemp3 * *scale2;
  sum_1(0,0,0) += ytemp1 * *scale2;
  sum_4(0,0,0) += ytemp4 * *scale2;
  sum_2(0,0,0) += ytemp2 * *scale2;
  sum_5(0,0,0) += ytemp5 * *scale2;
   
}

void fd3d_pml_kernel3(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
        const ACC<float>& yy_0, const ACC<float>& yy_1, const ACC<float>& yy_2, const ACC<float>& yy_3, const ACC<float>& yy_4, const ACC<float>& yy_5,
        const ACC<float>& dyyIn_0, const ACC<float>& dyyIn_1, const ACC<float>& dyyIn_2, const ACC<float>& dyyIn_3, const ACC<float>& dyyIn_4, const ACC<float>& dyyIn_5,
        ACC<float>& sum_0, ACC<float>& sum_1, ACC<float>& sum_2, ACC<float>& sum_3, ACC<float>& sum_4, ACC<float>& sum_5) {
  
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

  float sigma = mu(0,0,0)/rho(0,0,0);
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
  
  float px = dyyIn_0(0,0,0);
  float py = dyyIn_1(0,0,0);
  float pz = dyyIn_2(0,0,0);
  
  float vx = dyyIn_3(0,0,0);
  float vy = dyyIn_4(0,0,0);
  float vz = dyyIn_5(0,0,0);
  
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
    pxx += dyyIn_0(i,0,0)*c[i+half];
    pyx += dyyIn_1(i,0,0)*c[i+half];
    pzx += dyyIn_2(i,0,0)*c[i+half];
    
    vxx += dyyIn_3(i,0,0)*c[i+half];
    vyx += dyyIn_4(i,0,0)*c[i+half];
    vzx += dyyIn_5(i,0,0)*c[i+half];
    
    pxy += dyyIn_0(0,i,0)*c[i+half];
    pyy += dyyIn_1(0,i,0)*c[i+half];
    pzy += dyyIn_2(0,i,0)*c[i+half];
    
    vxy += dyyIn_3(0,i,0)*c[i+half];
    vyy += dyyIn_4(0,i,0)*c[i+half];
    vzy += dyyIn_5(0,i,0)*c[i+half];
    
    pxz += dyyIn_0(0,0,i)*c[i+half];
    pyz += dyyIn_1(0,0,i)*c[i+half];
    pzz += dyyIn_2(0,0,i)*c[i+half];
    
    vxz += dyyIn_3(0,0,i)*c[i+half];
    vyz += dyyIn_4(0,0,i)*c[i+half];
    vzz += dyyIn_5(0,0,i)*c[i+half];
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
  
  float ytemp0 =(vxx/rho(0,0,0) - sigmax*px) * *dt;
  float ytemp3 =((pxx+pyx+pxz)*mu(0,0,0) - sigmax*vx)* *dt;
  
  float ytemp1 =(vyy/rho(0,0,0) - sigmay*py)* *dt;
  float ytemp4 =((pxy+pyy+pyz)*mu(0,0,0) - sigmay*vy)* *dt;
  
  float ytemp2 =(vzz/rho(0,0,0) - sigmaz*pz)* *dt;
  float ytemp5 =((pxz+pyz+pzz)*mu(0,0,0) - sigmaz*vz)* *dt;


  yy_0(0,0,0) += sum_0(0,0,0) + ytemp0 * *scale2;
  yy_3(0,0,0) += sum_3(0,0,0) + ytemp3 * *scale2;
  yy_1(0,0,0) += sum_1(0,0,0) + ytemp1 * *scale2;
  yy_4(0,0,0) += sum_4(0,0,0) + ytemp4 * *scale2;
  yy_2(0,0,0) += sum_2(0,0,0) + ytemp2 * *scale2;
  yy_5(0,0,0) += sum_5(0,0,0) + ytemp5 * *scale2;
   
}
