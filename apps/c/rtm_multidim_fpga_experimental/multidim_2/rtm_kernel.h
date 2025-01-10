
#pragma once

void rtm_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, ACC<float>& rho, ACC<float>& mu, 
    ACC<float>& yy) {
    float x = 1.0*((float)(idx[0]-nx/2)/nx);
    float y = 1.0*((float)(idx[1]-ny/2)/ny);
    float z = 1.0*((float)(idx[2]-nz/2)/nz);
    //printf("x,y,z = %f %f %f\n",x,y,z);
    const float C = 1.0f;
    const float r0 = 0.001f;
    rho(0,0,0) = 1000.0f; /* density */
    mu(0,0,0) = 0.001f; /* bulk modulus */

    float val = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0); //idx[0] + idx[1] + idx[2];//
    yy(0,0,0,0) = val; 
    yy(1,0,0,0) = val;
}

void kernel_copy(const ACC<float> &in, ACC<float> &out) {
  out(0,0,0) = in(0,0,0);
}

void kernel_copy_d2(const ACC<float> &in, ACC<float> &out) {
  out(0,0,0,0) = in(0,0,0,0);
  out(1,0,0,0) = in(1,0,0,0);
}

void fd3d_pml_kernel1(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
    const ACC<float>& yy_0_1, const ACC<float>& yy_2_3, const ACC<float>& yy_4_5, 
    ACC<float>& dyy_0_1, ACC<float>& dyy_2_3, ACC<float>& dyy_4_5,
    ACC<float>& sum_0_1, ACC<float>& sum_2_3, ACC<float>& sum_4_5) {
    
    // #include "../coeffs/coeffs8.h"
    //  float* c = &coeffs[half_order+half_order*(order+1)];
    const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
    float invdx = 1.0 / dx;
    float invdy = 1.0 / dy;
    float invdz = 1.0 / dz;
    int xbeg=half_order;
    int xend=nx-half_order;
    int ybeg=half_order;
    int yend=ny-half_order;
    int zbeg=half_order;
    int zend=nz-half_order;
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
    
    float px = yy_0_1(0,0,0,0);
    float py = yy_0_1(1,0,0,0);
    float pz = yy_2_3(0,0,0,0);
    
    float vx = yy_2_3(1,0,0,0);
    float vy = yy_4_5(0,0,0,0);
    float vz = yy_4_5(1,0,0,0);
    
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

    pxx += yy_0_1(0,-4,0,0)*c[-4+half_order];
    pyx += yy_0_1(1,-4,0,0)*c[-4+half_order];
    pzx += yy_2_3(0,-4,0,0)*c[-4+half_order];

    vxx += yy_2_3(1,-4,0,0)*c[-4+half_order];
    vyx += yy_4_5(0,-4,0,0)*c[-4+half_order];
    vzx += yy_4_5(1,-4,0,0)*c[-4+half_order];

    pxy += yy_0_1(0,0,-4,0)*c[-4+half_order];
    pyy += yy_0_1(1,0,-4,0)*c[-4+half_order];
    pzy += yy_2_3(0,0,-4,0)*c[-4+half_order];

    vxy += yy_2_3(1,0,-4,0)*c[-4+half_order];
    vyy += yy_4_5(0,0,-4,0)*c[-4+half_order];
    vzy += yy_4_5(1,0,-4,0)*c[-4+half_order];

    pxz += yy_0_1(0,0,0,-4)*c[-4+half_order];
    pyz += yy_0_1(1,0,0,-4)*c[-4+half_order];
    pzz += yy_2_3(0,0,0,-4)*c[-4+half_order];

    vxz += yy_2_3(1,0,0,-4)*c[-4+half_order];
    vyz += yy_4_5(0,0,0,-4)*c[-4+half_order];
    vzz += yy_4_5(1,0,0,-4)*c[-4+half_order];

    pxx += yy_0_1(0,-3,0,0)*c[-3+half_order];
    pyx += yy_0_1(1,-3,0,0)*c[-3+half_order];
    pzx += yy_2_3(0,-3,0,0)*c[-3+half_order];

    vxx += yy_2_3(1,-3,0,0)*c[-3+half_order];
    vyx += yy_4_5(0,-3,0,0)*c[-3+half_order];
    vzx += yy_4_5(1,-3,0,0)*c[-3+half_order];

    pxy += yy_0_1(0,0,-3,0)*c[-3+half_order];
    pyy += yy_0_1(1,0,-3,0)*c[-3+half_order];
    pzy += yy_2_3(0,0,-3,0)*c[-3+half_order];

    vxy += yy_2_3(1,0,-3,0)*c[-3+half_order];
    vyy += yy_4_5(0,0,-3,0)*c[-3+half_order];
    vzy += yy_4_5(1,0,-3,0)*c[-3+half_order];

    pxz += yy_0_1(0,0,0,-3)*c[-3+half_order];
    pyz += yy_0_1(1,0,0,-3)*c[-3+half_order];
    pzz += yy_2_3(0,0,0,-3)*c[-3+half_order];

    vxz += yy_2_3(1,0,0,-3)*c[-3+half_order];
    vyz += yy_4_5(0,0,0,-3)*c[-3+half_order];
    vzz += yy_4_5(1,0,0,-3)*c[-3+half_order];

    pxx += yy_0_1(0,-2,0,0)*c[-2+half_order];
    pyx += yy_0_1(1,-2,0,0)*c[-2+half_order];
    pzx += yy_2_3(0,-2,0,0)*c[-2+half_order];

    vxx += yy_2_3(1,-2,0,0)*c[-2+half_order];
    vyx += yy_4_5(0,-2,0,0)*c[-2+half_order];
    vzx += yy_4_5(1,-2,0,0)*c[-2+half_order];

    pxy += yy_0_1(0,0,-2,0)*c[-2+half_order];
    pyy += yy_0_1(1,0,-2,0)*c[-2+half_order];
    pzy += yy_2_3(0,0,-2,0)*c[-2+half_order];

    vxy += yy_2_3(1,0,-2,0)*c[-2+half_order];
    vyy += yy_4_5(0,0,-2,0)*c[-2+half_order];
    vzy += yy_4_5(1,0,-2,0)*c[-2+half_order];

    pxz += yy_0_1(0,0,0,-2)*c[-2+half_order];
    pyz += yy_0_1(1,0,0,-2)*c[-2+half_order];
    pzz += yy_2_3(0,0,0,-2)*c[-2+half_order];

    vxz += yy_2_3(1,0,0,-2)*c[-2+half_order];
    vyz += yy_4_5(0,0,0,-2)*c[-2+half_order];
    vzz += yy_4_5(1,0,0,-2)*c[-2+half_order];

        pxx += yy_0_1(0,-1,0,0)*c[-1+half_order];
    pyx += yy_0_1(1,-1,0,0)*c[-1+half_order];
    pzx += yy_2_3(0,-1,0,0)*c[-1+half_order];

    vxx += yy_2_3(1,-1,0,0)*c[-1+half_order];
    vyx += yy_4_5(0,-1,0,0)*c[-1+half_order];
    vzx += yy_4_5(1,-1,0,0)*c[-1+half_order];

    pxy += yy_0_1(0,0,-1,0)*c[-1+half_order];
    pyy += yy_0_1(1,0,-1,0)*c[-1+half_order];
    pzy += yy_2_3(0,0,-1,0)*c[-1+half_order];

    vxy += yy_2_3(1,0,-1,0)*c[-1+half_order];
    vyy += yy_4_5(0,0,-1,0)*c[-1+half_order];
    vzy += yy_4_5(1,0,-1,0)*c[-1+half_order];

    pxz += yy_0_1(0,0,0,-1)*c[-1+half_order];
    pyz += yy_0_1(1,0,0,-1)*c[-1+half_order];
    pzz += yy_2_3(0,0,0,-1)*c[-1+half_order];

    vxz += yy_2_3(1,0,0,-1)*c[-1+half_order];
    vyz += yy_4_5(0,0,0,-1)*c[-1+half_order];
    vzz += yy_4_5(1,0,0,-1)*c[-1+half_order];

    pxx += yy_0_1(0,0,0,0)*c[half_order];
    pyx += yy_0_1(1,0,0,0)*c[half_order];
    pzx += yy_2_3(0,0,0,0)*c[half_order];

    vxx += yy_2_3(1,0,0,0)*c[half_order];
    vyx += yy_4_5(0,0,0,0)*c[half_order];
    vzx += yy_4_5(1,0,0,0)*c[half_order];

    pxy += yy_0_1(0,0,0,0)*c[half_order];
    pyy += yy_0_1(1,0,0,0)*c[half_order];
    pzy += yy_2_3(0,0,0,0)*c[half_order];

    vxy += yy_2_3(1,0,0,0)*c[half_order];
    vyy += yy_4_5(0,0,0,0)*c[half_order];
    vzy += yy_4_5(1,0,0,0)*c[half_order];

    pxz += yy_0_1(0,0,0,0)*c[half_order];
    pyz += yy_0_1(1,0,0,0)*c[half_order];
    pzz += yy_2_3(0,0,0,0)*c[half_order];

    vxz += yy_2_3(1,0,0,0)*c[half_order];
    vyz += yy_4_5(0,0,0,0)*c[half_order];
    vzz += yy_4_5(1,0,0,0)*c[half_order];

    pxx += yy_0_1(0,1,0,0)*c[1+half_order];
    pyx += yy_0_1(1,1,0,0)*c[1+half_order];
    pzx += yy_2_3(0,1,0,0)*c[1+half_order];

    vxx += yy_2_3(1,1,0,0)*c[1+half_order];
    vyx += yy_4_5(0,1,0,0)*c[1+half_order];
    vzx += yy_4_5(1,1,0,0)*c[1+half_order];

    pxy += yy_0_1(0,0,1,0)*c[1+half_order];
    pyy += yy_0_1(1,0,1,0)*c[1+half_order];
    pzy += yy_2_3(0,0,1,0)*c[1+half_order];

    vxy += yy_2_3(1,0,1,0)*c[1+half_order];
    vyy += yy_4_5(0,0,1,0)*c[1+half_order];
    vzy += yy_4_5(1,0,1,0)*c[1+half_order];

    pxz += yy_0_1(0,0,0,1)*c[1+half_order];
    pyz += yy_0_1(1,0,0,1)*c[1+half_order];
    pzz += yy_2_3(0,0,0,1)*c[1+half_order];

    vxz += yy_2_3(1,0,0,1)*c[1+half_order];
    vyz += yy_4_5(0,0,0,1)*c[1+half_order];
    vzz += yy_4_5(1,0,0,1)*c[1+half_order];

    pxx += yy_0_1(0,2,0,0)*c[2+half_order];
    pyx += yy_0_1(1,2,0,0)*c[2+half_order];
    pzx += yy_2_3(0,2,0,0)*c[2+half_order];

    vxx += yy_2_3(1,2,0,0)*c[2+half_order];
    vyx += yy_4_5(0,2,0,0)*c[2+half_order];
    vzx += yy_4_5(1,2,0,0)*c[2+half_order];

    pxy += yy_0_1(0,0,2,0)*c[2+half_order];
    pyy += yy_0_1(1,0,2,0)*c[2+half_order];
    pzy += yy_2_3(0,0,2,0)*c[2+half_order];

    vxy += yy_2_3(1,0,2,0)*c[2+half_order];
    vyy += yy_4_5(0,0,2,0)*c[2+half_order];
    vzy += yy_4_5(1,0,2,0)*c[2+half_order];

    pxz += yy_0_1(0,0,0,2)*c[2+half_order];
    pyz += yy_0_1(1,0,0,2)*c[2+half_order];
    pzz += yy_2_3(0,0,0,2)*c[2+half_order];

    vxz += yy_2_3(1,0,0,2)*c[2+half_order];
    vyz += yy_4_5(0,0,0,2)*c[2+half_order];
    vzz += yy_4_5(1,0,0,2)*c[2+half_order];

    pxx += yy_0_1(0,3,0,0)*c[3+half_order];
    pyx += yy_0_1(1,3,0,0)*c[3+half_order];
    pzx += yy_2_3(0,3,0,0)*c[3+half_order];

    vxx += yy_2_3(1,3,0,0)*c[3+half_order];
    vyx += yy_4_5(0,3,0,0)*c[3+half_order];
    vzx += yy_4_5(1,3,0,0)*c[3+half_order];

    pxy += yy_0_1(0,0,3,0)*c[3+half_order];
    pyy += yy_0_1(1,0,3,0)*c[3+half_order];
    pzy += yy_2_3(0,0,3,0)*c[3+half_order];

    vxy += yy_2_3(1,0,3,0)*c[3+half_order];
    vyy += yy_4_5(0,0,3,0)*c[3+half_order];
    vzy += yy_4_5(1,0,3,0)*c[3+half_order];

    pxz += yy_0_1(0,0,0,3)*c[3+half_order];
    pyz += yy_0_1(1,0,0,3)*c[3+half_order];
    pzz += yy_2_3(0,0,0,3)*c[3+half_order];

    vxz += yy_2_3(1,0,0,3)*c[3+half_order];
    vyz += yy_4_5(0,0,0,3)*c[3+half_order];
    vzz += yy_4_5(1,0,0,3)*c[3+half_order];

    pxx += yy_0_1(0,4,0,0)*c[4+half_order];
    pyx += yy_0_1(1,4,0,0)*c[4+half_order];
    pzx += yy_2_3(0,4,0,0)*c[4+half_order];

    vxx += yy_2_3(1,4,0,0)*c[4+half_order];
    vyx += yy_4_5(0,4,0,0)*c[4+half_order];
    vzx += yy_4_5(1,4,0,0)*c[4+half_order];

    pxy += yy_0_1(0,0,4,0)*c[4+half_order];
    pyy += yy_0_1(1,0,4,0)*c[4+half_order];
    pzy += yy_2_3(0,0,4,0)*c[4+half_order];

    vxy += yy_2_3(1,0,4,0)*c[4+half_order];
    vyy += yy_4_5(0,0,4,0)*c[4+half_order];
    vzy += yy_4_5(1,0,4,0)*c[4+half_order];

    pxz += yy_0_1(0,0,0,4)*c[4+half_order];
    pyz += yy_0_1(1,0,0,4)*c[4+half_order];
    pzz += yy_2_3(0,0,0,4)*c[4+half_order];

    vxz += yy_2_3(1,0,0,4)*c[4+half_order];
    vyz += yy_4_5(0,0,0,4)*c[4+half_order];
    vzz += yy_4_5(1,0,0,4)*c[4+half_order];

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



    dyy_0_1(0,0,0,0) = yy_0_1(0,0,0,0) + ytemp0* *scale1;
    dyy_2_3(1,0,0,0) = yy_2_3(1,0,0,0) + ytemp3* *scale1;
    dyy_0_1(1,0,0,0) = yy_0_1(1,0,0,0) + ytemp1* *scale1;
    dyy_4_5(0,0,0,0) = yy_4_5(0,0,0,0) + ytemp4* *scale1;
    dyy_2_3(0,0,0,0) = yy_2_3(0,0,0,0) + ytemp2* *scale1;
    dyy_4_5(1,0,0,0) = yy_4_5(1,0,0,0) + ytemp5* *scale1;

    sum_0_1(0,0,0,0) += ytemp0 * *scale2;
    sum_2_3(1,0,0,0) += ytemp3 * *scale2;
    sum_0_1(1,0,0,0) += ytemp1 * *scale2;
    sum_4_5(0,0,0,0) += ytemp4 * *scale2;
    sum_2_3(0,0,0,0) += ytemp2 * *scale2;
    sum_4_5(1,0,0,0) += ytemp5 * *scale2;
}

void fd3d_pml_kernel2(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
        const ACC<float>& yy_0_1,  const ACC<float>& yy_2_3, const ACC<float>& yy_4_5,
        const ACC<float>& dyyIn_0_1, const ACC<float>& dyyIn_2_3, const ACC<float>& dyyIn_4_5,
        ACC<float>& dyyOut_0_1, ACC<float>& dyyOut_2_3, ACC<float>& dyyOut_4_5,
        ACC<float>& sum_0_1, ACC<float>& sum_2_3, ACC<float>& sum_4_5) {
    
    // #include "../coeffs/coeffs8.h"
    //  float* c = &coeffs[half_order+half_order*(order+1)];
    const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
    float invdx = 1.0 / dx;
    float invdy = 1.0 / dy;
    float invdz = 1.0 / dz;
    int xbeg=half_order;
    int xend=nx-half_order;
    int ybeg=half_order;
    int yend=ny-half_order;
    int zbeg=half_order;
    int zend=nz-half_order;
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
    
    float px = dyyIn_0_1(0,0,0,0);
    float py = dyyIn_0_1(1,0,0,0);
    float pz = dyyIn_2_3(0,0,0,0);
    
    float vx = dyyIn_2_3(1,0,0,0);
    float vy = dyyIn_4_5(0,0,0,0);
    float vz = dyyIn_4_5(1,0,0,0);
    
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

    pxx += dyyIn_0_1(0,-4,0,0)*c[-4+half_order];
    pyx += dyyIn_0_1(1,-4,0,0)*c[-4+half_order];
    pzx += dyyIn_2_3(0,-4,0,0)*c[-4+half_order];
    
    vxx += dyyIn_2_3(1,-4,0,0)*c[-4+half_order];
    vyx += dyyIn_4_5(0,-4,0,0)*c[-4+half_order];
    vzx += dyyIn_4_5(1,-4,0,0)*c[-4+half_order];
    
    pxy += dyyIn_0_1(0,0,-4,0)*c[-4+half_order];
    pyy += dyyIn_0_1(1,0,-4,0)*c[-4+half_order];
    pzy += dyyIn_2_3(0,0,-4,0)*c[-4+half_order];
    
    vxy += dyyIn_2_3(1,0,-4,0)*c[-4+half_order];
    vyy += dyyIn_4_5(0,0,-4,0)*c[-4+half_order];
    vzy += dyyIn_4_5(1,0,-4,0)*c[-4+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-4)*c[-4+half_order];
    pyz += dyyIn_0_1(1,0,0,-4)*c[-4+half_order];
    pzz += dyyIn_2_3(0,0,0,-4)*c[-4+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-4)*c[-4+half_order];
    vyz += dyyIn_4_5(0,0,0,-4)*c[-4+half_order];
    vzz += dyyIn_4_5(1,0,0,-4)*c[-4+half_order];

    // # i = -3
    pxx += dyyIn_0_1(0,-3,0,0)*c[-3+half_order];
    pyx += dyyIn_0_1(1,-3,0,0)*c[-3+half_order];
    pzx += dyyIn_2_3(0,-3,0,0)*c[-3+half_order];
    
    vxx += dyyIn_2_3(1,-3,0,0)*c[-3+half_order];
    vyx += dyyIn_4_5(0,-3,0,0)*c[-3+half_order];
    vzx += dyyIn_4_5(1,-3,0,0)*c[-3+half_order];
    
    pxy += dyyIn_0_1(0,0,-3,0)*c[-3+half_order];
    pyy += dyyIn_0_1(1,0,-3,0)*c[-3+half_order];
    pzy += dyyIn_2_3(0,0,-3,0)*c[-3+half_order];
    
    vxy += dyyIn_2_3(1,0,-3,0)*c[-3+half_order];
    vyy += dyyIn_4_5(0,0,-3,0)*c[-3+half_order];
    vzy += dyyIn_4_5(1,0,-3,0)*c[-3+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-3)*c[-3+half_order];
    pyz += dyyIn_0_1(1,0,0,-3)*c[-3+half_order];
    pzz += dyyIn_2_3(0,0,0,-3)*c[-3+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-3)*c[-3+half_order];
    vyz += dyyIn_4_5(0,0,0,-3)*c[-3+half_order];
    vzz += dyyIn_4_5(1,0,0,-3)*c[-3+half_order];

    // i = - 2
    
    pxx += dyyIn_0_1(0,-2,0,0)*c[-2+half_order];
    pyx += dyyIn_0_1(1,-2,0,0)*c[-2+half_order];
    pzx += dyyIn_2_3(0,-2,0,0)*c[-2+half_order];
    
    vxx += dyyIn_2_3(1,-2,0,0)*c[-2+half_order];
    vyx += dyyIn_4_5(0,-2,0,0)*c[-2+half_order];
    vzx += dyyIn_4_5(1,-2,0,0)*c[-2+half_order];
    
    pxy += dyyIn_0_1(0,0,-2,0)*c[-2+half_order];
    pyy += dyyIn_0_1(1,0,-2,0)*c[-2+half_order];
    pzy += dyyIn_2_3(0,0,-2,0)*c[-2+half_order];
    
    vxy += dyyIn_2_3(1,0,-2,0)*c[-2+half_order];
    vyy += dyyIn_4_5(0,0,-2,0)*c[-2+half_order];
    vzy += dyyIn_4_5(1,0,-2,0)*c[-2+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-2)*c[-2+half_order];
    pyz += dyyIn_0_1(1,0,0,-2)*c[-2+half_order];
    pzz += dyyIn_2_3(0,0,0,-2)*c[-2+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-2)*c[-2+half_order];
    vyz += dyyIn_4_5(0,0,0,-2)*c[-2+half_order];
    vzz += dyyIn_4_5(1,0,0,-2)*c[-2+half_order];

    // i = - 1
    
    pxx += dyyIn_0_1(0,-1,0,0)*c[-1+half_order];
    pyx += dyyIn_0_1(1,-1,0,0)*c[-1+half_order];
    pzx += dyyIn_2_3(0,-1,0,0)*c[-1+half_order];
    
    vxx += dyyIn_2_3(1,-1,0,0)*c[-1+half_order];
    vyx += dyyIn_4_5(0,-1,0,0)*c[-1+half_order];
    vzx += dyyIn_4_5(1,-1,0,0)*c[-1+half_order];
    
    pxy += dyyIn_0_1(0,0,-1,0)*c[-1+half_order];
    pyy += dyyIn_0_1(1,0,-1,0)*c[-1+half_order];
    pzy += dyyIn_2_3(0,0,-1,0)*c[-1+half_order];
    
    vxy += dyyIn_2_3(1,0,-1,0)*c[-1+half_order];
    vyy += dyyIn_4_5(0,0,-1,0)*c[-1+half_order];
    vzy += dyyIn_4_5(1,0,-1,0)*c[-1+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-1)*c[-1+half_order];
    pyz += dyyIn_0_1(1,0,0,-1)*c[-1+half_order];
    pzz += dyyIn_2_3(0,0,0,-1)*c[-1+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-1)*c[-1+half_order];
    vyz += dyyIn_4_5(0,0,0,-1)*c[-1+half_order];
    vzz += dyyIn_4_5(1,0,0,-1)*c[-1+half_order];

    // i = 0
    
    pxx += dyyIn_0_1(0,0,0,0)*c[half_order];
    pyx += dyyIn_0_1(1,0,0,0)*c[half_order];
    pzx += dyyIn_2_3(0,0,0,0)*c[half_order];
    
    vxx += dyyIn_2_3(1,0,0,0)*c[half_order];
    vyx += dyyIn_4_5(0,0,0,0)*c[half_order];
    vzx += dyyIn_4_5(1,0,0,0)*c[half_order];
    
    pxy += dyyIn_0_1(0,0,0,0)*c[half_order];
    pyy += dyyIn_0_1(1,0,0,0)*c[half_order];
    pzy += dyyIn_2_3(0,0,0,0)*c[half_order];
    
    vxy += dyyIn_2_3(1,0,0,0)*c[half_order];
    vyy += dyyIn_4_5(0,0,0,0)*c[half_order];
    vzy += dyyIn_4_5(1,0,0,0)*c[half_order];
    
    pxz += dyyIn_0_1(0,0,0,0)*c[half_order];
    pyz += dyyIn_0_1(1,0,0,0)*c[half_order];
    pzz += dyyIn_2_3(0,0,0,0)*c[half_order];
    
    vxz += dyyIn_2_3(1,0,0,0)*c[half_order];
    vyz += dyyIn_4_5(0,0,0,0)*c[half_order];
    vzz += dyyIn_4_5(1,0,0,0)*c[half_order];

    // i = 1
    
    pxx += dyyIn_0_1(0,1,0,0)*c[1+half_order];
    pyx += dyyIn_0_1(1,1,0,0)*c[1+half_order];
    pzx += dyyIn_2_3(0,1,0,0)*c[1+half_order];
    
    vxx += dyyIn_2_3(1,1,0,0)*c[1+half_order];
    vyx += dyyIn_4_5(0,1,0,0)*c[1+half_order];
    vzx += dyyIn_4_5(1,1,0,0)*c[1+half_order];
    
    pxy += dyyIn_0_1(0,0,1,0)*c[1+half_order];
    pyy += dyyIn_0_1(1,0,1,0)*c[1+half_order];
    pzy += dyyIn_2_3(0,0,1,0)*c[1+half_order];
    
    vxy += dyyIn_2_3(1,0,1,0)*c[1+half_order];
    vyy += dyyIn_4_5(0,0,1,0)*c[1+half_order];
    vzy += dyyIn_4_5(1,0,1,0)*c[1+half_order];
    
    pxz += dyyIn_0_1(0,0,0,1)*c[1+half_order];
    pyz += dyyIn_0_1(1,0,0,1)*c[1+half_order];
    pzz += dyyIn_2_3(0,0,0,1)*c[1+half_order];
    
    vxz += dyyIn_2_3(1,0,0,1)*c[1+half_order];
    vyz += dyyIn_4_5(0,0,0,1)*c[1+half_order];
    vzz += dyyIn_4_5(1,0,0,1)*c[1+half_order];

    // i = 2
    
    pxx += dyyIn_0_1(0,2,0,0)*c[2+half_order];
    pyx += dyyIn_0_1(1,2,0,0)*c[2+half_order];
    pzx += dyyIn_2_3(0,2,0,0)*c[2+half_order];
    
    vxx += dyyIn_2_3(1,2,0,0)*c[2+half_order];
    vyx += dyyIn_4_5(0,2,0,0)*c[2+half_order];
    vzx += dyyIn_4_5(1,2,0,0)*c[2+half_order];
    
    pxy += dyyIn_0_1(0,0,2,0)*c[2+half_order];
    pyy += dyyIn_0_1(1,0,2,0)*c[2+half_order];
    pzy += dyyIn_2_3(0,0,2,0)*c[2+half_order];
    
    vxy += dyyIn_2_3(1,0,2,0)*c[2+half_order];
    vyy += dyyIn_4_5(0,0,2,0)*c[2+half_order];
    vzy += dyyIn_4_5(1,0,2,0)*c[2+half_order];
    
    pxz += dyyIn_0_1(0,0,0,2)*c[2+half_order];
    pyz += dyyIn_0_1(1,0,0,2)*c[2+half_order];
    pzz += dyyIn_2_3(0,0,0,2)*c[2+half_order];
    
    vxz += dyyIn_2_3(1,0,0,2)*c[2+half_order];
    vyz += dyyIn_4_5(0,0,0,2)*c[2+half_order];
    vzz += dyyIn_4_5(1,0,0,2)*c[2+half_order];

    // i = 3
    
    pxx += dyyIn_0_1(0,3,0,0)*c[3+half_order];
    pyx += dyyIn_0_1(1,3,0,0)*c[3+half_order];
    pzx += dyyIn_2_3(0,3,0,0)*c[3+half_order];
    
    vxx += dyyIn_2_3(1,3,0,0)*c[3+half_order];
    vyx += dyyIn_4_5(0,3,0,0)*c[3+half_order];
    vzx += dyyIn_4_5(1,3,0,0)*c[3+half_order];
    
    pxy += dyyIn_0_1(0,0,3,0)*c[3+half_order];
    pyy += dyyIn_0_1(1,0,3,0)*c[3+half_order];
    pzy += dyyIn_2_3(0,0,3,0)*c[3+half_order];
    
    vxy += dyyIn_2_3(1,0,3,0)*c[3+half_order];
    vyy += dyyIn_4_5(0,0,3,0)*c[3+half_order];
    vzy += dyyIn_4_5(1,0,3,0)*c[3+half_order];
    
    pxz += dyyIn_0_1(0,0,0,3)*c[3+half_order];
    pyz += dyyIn_0_1(1,0,0,3)*c[3+half_order];
    pzz += dyyIn_2_3(0,0,0,3)*c[3+half_order];
    
    vxz += dyyIn_2_3(1,0,0,3)*c[3+half_order];
    vyz += dyyIn_4_5(0,0,0,3)*c[3+half_order];
    vzz += dyyIn_4_5(1,0,0,3)*c[3+half_order];

    // i = 4
    
    pxx += dyyIn_0_1(0,4,0,0)*c[4+half_order];
    pyx += dyyIn_0_1(1,4,0,0)*c[4+half_order];
    pzx += dyyIn_2_3(0,4,0,0)*c[4+half_order];
    
    vxx += dyyIn_2_3(1,4,0,0)*c[4+half_order];
    vyx += dyyIn_4_5(0,4,0,0)*c[4+half_order];
    vzx += dyyIn_4_5(1,4,0,0)*c[4+half_order];
    
    pxy += dyyIn_0_1(0,0,4,0)*c[4+half_order];
    pyy += dyyIn_0_1(1,0,4,0)*c[4+half_order];
    pzy += dyyIn_2_3(0,0,4,0)*c[4+half_order];
    
    vxy += dyyIn_2_3(1,0,4,0)*c[4+half_order];
    vyy += dyyIn_4_5(0,0,4,0)*c[4+half_order];
    vzy += dyyIn_4_5(1,0,4,0)*c[4+half_order];
    
    pxz += dyyIn_0_1(0,0,0,4)*c[4+half_order];
    pyz += dyyIn_0_1(1,0,0,4)*c[4+half_order];
    pzz += dyyIn_2_3(0,0,0,4)*c[4+half_order];
    
    vxz += dyyIn_2_3(1,0,0,4)*c[4+half_order];
    vyz += dyyIn_4_5(0,0,0,4)*c[4+half_order];
    vzz += dyyIn_4_5(1,0,0,4)*c[4+half_order];

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



    dyyOut_0_1(0,0,0,0) = yy_0_1(0,0,0,0) + ytemp0* *scale1;
    dyyOut_2_3(1,0,0,0) = yy_2_3(1,0,0,0) + ytemp3* *scale1;
    dyyOut_0_1(1,0,0,0) = yy_0_1(1,0,0,0) + ytemp1* *scale1;
    dyyOut_4_5(0,0,0,0) = yy_4_5(0,0,0,0) + ytemp4* *scale1;
    dyyOut_2_3(0,0,0,0) = yy_2_3(0,0,0,0) + ytemp2* *scale1;
    dyyOut_4_5(1,0,0,0) = yy_4_5(1,0,0,0) + ytemp5* *scale1;

    sum_0_1(0,0,0,0) += ytemp0 * *scale2;
    sum_2_3(1,0,0,0) += ytemp3 * *scale2;
    sum_0_1(1,0,0,0) += ytemp1 * *scale2;
    sum_4_5(0,0,0,0) += ytemp4 * *scale2;
    sum_2_3(0,0,0,0) += ytemp2 * *scale2;
    sum_4_5(1,0,0,0) += ytemp5 * *scale2;
}

void fd3d_pml_kernel3(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
        ACC<float>& yy_0_1, ACC<float>& yy_2_3, ACC<float>& yy_4_5,
        const ACC<float>& dyyIn_0_1, const ACC<float>& dyyIn_2_3, const ACC<float>& dyyIn_4_5,
        ACC<float>& sum_0_1, ACC<float>& sum_2_3, ACC<float>& sum_4_5) {
  
    // #include "../coeffs/coeffs8.h"
    //  float* c = &coeffs[half_order+half_order*(order+1)];
    const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
    float invdx = 1.0 / dx;
    float invdy = 1.0 / dy;
    float invdz = 1.0 / dz;
    int xbeg=half_order;
    int xend=nx-half_order;
    int ybeg=half_order;
    int yend=ny-half_order;
    int zbeg=half_order;
    int zend=nz-half_order;
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
    
    float px = dyyIn_0_1(0,0,0,0);
    float py = dyyIn_0_1(1,0,0,0);
    float pz = dyyIn_2_3(0,0,0,0);
    
    float vx = dyyIn_2_3(1,0,0,0);
    float vy = dyyIn_4_5(0,0,0,0);
    float vz = dyyIn_4_5(1,0,0,0);
    
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

    pxx += dyyIn_0_1(0,-4,0,0)*c[-4+half_order];
    pyx += dyyIn_0_1(1,-4,0,0)*c[-4+half_order];
    pzx += dyyIn_2_3(0,-4,0,0)*c[-4+half_order];
    
    vxx += dyyIn_2_3(1,-4,0,0)*c[-4+half_order];
    vyx += dyyIn_4_5(0,-4,0,0)*c[-4+half_order];
    vzx += dyyIn_4_5(1,-4,0,0)*c[-4+half_order];
    
    pxy += dyyIn_0_1(0,0,-4,0)*c[-4+half_order];
    pyy += dyyIn_0_1(1,0,-4,0)*c[-4+half_order];
    pzy += dyyIn_2_3(0,0,-4,0)*c[-4+half_order];
    
    vxy += dyyIn_2_3(1,0,-4,0)*c[-4+half_order];
    vyy += dyyIn_4_5(0,0,-4,0)*c[-4+half_order];
    vzy += dyyIn_4_5(1,0,-4,0)*c[-4+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-4)*c[-4+half_order];
    pyz += dyyIn_0_1(1,0,0,-4)*c[-4+half_order];
    pzz += dyyIn_2_3(0,0,0,-4)*c[-4+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-4)*c[-4+half_order];
    vyz += dyyIn_4_5(0,0,0,-4)*c[-4+half_order];
    vzz += dyyIn_4_5(1,0,0,-4)*c[-4+half_order];

    // # i = -3
    pxx += dyyIn_0_1(0,-3,0,0)*c[-3+half_order];
    pyx += dyyIn_0_1(1,-3,0,0)*c[-3+half_order];
    pzx += dyyIn_2_3(0,-3,0,0)*c[-3+half_order];
    
    vxx += dyyIn_2_3(1,-3,0,0)*c[-3+half_order];
    vyx += dyyIn_4_5(0,-3,0,0)*c[-3+half_order];
    vzx += dyyIn_4_5(1,-3,0,0)*c[-3+half_order];
    
    pxy += dyyIn_0_1(0,0,-3,0)*c[-3+half_order];
    pyy += dyyIn_0_1(1,0,-3,0)*c[-3+half_order];
    pzy += dyyIn_2_3(0,0,-3,0)*c[-3+half_order];
    
    vxy += dyyIn_2_3(1,0,-3,0)*c[-3+half_order];
    vyy += dyyIn_4_5(0,0,-3,0)*c[-3+half_order];
    vzy += dyyIn_4_5(1,0,-3,0)*c[-3+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-3)*c[-3+half_order];
    pyz += dyyIn_0_1(1,0,0,-3)*c[-3+half_order];
    pzz += dyyIn_2_3(0,0,0,-3)*c[-3+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-3)*c[-3+half_order];
    vyz += dyyIn_4_5(0,0,0,-3)*c[-3+half_order];
    vzz += dyyIn_4_5(1,0,0,-3)*c[-3+half_order];

    // i = - 2
    
    pxx += dyyIn_0_1(0,-2,0,0)*c[-2+half_order];
    pyx += dyyIn_0_1(1,-2,0,0)*c[-2+half_order];
    pzx += dyyIn_2_3(0,-2,0,0)*c[-2+half_order];
    
    vxx += dyyIn_2_3(1,-2,0,0)*c[-2+half_order];
    vyx += dyyIn_4_5(0,-2,0,0)*c[-2+half_order];
    vzx += dyyIn_4_5(1,-2,0,0)*c[-2+half_order];
    
    pxy += dyyIn_0_1(0,0,-2,0)*c[-2+half_order];
    pyy += dyyIn_0_1(1,0,-2,0)*c[-2+half_order];
    pzy += dyyIn_2_3(0,0,-2,0)*c[-2+half_order];
    
    vxy += dyyIn_2_3(1,0,-2,0)*c[-2+half_order];
    vyy += dyyIn_4_5(0,0,-2,0)*c[-2+half_order];
    vzy += dyyIn_4_5(1,0,-2,0)*c[-2+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-2)*c[-2+half_order];
    pyz += dyyIn_0_1(1,0,0,-2)*c[-2+half_order];
    pzz += dyyIn_2_3(0,0,0,-2)*c[-2+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-2)*c[-2+half_order];
    vyz += dyyIn_4_5(0,0,0,-2)*c[-2+half_order];
    vzz += dyyIn_4_5(1,0,0,-2)*c[-2+half_order];

    // i = - 1
    
    pxx += dyyIn_0_1(0,-1,0,0)*c[-1+half_order];
    pyx += dyyIn_0_1(1,-1,0,0)*c[-1+half_order];
    pzx += dyyIn_2_3(0,-1,0,0)*c[-1+half_order];
    
    vxx += dyyIn_2_3(1,-1,0,0)*c[-1+half_order];
    vyx += dyyIn_4_5(0,-1,0,0)*c[-1+half_order];
    vzx += dyyIn_4_5(1,-1,0,0)*c[-1+half_order];
    
    pxy += dyyIn_0_1(0,0,-1,0)*c[-1+half_order];
    pyy += dyyIn_0_1(1,0,-1,0)*c[-1+half_order];
    pzy += dyyIn_2_3(0,0,-1,0)*c[-1+half_order];
    
    vxy += dyyIn_2_3(1,0,-1,0)*c[-1+half_order];
    vyy += dyyIn_4_5(0,0,-1,0)*c[-1+half_order];
    vzy += dyyIn_4_5(1,0,-1,0)*c[-1+half_order];
    
    pxz += dyyIn_0_1(0,0,0,-1)*c[-1+half_order];
    pyz += dyyIn_0_1(1,0,0,-1)*c[-1+half_order];
    pzz += dyyIn_2_3(0,0,0,-1)*c[-1+half_order];
    
    vxz += dyyIn_2_3(1,0,0,-1)*c[-1+half_order];
    vyz += dyyIn_4_5(0,0,0,-1)*c[-1+half_order];
    vzz += dyyIn_4_5(1,0,0,-1)*c[-1+half_order];

    // i = 0
    
    pxx += dyyIn_0_1(0,0,0,0)*c[half_order];
    pyx += dyyIn_0_1(1,0,0,0)*c[half_order];
    pzx += dyyIn_2_3(0,0,0,0)*c[half_order];
    
    vxx += dyyIn_2_3(1,0,0,0)*c[half_order];
    vyx += dyyIn_4_5(0,0,0,0)*c[half_order];
    vzx += dyyIn_4_5(1,0,0,0)*c[half_order];
    
    pxy += dyyIn_0_1(0,0,0,0)*c[half_order];
    pyy += dyyIn_0_1(1,0,0,0)*c[half_order];
    pzy += dyyIn_2_3(0,0,0,0)*c[half_order];
    
    vxy += dyyIn_2_3(1,0,0,0)*c[half_order];
    vyy += dyyIn_4_5(0,0,0,0)*c[half_order];
    vzy += dyyIn_4_5(1,0,0,0)*c[half_order];
    
    pxz += dyyIn_0_1(0,0,0,0)*c[half_order];
    pyz += dyyIn_0_1(1,0,0,0)*c[half_order];
    pzz += dyyIn_2_3(0,0,0,0)*c[half_order];
    
    vxz += dyyIn_2_3(1,0,0,0)*c[half_order];
    vyz += dyyIn_4_5(0,0,0,0)*c[half_order];
    vzz += dyyIn_4_5(1,0,0,0)*c[half_order];

    // i = 1
    
    pxx += dyyIn_0_1(0,1,0,0)*c[1+half_order];
    pyx += dyyIn_0_1(1,1,0,0)*c[1+half_order];
    pzx += dyyIn_2_3(0,1,0,0)*c[1+half_order];
    
    vxx += dyyIn_2_3(1,1,0,0)*c[1+half_order];
    vyx += dyyIn_4_5(0,1,0,0)*c[1+half_order];
    vzx += dyyIn_4_5(1,1,0,0)*c[1+half_order];
    
    pxy += dyyIn_0_1(0,0,1,0)*c[1+half_order];
    pyy += dyyIn_0_1(1,0,1,0)*c[1+half_order];
    pzy += dyyIn_2_3(0,0,1,0)*c[1+half_order];
    
    vxy += dyyIn_2_3(1,0,1,0)*c[1+half_order];
    vyy += dyyIn_4_5(0,0,1,0)*c[1+half_order];
    vzy += dyyIn_4_5(1,0,1,0)*c[1+half_order];
    
    pxz += dyyIn_0_1(0,0,0,1)*c[1+half_order];
    pyz += dyyIn_0_1(1,0,0,1)*c[1+half_order];
    pzz += dyyIn_2_3(0,0,0,1)*c[1+half_order];
    
    vxz += dyyIn_2_3(1,0,0,1)*c[1+half_order];
    vyz += dyyIn_4_5(0,0,0,1)*c[1+half_order];
    vzz += dyyIn_4_5(1,0,0,1)*c[1+half_order];

    // i = 2
    
    pxx += dyyIn_0_1(0,2,0,0)*c[2+half_order];
    pyx += dyyIn_0_1(1,2,0,0)*c[2+half_order];
    pzx += dyyIn_2_3(0,2,0,0)*c[2+half_order];
    
    vxx += dyyIn_2_3(1,2,0,0)*c[2+half_order];
    vyx += dyyIn_4_5(0,2,0,0)*c[2+half_order];
    vzx += dyyIn_4_5(1,2,0,0)*c[2+half_order];
    
    pxy += dyyIn_0_1(0,0,2,0)*c[2+half_order];
    pyy += dyyIn_0_1(1,0,2,0)*c[2+half_order];
    pzy += dyyIn_2_3(0,0,2,0)*c[2+half_order];
    
    vxy += dyyIn_2_3(1,0,2,0)*c[2+half_order];
    vyy += dyyIn_4_5(0,0,2,0)*c[2+half_order];
    vzy += dyyIn_4_5(1,0,2,0)*c[2+half_order];
    
    pxz += dyyIn_0_1(0,0,0,2)*c[2+half_order];
    pyz += dyyIn_0_1(1,0,0,2)*c[2+half_order];
    pzz += dyyIn_2_3(0,0,0,2)*c[2+half_order];
    
    vxz += dyyIn_2_3(1,0,0,2)*c[2+half_order];
    vyz += dyyIn_4_5(0,0,0,2)*c[2+half_order];
    vzz += dyyIn_4_5(1,0,0,2)*c[2+half_order];

    // i = 3
    
    pxx += dyyIn_0_1(0,3,0,0)*c[3+half_order];
    pyx += dyyIn_0_1(1,3,0,0)*c[3+half_order];
    pzx += dyyIn_2_3(0,3,0,0)*c[3+half_order];
    
    vxx += dyyIn_2_3(1,3,0,0)*c[3+half_order];
    vyx += dyyIn_4_5(0,3,0,0)*c[3+half_order];
    vzx += dyyIn_4_5(1,3,0,0)*c[3+half_order];
    
    pxy += dyyIn_0_1(0,0,3,0)*c[3+half_order];
    pyy += dyyIn_0_1(1,0,3,0)*c[3+half_order];
    pzy += dyyIn_2_3(0,0,3,0)*c[3+half_order];
    
    vxy += dyyIn_2_3(1,0,3,0)*c[3+half_order];
    vyy += dyyIn_4_5(0,0,3,0)*c[3+half_order];
    vzy += dyyIn_4_5(1,0,3,0)*c[3+half_order];
    
    pxz += dyyIn_0_1(0,0,0,3)*c[3+half_order];
    pyz += dyyIn_0_1(1,0,0,3)*c[3+half_order];
    pzz += dyyIn_2_3(0,0,0,3)*c[3+half_order];
    
    vxz += dyyIn_2_3(1,0,0,3)*c[3+half_order];
    vyz += dyyIn_4_5(0,0,0,3)*c[3+half_order];
    vzz += dyyIn_4_5(1,0,0,3)*c[3+half_order];

    // i = 4
    
    pxx += dyyIn_0_1(0,4,0,0)*c[4+half_order];
    pyx += dyyIn_0_1(1,4,0,0)*c[4+half_order];
    pzx += dyyIn_2_3(0,4,0,0)*c[4+half_order];
    
    vxx += dyyIn_2_3(1,4,0,0)*c[4+half_order];
    vyx += dyyIn_4_5(0,4,0,0)*c[4+half_order];
    vzx += dyyIn_4_5(1,4,0,0)*c[4+half_order];
    
    pxy += dyyIn_0_1(0,0,4,0)*c[4+half_order];
    pyy += dyyIn_0_1(1,0,4,0)*c[4+half_order];
    pzy += dyyIn_2_3(0,0,4,0)*c[4+half_order];
    
    vxy += dyyIn_2_3(1,0,4,0)*c[4+half_order];
    vyy += dyyIn_4_5(0,0,4,0)*c[4+half_order];
    vzy += dyyIn_4_5(1,0,4,0)*c[4+half_order];
    
    pxz += dyyIn_0_1(0,0,0,4)*c[4+half_order];
    pyz += dyyIn_0_1(1,0,0,4)*c[4+half_order];
    pzz += dyyIn_2_3(0,0,0,4)*c[4+half_order];
    
    vxz += dyyIn_2_3(1,0,0,4)*c[4+half_order];
    vyz += dyyIn_4_5(0,0,0,4)*c[4+half_order];
    vzz += dyyIn_4_5(1,0,0,4)*c[4+half_order];


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


    yy_0_1(0,0,0,0) += sum_0_1(0,0,0,0) + ytemp0 * *scale2;
    yy_2_3(1,0,0,0) += sum_2_3(1,0,0,0) + ytemp3 * *scale2;
    yy_0_1(1,0,0,0) += sum_0_1(1,0,0,0) + ytemp1 * *scale2;
    yy_4_5(0,0,0,0) += sum_4_5(0,0,0,0) + ytemp4 * *scale2;
    yy_2_3(0,0,0,0) += sum_2_3(0,0,0,0) + ytemp2 * *scale2;
    yy_4_5(1,0,0,0) += sum_4_5(1,0,0,0) + ytemp5 * *scale2;
    
}
