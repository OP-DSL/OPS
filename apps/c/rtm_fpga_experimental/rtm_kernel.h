
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

    pxx += yy_0(-4,0,0)*c[-4+half];
    pyx += yy_1(-4,0,0)*c[-4+half];
    pzx += yy_2(-4,0,0)*c[-4+half];

    vxx += yy_3(-4,0,0)*c[-4+half];
    vyx += yy_4(-4,0,0)*c[-4+half];
    vzx += yy_5(-4,0,0)*c[-4+half];

    pxy += yy_0(0,-4,0)*c[-4+half];
    pyy += yy_1(0,-4,0)*c[-4+half];
    pzy += yy_2(0,-4,0)*c[-4+half];

    vxy += yy_3(0,-4,0)*c[-4+half];
    vyy += yy_4(0,-4,0)*c[-4+half];
    vzy += yy_5(0,-4,0)*c[-4+half];

    pxz += yy_0(0,0,-4)*c[-4+half];
    pyz += yy_1(0,0,-4)*c[-4+half];
    pzz += yy_2(0,0,-4)*c[-4+half];

    vxz += yy_3(0,0,-4)*c[-4+half];
    vyz += yy_4(0,0,-4)*c[-4+half];
    vzz += yy_5(0,0,-4)*c[-4+half];

    pxx += yy_0(-3,0,0)*c[-3+half];
    pyx += yy_1(-3,0,0)*c[-3+half];
    pzx += yy_2(-3,0,0)*c[-3+half];

    vxx += yy_3(-3,0,0)*c[-3+half];
    vyx += yy_4(-3,0,0)*c[-3+half];
    vzx += yy_5(-3,0,0)*c[-3+half];

    pxy += yy_0(0,-3,0)*c[-3+half];
    pyy += yy_1(0,-3,0)*c[-3+half];
    pzy += yy_2(0,-3,0)*c[-3+half];

    vxy += yy_3(0,-3,0)*c[-3+half];
    vyy += yy_4(0,-3,0)*c[-3+half];
    vzy += yy_5(0,-3,0)*c[-3+half];

    pxz += yy_0(0,0,-3)*c[-3+half];
    pyz += yy_1(0,0,-3)*c[-3+half];
    pzz += yy_2(0,0,-3)*c[-3+half];

    vxz += yy_3(0,0,-3)*c[-3+half];
    vyz += yy_4(0,0,-3)*c[-3+half];
    vzz += yy_5(0,0,-3)*c[-3+half];

    pxx += yy_0(-2,0,0)*c[-2+half];
    pyx += yy_1(-2,0,0)*c[-2+half];
    pzx += yy_2(-2,0,0)*c[-2+half];

    vxx += yy_3(-2,0,0)*c[-2+half];
    vyx += yy_4(-2,0,0)*c[-2+half];
    vzx += yy_5(-2,0,0)*c[-2+half];

    pxy += yy_0(0,-2,0)*c[-2+half];
    pyy += yy_1(0,-2,0)*c[-2+half];
    pzy += yy_2(0,-2,0)*c[-2+half];

    vxy += yy_3(0,-2,0)*c[-2+half];
    vyy += yy_4(0,-2,0)*c[-2+half];
    vzy += yy_5(0,-2,0)*c[-2+half];

    pxz += yy_0(0,0,-2)*c[-2+half];
    pyz += yy_1(0,0,-2)*c[-2+half];
    pzz += yy_2(0,0,-2)*c[-2+half];

    vxz += yy_3(0,0,-2)*c[-2+half];
    vyz += yy_4(0,0,-2)*c[-2+half];
    vzz += yy_5(0,0,-2)*c[-2+half];

        pxx += yy_0(-1,0,0)*c[-1+half];
    pyx += yy_1(-1,0,0)*c[-1+half];
    pzx += yy_2(-1,0,0)*c[-1+half];

    vxx += yy_3(-1,0,0)*c[-1+half];
    vyx += yy_4(-1,0,0)*c[-1+half];
    vzx += yy_5(-1,0,0)*c[-1+half];

    pxy += yy_0(0,-1,0)*c[-1+half];
    pyy += yy_1(0,-1,0)*c[-1+half];
    pzy += yy_2(0,-1,0)*c[-1+half];

    vxy += yy_3(0,-1,0)*c[-1+half];
    vyy += yy_4(0,-1,0)*c[-1+half];
    vzy += yy_5(0,-1,0)*c[-1+half];

    pxz += yy_0(0,0,-1)*c[-1+half];
    pyz += yy_1(0,0,-1)*c[-1+half];
    pzz += yy_2(0,0,-1)*c[-1+half];

    vxz += yy_3(0,0,-1)*c[-1+half];
    vyz += yy_4(0,0,-1)*c[-1+half];
    vzz += yy_5(0,0,-1)*c[-1+half];

    pxx += yy_0(0,0,0)*c[half];
    pyx += yy_1(0,0,0)*c[half];
    pzx += yy_2(0,0,0)*c[half];

    vxx += yy_3(0,0,0)*c[half];
    vyx += yy_4(0,0,0)*c[half];
    vzx += yy_5(0,0,0)*c[half];

    pxy += yy_0(0,0,0)*c[half];
    pyy += yy_1(0,0,0)*c[half];
    pzy += yy_2(0,0,0)*c[half];

    vxy += yy_3(0,0,0)*c[half];
    vyy += yy_4(0,0,0)*c[half];
    vzy += yy_5(0,0,0)*c[half];

    pxz += yy_0(0,0,0)*c[half];
    pyz += yy_1(0,0,0)*c[half];
    pzz += yy_2(0,0,0)*c[half];

    vxz += yy_3(0,0,0)*c[half];
    vyz += yy_4(0,0,0)*c[half];
    vzz += yy_5(0,0,0)*c[half];

    pxx += yy_0(1,0,0)*c[1+half];
    pyx += yy_1(1,0,0)*c[1+half];
    pzx += yy_2(1,0,0)*c[1+half];

    vxx += yy_3(1,0,0)*c[1+half];
    vyx += yy_4(1,0,0)*c[1+half];
    vzx += yy_5(1,0,0)*c[1+half];

    pxy += yy_0(0,1,0)*c[1+half];
    pyy += yy_1(0,1,0)*c[1+half];
    pzy += yy_2(0,1,0)*c[1+half];

    vxy += yy_3(0,1,0)*c[1+half];
    vyy += yy_4(0,1,0)*c[1+half];
    vzy += yy_5(0,1,0)*c[1+half];

    pxz += yy_0(0,0,1)*c[1+half];
    pyz += yy_1(0,0,1)*c[1+half];
    pzz += yy_2(0,0,1)*c[1+half];

    vxz += yy_3(0,0,1)*c[1+half];
    vyz += yy_4(0,0,1)*c[1+half];
    vzz += yy_5(0,0,1)*c[1+half];

    pxx += yy_0(2,0,0)*c[2+half];
    pyx += yy_1(2,0,0)*c[2+half];
    pzx += yy_2(2,0,0)*c[2+half];

    vxx += yy_3(2,0,0)*c[2+half];
    vyx += yy_4(2,0,0)*c[2+half];
    vzx += yy_5(2,0,0)*c[2+half];

    pxy += yy_0(0,2,0)*c[2+half];
    pyy += yy_1(0,2,0)*c[2+half];
    pzy += yy_2(0,2,0)*c[2+half];

    vxy += yy_3(0,2,0)*c[2+half];
    vyy += yy_4(0,2,0)*c[2+half];
    vzy += yy_5(0,2,0)*c[2+half];

    pxz += yy_0(0,0,2)*c[2+half];
    pyz += yy_1(0,0,2)*c[2+half];
    pzz += yy_2(0,0,2)*c[2+half];

    vxz += yy_3(0,0,2)*c[2+half];
    vyz += yy_4(0,0,2)*c[2+half];
    vzz += yy_5(0,0,2)*c[2+half];

    pxx += yy_0(3,0,0)*c[3+half];
    pyx += yy_1(3,0,0)*c[3+half];
    pzx += yy_2(3,0,0)*c[3+half];

    vxx += yy_3(3,0,0)*c[3+half];
    vyx += yy_4(3,0,0)*c[3+half];
    vzx += yy_5(3,0,0)*c[3+half];

    pxy += yy_0(0,3,0)*c[3+half];
    pyy += yy_1(0,3,0)*c[3+half];
    pzy += yy_2(0,3,0)*c[3+half];

    vxy += yy_3(0,3,0)*c[3+half];
    vyy += yy_4(0,3,0)*c[3+half];
    vzy += yy_5(0,3,0)*c[3+half];

    pxz += yy_0(0,0,3)*c[3+half];
    pyz += yy_1(0,0,3)*c[3+half];
    pzz += yy_2(0,0,3)*c[3+half];

    vxz += yy_3(0,0,3)*c[3+half];
    vyz += yy_4(0,0,3)*c[3+half];
    vzz += yy_5(0,0,3)*c[3+half];

    pxx += yy_0(4,0,0)*c[4+half];
    pyx += yy_1(4,0,0)*c[4+half];
    pzx += yy_2(4,0,0)*c[4+half];

    vxx += yy_3(4,0,0)*c[4+half];
    vyx += yy_4(4,0,0)*c[4+half];
    vzx += yy_5(4,0,0)*c[4+half];

    pxy += yy_0(0,4,0)*c[4+half];
    pyy += yy_1(0,4,0)*c[4+half];
    pzy += yy_2(0,4,0)*c[4+half];

    vxy += yy_3(0,4,0)*c[4+half];
    vyy += yy_4(0,4,0)*c[4+half];
    vzy += yy_5(0,4,0)*c[4+half];

    pxz += yy_0(0,0,4)*c[4+half];
    pyz += yy_1(0,0,4)*c[4+half];
    pzz += yy_2(0,0,4)*c[4+half];

    vxz += yy_3(0,0,4)*c[4+half];
    vyz += yy_4(0,0,4)*c[4+half];
    vzz += yy_5(0,0,4)*c[4+half];

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
        ACC<float>& dyyOut_0, ACC<float>& dyyOut_1, ACC<float>& dyyOut_2, ACC<float>& dyyOut_3, ACC<float>& dyyOut_4, ACC<float>& dyyOut_5, 
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

    pxx += dyyIn_0(-4,0,0)*c[-4+half];
    pyx += dyyIn_1(-4,0,0)*c[-4+half];
    pzx += dyyIn_2(-4,0,0)*c[-4+half];
    
    vxx += dyyIn_3(-4,0,0)*c[-4+half];
    vyx += dyyIn_4(-4,0,0)*c[-4+half];
    vzx += dyyIn_5(-4,0,0)*c[-4+half];
    
    pxy += dyyIn_0(0,-4,0)*c[-4+half];
    pyy += dyyIn_1(0,-4,0)*c[-4+half];
    pzy += dyyIn_2(0,-4,0)*c[-4+half];
    
    vxy += dyyIn_3(0,-4,0)*c[-4+half];
    vyy += dyyIn_4(0,-4,0)*c[-4+half];
    vzy += dyyIn_5(0,-4,0)*c[-4+half];
    
    pxz += dyyIn_0(0,0,-4)*c[-4+half];
    pyz += dyyIn_1(0,0,-4)*c[-4+half];
    pzz += dyyIn_2(0,0,-4)*c[-4+half];
    
    vxz += dyyIn_3(0,0,-4)*c[-4+half];
    vyz += dyyIn_4(0,0,-4)*c[-4+half];
    vzz += dyyIn_5(0,0,-4)*c[-4+half];

    // # i = -3
    pxx += dyyIn_0(-3,0,0)*c[-3+half];
    pyx += dyyIn_1(-3,0,0)*c[-3+half];
    pzx += dyyIn_2(-3,0,0)*c[-3+half];
    
    vxx += dyyIn_3(-3,0,0)*c[-3+half];
    vyx += dyyIn_4(-3,0,0)*c[-3+half];
    vzx += dyyIn_5(-3,0,0)*c[-3+half];
    
    pxy += dyyIn_0(0,-3,0)*c[-3+half];
    pyy += dyyIn_1(0,-3,0)*c[-3+half];
    pzy += dyyIn_2(0,-3,0)*c[-3+half];
    
    vxy += dyyIn_3(0,-3,0)*c[-3+half];
    vyy += dyyIn_4(0,-3,0)*c[-3+half];
    vzy += dyyIn_5(0,-3,0)*c[-3+half];
    
    pxz += dyyIn_0(0,0,-3)*c[-3+half];
    pyz += dyyIn_1(0,0,-3)*c[-3+half];
    pzz += dyyIn_2(0,0,-3)*c[-3+half];
    
    vxz += dyyIn_3(0,0,-3)*c[-3+half];
    vyz += dyyIn_4(0,0,-3)*c[-3+half];
    vzz += dyyIn_5(0,0,-3)*c[-3+half];

    // i = - 2
    
    pxx += dyyIn_0(-2,0,0)*c[-2+half];
    pyx += dyyIn_1(-2,0,0)*c[-2+half];
    pzx += dyyIn_2(-2,0,0)*c[-2+half];
    
    vxx += dyyIn_3(-2,0,0)*c[-2+half];
    vyx += dyyIn_4(-2,0,0)*c[-2+half];
    vzx += dyyIn_5(-2,0,0)*c[-2+half];
    
    pxy += dyyIn_0(0,-2,0)*c[-2+half];
    pyy += dyyIn_1(0,-2,0)*c[-2+half];
    pzy += dyyIn_2(0,-2,0)*c[-2+half];
    
    vxy += dyyIn_3(0,-2,0)*c[-2+half];
    vyy += dyyIn_4(0,-2,0)*c[-2+half];
    vzy += dyyIn_5(0,-2,0)*c[-2+half];
    
    pxz += dyyIn_0(0,0,-2)*c[-2+half];
    pyz += dyyIn_1(0,0,-2)*c[-2+half];
    pzz += dyyIn_2(0,0,-2)*c[-2+half];
    
    vxz += dyyIn_3(0,0,-2)*c[-2+half];
    vyz += dyyIn_4(0,0,-2)*c[-2+half];
    vzz += dyyIn_5(0,0,-2)*c[-2+half];

    // i = - 1
    
    pxx += dyyIn_0(-1,0,0)*c[-1+half];
    pyx += dyyIn_1(-1,0,0)*c[-1+half];
    pzx += dyyIn_2(-1,0,0)*c[-1+half];
    
    vxx += dyyIn_3(-1,0,0)*c[-1+half];
    vyx += dyyIn_4(-1,0,0)*c[-1+half];
    vzx += dyyIn_5(-1,0,0)*c[-1+half];
    
    pxy += dyyIn_0(0,-1,0)*c[-1+half];
    pyy += dyyIn_1(0,-1,0)*c[-1+half];
    pzy += dyyIn_2(0,-1,0)*c[-1+half];
    
    vxy += dyyIn_3(0,-1,0)*c[-1+half];
    vyy += dyyIn_4(0,-1,0)*c[-1+half];
    vzy += dyyIn_5(0,-1,0)*c[-1+half];
    
    pxz += dyyIn_0(0,0,-1)*c[-1+half];
    pyz += dyyIn_1(0,0,-1)*c[-1+half];
    pzz += dyyIn_2(0,0,-1)*c[-1+half];
    
    vxz += dyyIn_3(0,0,-1)*c[-1+half];
    vyz += dyyIn_4(0,0,-1)*c[-1+half];
    vzz += dyyIn_5(0,0,-1)*c[-1+half];

    // i = 0
    
    pxx += dyyIn_0(0,0,0)*c[half];
    pyx += dyyIn_1(0,0,0)*c[half];
    pzx += dyyIn_2(0,0,0)*c[half];
    
    vxx += dyyIn_3(0,0,0)*c[half];
    vyx += dyyIn_4(0,0,0)*c[half];
    vzx += dyyIn_5(0,0,0)*c[half];
    
    pxy += dyyIn_0(0,0,0)*c[half];
    pyy += dyyIn_1(0,0,0)*c[half];
    pzy += dyyIn_2(0,0,0)*c[half];
    
    vxy += dyyIn_3(0,0,0)*c[half];
    vyy += dyyIn_4(0,0,0)*c[half];
    vzy += dyyIn_5(0,0,0)*c[half];
    
    pxz += dyyIn_0(0,0,0)*c[half];
    pyz += dyyIn_1(0,0,0)*c[half];
    pzz += dyyIn_2(0,0,0)*c[half];
    
    vxz += dyyIn_3(0,0,0)*c[half];
    vyz += dyyIn_4(0,0,0)*c[half];
    vzz += dyyIn_5(0,0,0)*c[half];

    // i = 1
    
    pxx += dyyIn_0(1,0,0)*c[1+half];
    pyx += dyyIn_1(1,0,0)*c[1+half];
    pzx += dyyIn_2(1,0,0)*c[1+half];
    
    vxx += dyyIn_3(1,0,0)*c[1+half];
    vyx += dyyIn_4(1,0,0)*c[1+half];
    vzx += dyyIn_5(1,0,0)*c[1+half];
    
    pxy += dyyIn_0(0,1,0)*c[1+half];
    pyy += dyyIn_1(0,1,0)*c[1+half];
    pzy += dyyIn_2(0,1,0)*c[1+half];
    
    vxy += dyyIn_3(0,1,0)*c[1+half];
    vyy += dyyIn_4(0,1,0)*c[1+half];
    vzy += dyyIn_5(0,1,0)*c[1+half];
    
    pxz += dyyIn_0(0,0,1)*c[1+half];
    pyz += dyyIn_1(0,0,1)*c[1+half];
    pzz += dyyIn_2(0,0,1)*c[1+half];
    
    vxz += dyyIn_3(0,0,1)*c[1+half];
    vyz += dyyIn_4(0,0,1)*c[1+half];
    vzz += dyyIn_5(0,0,1)*c[1+half];

    // i = 2
    
    pxx += dyyIn_0(2,0,0)*c[2+half];
    pyx += dyyIn_1(2,0,0)*c[2+half];
    pzx += dyyIn_2(2,0,0)*c[2+half];
    
    vxx += dyyIn_3(2,0,0)*c[2+half];
    vyx += dyyIn_4(2,0,0)*c[2+half];
    vzx += dyyIn_5(2,0,0)*c[2+half];
    
    pxy += dyyIn_0(0,2,0)*c[2+half];
    pyy += dyyIn_1(0,2,0)*c[2+half];
    pzy += dyyIn_2(0,2,0)*c[2+half];
    
    vxy += dyyIn_3(0,2,0)*c[2+half];
    vyy += dyyIn_4(0,2,0)*c[2+half];
    vzy += dyyIn_5(0,2,0)*c[2+half];
    
    pxz += dyyIn_0(0,0,2)*c[2+half];
    pyz += dyyIn_1(0,0,2)*c[2+half];
    pzz += dyyIn_2(0,0,2)*c[2+half];
    
    vxz += dyyIn_3(0,0,2)*c[2+half];
    vyz += dyyIn_4(0,0,2)*c[2+half];
    vzz += dyyIn_5(0,0,2)*c[2+half];

    // i = 3
    
    pxx += dyyIn_0(3,0,0)*c[3+half];
    pyx += dyyIn_1(3,0,0)*c[3+half];
    pzx += dyyIn_2(3,0,0)*c[3+half];
    
    vxx += dyyIn_3(3,0,0)*c[3+half];
    vyx += dyyIn_4(3,0,0)*c[3+half];
    vzx += dyyIn_5(3,0,0)*c[3+half];
    
    pxy += dyyIn_0(0,3,0)*c[3+half];
    pyy += dyyIn_1(0,3,0)*c[3+half];
    pzy += dyyIn_2(0,3,0)*c[3+half];
    
    vxy += dyyIn_3(0,3,0)*c[3+half];
    vyy += dyyIn_4(0,3,0)*c[3+half];
    vzy += dyyIn_5(0,3,0)*c[3+half];
    
    pxz += dyyIn_0(0,0,3)*c[3+half];
    pyz += dyyIn_1(0,0,3)*c[3+half];
    pzz += dyyIn_2(0,0,3)*c[3+half];
    
    vxz += dyyIn_3(0,0,3)*c[3+half];
    vyz += dyyIn_4(0,0,3)*c[3+half];
    vzz += dyyIn_5(0,0,3)*c[3+half];

    // i = 4
    
    pxx += dyyIn_0(4,0,0)*c[4+half];
    pyx += dyyIn_1(4,0,0)*c[4+half];
    pzx += dyyIn_2(4,0,0)*c[4+half];
    
    vxx += dyyIn_3(4,0,0)*c[4+half];
    vyx += dyyIn_4(4,0,0)*c[4+half];
    vzx += dyyIn_5(4,0,0)*c[4+half];
    
    pxy += dyyIn_0(0,4,0)*c[4+half];
    pyy += dyyIn_1(0,4,0)*c[4+half];
    pzy += dyyIn_2(0,4,0)*c[4+half];
    
    vxy += dyyIn_3(0,4,0)*c[4+half];
    vyy += dyyIn_4(0,4,0)*c[4+half];
    vzy += dyyIn_5(0,4,0)*c[4+half];
    
    pxz += dyyIn_0(0,0,4)*c[4+half];
    pyz += dyyIn_1(0,0,4)*c[4+half];
    pzz += dyyIn_2(0,0,4)*c[4+half];
    
    vxz += dyyIn_3(0,0,4)*c[4+half];
    vyz += dyyIn_4(0,0,4)*c[4+half];
    vzz += dyyIn_5(0,0,4)*c[4+half];

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
        ACC<float>& yy_0, ACC<float>& yy_1, ACC<float>& yy_2, ACC<float>& yy_3, ACC<float>& yy_4, ACC<float>& yy_5,
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

    pxx += dyyIn_0(-4,0,0)*c[-4+half];
    pyx += dyyIn_1(-4,0,0)*c[-4+half];
    pzx += dyyIn_2(-4,0,0)*c[-4+half];
    
    vxx += dyyIn_3(-4,0,0)*c[-4+half];
    vyx += dyyIn_4(-4,0,0)*c[-4+half];
    vzx += dyyIn_5(-4,0,0)*c[-4+half];
    
    pxy += dyyIn_0(0,-4,0)*c[-4+half];
    pyy += dyyIn_1(0,-4,0)*c[-4+half];
    pzy += dyyIn_2(0,-4,0)*c[-4+half];
    
    vxy += dyyIn_3(0,-4,0)*c[-4+half];
    vyy += dyyIn_4(0,-4,0)*c[-4+half];
    vzy += dyyIn_5(0,-4,0)*c[-4+half];
    
    pxz += dyyIn_0(0,0,-4)*c[-4+half];
    pyz += dyyIn_1(0,0,-4)*c[-4+half];
    pzz += dyyIn_2(0,0,-4)*c[-4+half];
    
    vxz += dyyIn_3(0,0,-4)*c[-4+half];
    vyz += dyyIn_4(0,0,-4)*c[-4+half];
    vzz += dyyIn_5(0,0,-4)*c[-4+half];

    // # i = -3
    pxx += dyyIn_0(-3,0,0)*c[-3+half];
    pyx += dyyIn_1(-3,0,0)*c[-3+half];
    pzx += dyyIn_2(-3,0,0)*c[-3+half];
    
    vxx += dyyIn_3(-3,0,0)*c[-3+half];
    vyx += dyyIn_4(-3,0,0)*c[-3+half];
    vzx += dyyIn_5(-3,0,0)*c[-3+half];
    
    pxy += dyyIn_0(0,-3,0)*c[-3+half];
    pyy += dyyIn_1(0,-3,0)*c[-3+half];
    pzy += dyyIn_2(0,-3,0)*c[-3+half];
    
    vxy += dyyIn_3(0,-3,0)*c[-3+half];
    vyy += dyyIn_4(0,-3,0)*c[-3+half];
    vzy += dyyIn_5(0,-3,0)*c[-3+half];
    
    pxz += dyyIn_0(0,0,-3)*c[-3+half];
    pyz += dyyIn_1(0,0,-3)*c[-3+half];
    pzz += dyyIn_2(0,0,-3)*c[-3+half];
    
    vxz += dyyIn_3(0,0,-3)*c[-3+half];
    vyz += dyyIn_4(0,0,-3)*c[-3+half];
    vzz += dyyIn_5(0,0,-3)*c[-3+half];

    // i = - 2
    
    pxx += dyyIn_0(-2,0,0)*c[-2+half];
    pyx += dyyIn_1(-2,0,0)*c[-2+half];
    pzx += dyyIn_2(-2,0,0)*c[-2+half];
    
    vxx += dyyIn_3(-2,0,0)*c[-2+half];
    vyx += dyyIn_4(-2,0,0)*c[-2+half];
    vzx += dyyIn_5(-2,0,0)*c[-2+half];
    
    pxy += dyyIn_0(0,-2,0)*c[-2+half];
    pyy += dyyIn_1(0,-2,0)*c[-2+half];
    pzy += dyyIn_2(0,-2,0)*c[-2+half];
    
    vxy += dyyIn_3(0,-2,0)*c[-2+half];
    vyy += dyyIn_4(0,-2,0)*c[-2+half];
    vzy += dyyIn_5(0,-2,0)*c[-2+half];
    
    pxz += dyyIn_0(0,0,-2)*c[-2+half];
    pyz += dyyIn_1(0,0,-2)*c[-2+half];
    pzz += dyyIn_2(0,0,-2)*c[-2+half];
    
    vxz += dyyIn_3(0,0,-2)*c[-2+half];
    vyz += dyyIn_4(0,0,-2)*c[-2+half];
    vzz += dyyIn_5(0,0,-2)*c[-2+half];

    // i = - 1
    
    pxx += dyyIn_0(-1,0,0)*c[-1+half];
    pyx += dyyIn_1(-1,0,0)*c[-1+half];
    pzx += dyyIn_2(-1,0,0)*c[-1+half];
    
    vxx += dyyIn_3(-1,0,0)*c[-1+half];
    vyx += dyyIn_4(-1,0,0)*c[-1+half];
    vzx += dyyIn_5(-1,0,0)*c[-1+half];
    
    pxy += dyyIn_0(0,-1,0)*c[-1+half];
    pyy += dyyIn_1(0,-1,0)*c[-1+half];
    pzy += dyyIn_2(0,-1,0)*c[-1+half];
    
    vxy += dyyIn_3(0,-1,0)*c[-1+half];
    vyy += dyyIn_4(0,-1,0)*c[-1+half];
    vzy += dyyIn_5(0,-1,0)*c[-1+half];
    
    pxz += dyyIn_0(0,0,-1)*c[-1+half];
    pyz += dyyIn_1(0,0,-1)*c[-1+half];
    pzz += dyyIn_2(0,0,-1)*c[-1+half];
    
    vxz += dyyIn_3(0,0,-1)*c[-1+half];
    vyz += dyyIn_4(0,0,-1)*c[-1+half];
    vzz += dyyIn_5(0,0,-1)*c[-1+half];

    // i = 0
    
    pxx += dyyIn_0(0,0,0)*c[half];
    pyx += dyyIn_1(0,0,0)*c[half];
    pzx += dyyIn_2(0,0,0)*c[half];
    
    vxx += dyyIn_3(0,0,0)*c[half];
    vyx += dyyIn_4(0,0,0)*c[half];
    vzx += dyyIn_5(0,0,0)*c[half];
    
    pxy += dyyIn_0(0,0,0)*c[half];
    pyy += dyyIn_1(0,0,0)*c[half];
    pzy += dyyIn_2(0,0,0)*c[half];
    
    vxy += dyyIn_3(0,0,0)*c[half];
    vyy += dyyIn_4(0,0,0)*c[half];
    vzy += dyyIn_5(0,0,0)*c[half];
    
    pxz += dyyIn_0(0,0,0)*c[half];
    pyz += dyyIn_1(0,0,0)*c[half];
    pzz += dyyIn_2(0,0,0)*c[half];
    
    vxz += dyyIn_3(0,0,0)*c[half];
    vyz += dyyIn_4(0,0,0)*c[half];
    vzz += dyyIn_5(0,0,0)*c[half];

    // i = 1
    
    pxx += dyyIn_0(1,0,0)*c[1+half];
    pyx += dyyIn_1(1,0,0)*c[1+half];
    pzx += dyyIn_2(1,0,0)*c[1+half];
    
    vxx += dyyIn_3(1,0,0)*c[1+half];
    vyx += dyyIn_4(1,0,0)*c[1+half];
    vzx += dyyIn_5(1,0,0)*c[1+half];
    
    pxy += dyyIn_0(0,1,0)*c[1+half];
    pyy += dyyIn_1(0,1,0)*c[1+half];
    pzy += dyyIn_2(0,1,0)*c[1+half];
    
    vxy += dyyIn_3(0,1,0)*c[1+half];
    vyy += dyyIn_4(0,1,0)*c[1+half];
    vzy += dyyIn_5(0,1,0)*c[1+half];
    
    pxz += dyyIn_0(0,0,1)*c[1+half];
    pyz += dyyIn_1(0,0,1)*c[1+half];
    pzz += dyyIn_2(0,0,1)*c[1+half];
    
    vxz += dyyIn_3(0,0,1)*c[1+half];
    vyz += dyyIn_4(0,0,1)*c[1+half];
    vzz += dyyIn_5(0,0,1)*c[1+half];

    // i = 2
    
    pxx += dyyIn_0(2,0,0)*c[2+half];
    pyx += dyyIn_1(2,0,0)*c[2+half];
    pzx += dyyIn_2(2,0,0)*c[2+half];
    
    vxx += dyyIn_3(2,0,0)*c[2+half];
    vyx += dyyIn_4(2,0,0)*c[2+half];
    vzx += dyyIn_5(2,0,0)*c[2+half];
    
    pxy += dyyIn_0(0,2,0)*c[2+half];
    pyy += dyyIn_1(0,2,0)*c[2+half];
    pzy += dyyIn_2(0,2,0)*c[2+half];
    
    vxy += dyyIn_3(0,2,0)*c[2+half];
    vyy += dyyIn_4(0,2,0)*c[2+half];
    vzy += dyyIn_5(0,2,0)*c[2+half];
    
    pxz += dyyIn_0(0,0,2)*c[2+half];
    pyz += dyyIn_1(0,0,2)*c[2+half];
    pzz += dyyIn_2(0,0,2)*c[2+half];
    
    vxz += dyyIn_3(0,0,2)*c[2+half];
    vyz += dyyIn_4(0,0,2)*c[2+half];
    vzz += dyyIn_5(0,0,2)*c[2+half];

    // i = 3
    
    pxx += dyyIn_0(3,0,0)*c[3+half];
    pyx += dyyIn_1(3,0,0)*c[3+half];
    pzx += dyyIn_2(3,0,0)*c[3+half];
    
    vxx += dyyIn_3(3,0,0)*c[3+half];
    vyx += dyyIn_4(3,0,0)*c[3+half];
    vzx += dyyIn_5(3,0,0)*c[3+half];
    
    pxy += dyyIn_0(0,3,0)*c[3+half];
    pyy += dyyIn_1(0,3,0)*c[3+half];
    pzy += dyyIn_2(0,3,0)*c[3+half];
    
    vxy += dyyIn_3(0,3,0)*c[3+half];
    vyy += dyyIn_4(0,3,0)*c[3+half];
    vzy += dyyIn_5(0,3,0)*c[3+half];
    
    pxz += dyyIn_0(0,0,3)*c[3+half];
    pyz += dyyIn_1(0,0,3)*c[3+half];
    pzz += dyyIn_2(0,0,3)*c[3+half];
    
    vxz += dyyIn_3(0,0,3)*c[3+half];
    vyz += dyyIn_4(0,0,3)*c[3+half];
    vzz += dyyIn_5(0,0,3)*c[3+half];

    // i = 4
    
    pxx += dyyIn_0(4,0,0)*c[4+half];
    pyx += dyyIn_1(4,0,0)*c[4+half];
    pzx += dyyIn_2(4,0,0)*c[4+half];
    
    vxx += dyyIn_3(4,0,0)*c[4+half];
    vyx += dyyIn_4(4,0,0)*c[4+half];
    vzx += dyyIn_5(4,0,0)*c[4+half];
    
    pxy += dyyIn_0(0,4,0)*c[4+half];
    pyy += dyyIn_1(0,4,0)*c[4+half];
    pzy += dyyIn_2(0,4,0)*c[4+half];
    
    vxy += dyyIn_3(0,4,0)*c[4+half];
    vyy += dyyIn_4(0,4,0)*c[4+half];
    vzy += dyyIn_5(0,4,0)*c[4+half];
    
    pxz += dyyIn_0(0,0,4)*c[4+half];
    pyz += dyyIn_1(0,0,4)*c[4+half];
    pzz += dyyIn_2(0,0,4)*c[4+half];
    
    vxz += dyyIn_3(0,0,4)*c[4+half];
    vyz += dyyIn_4(0,0,4)*c[4+half];
    vzz += dyyIn_5(0,0,4)*c[4+half];


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
