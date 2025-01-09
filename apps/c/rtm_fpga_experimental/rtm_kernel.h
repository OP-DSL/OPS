
#pragma once

void rtm_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, ACC<float>& rho, ACC<float>& mu, 
    ACC<float>& yy_0) {
    float x = 1.0*((float)(idx[0]-nx/2)/nx);
    float y = 1.0*((float)(idx[1]-ny/2)/ny);
    float z = 1.0*((float)(idx[2]-nz/2)/nz);
    //printf("x,y,z = %f %f %f\n",x,y,z);
    const float C = 1.0f;
    const float r0 = 0.001f;
    rho(0,0,0) = 1000.0f; /* density */
    mu(0,0,0) = 0.001f; /* bulk modulus */

    yy_0(0,0,0) = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0); //idx[0] + idx[1] + idx[2];//
}

void kernel_copy(const ACC<float> &in, ACC<float> &out) {
  out(0,0,0) = in(0,0,0);
}

void fd3d_pml_kernel1(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
    const ACC<float>& yy_0, const ACC<float>& yy_1, const ACC<float>& yy_2, const ACC<float>& yy_3, const ACC<float>& yy_4, const ACC<float>& yy_5,
    ACC<float>& dyy_0, ACC<float>& dyy_1, ACC<float>& dyy_2, ACC<float>& dyy_3, ACC<float>& dyy_4, ACC<float>& dyy_5, 
    ACC<float>& sum_0, ACC<float>& sum_1, ACC<float>& sum_2, ACC<float>& sum_3, ACC<float>& sum_4, ACC<float>& sum_5) {
    
    // #include "../coeffs/coeffs8.h"
    //  float* c = &coeffs[half_order+half_order*(order+1)];
    const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
    // float invdx = 1.0 / dx;
    // float invdy = 1.0 / dy;
    // float invdz = 1.0 / dz;
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
    float sigma_factored = sigma * 0.1f;

    if(idx[0]<=xbeg+pml_width){
        float tmp0 = xbeg+pml_width;
        float tmp1 = tmp0 -idx[0];
        sigmax = tmp1 * sigma_factored;///pml_width;
    }
    if(idx[0]>=xend-pml_width){
        float tmp0 = xend-pml_width;
        float tmp1 = idx[0] - tmp0;
        sigmax = tmp1 * sigma_factored;///pml_width;
    }
    if(idx[1]<=ybeg+pml_width){
        float tmp0 = ybeg+pml_width;
        float tmp1 = tmp0 - idx[1]; 
        sigmay= tmp1 * sigma_factored;///pml_width;
    }
    if(idx[1]>=yend-pml_width){
        float tmp0 = yend-pml_width;
        float tmp1 = idx[1] - tmp0; 
        sigmay= tmp1 * sigma_factored;///pml_width;
    }
    if(idx[2]<=zbeg+pml_width){
        float tmp0 = zbeg+pml_width;
        float tmp1 = tmp0 - idx[2];
        sigmaz=tmp1 * sigma_factored;///pml_width;
    }
    if(idx[2]>=zend-pml_width){
        float tmp0 = zend-pml_width;
        float tmp1 = idx[2] - tmp0;
        sigmaz=tmp1 * sigma_factored;///pml_width;
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

    pxx += yy_0(-4,0,0)*c[-4+half_order];
    pyx += yy_1(-4,0,0)*c[-4+half_order];
    pzx += yy_2(-4,0,0)*c[-4+half_order];

    vxx += yy_3(-4,0,0)*c[-4+half_order];
    vyx += yy_4(-4,0,0)*c[-4+half_order];
    vzx += yy_5(-4,0,0)*c[-4+half_order];

    pxy += yy_0(0,-4,0)*c[-4+half_order];
    pyy += yy_1(0,-4,0)*c[-4+half_order];
    pzy += yy_2(0,-4,0)*c[-4+half_order];

    vxy += yy_3(0,-4,0)*c[-4+half_order];
    vyy += yy_4(0,-4,0)*c[-4+half_order];
    vzy += yy_5(0,-4,0)*c[-4+half_order];

    pxz += yy_0(0,0,-4)*c[-4+half_order];
    pyz += yy_1(0,0,-4)*c[-4+half_order];
    pzz += yy_2(0,0,-4)*c[-4+half_order];

    vxz += yy_3(0,0,-4)*c[-4+half_order];
    vyz += yy_4(0,0,-4)*c[-4+half_order];
    vzz += yy_5(0,0,-4)*c[-4+half_order];

    pxx += yy_0(-3,0,0)*c[-3+half_order];
    pyx += yy_1(-3,0,0)*c[-3+half_order];
    pzx += yy_2(-3,0,0)*c[-3+half_order];

    vxx += yy_3(-3,0,0)*c[-3+half_order];
    vyx += yy_4(-3,0,0)*c[-3+half_order];
    vzx += yy_5(-3,0,0)*c[-3+half_order];

    pxy += yy_0(0,-3,0)*c[-3+half_order];
    pyy += yy_1(0,-3,0)*c[-3+half_order];
    pzy += yy_2(0,-3,0)*c[-3+half_order];

    vxy += yy_3(0,-3,0)*c[-3+half_order];
    vyy += yy_4(0,-3,0)*c[-3+half_order];
    vzy += yy_5(0,-3,0)*c[-3+half_order];

    pxz += yy_0(0,0,-3)*c[-3+half_order];
    pyz += yy_1(0,0,-3)*c[-3+half_order];
    pzz += yy_2(0,0,-3)*c[-3+half_order];

    vxz += yy_3(0,0,-3)*c[-3+half_order];
    vyz += yy_4(0,0,-3)*c[-3+half_order];
    vzz += yy_5(0,0,-3)*c[-3+half_order];

    pxx += yy_0(-2,0,0)*c[-2+half_order];
    pyx += yy_1(-2,0,0)*c[-2+half_order];
    pzx += yy_2(-2,0,0)*c[-2+half_order];

    vxx += yy_3(-2,0,0)*c[-2+half_order];
    vyx += yy_4(-2,0,0)*c[-2+half_order];
    vzx += yy_5(-2,0,0)*c[-2+half_order];

    pxy += yy_0(0,-2,0)*c[-2+half_order];
    pyy += yy_1(0,-2,0)*c[-2+half_order];
    pzy += yy_2(0,-2,0)*c[-2+half_order];

    vxy += yy_3(0,-2,0)*c[-2+half_order];
    vyy += yy_4(0,-2,0)*c[-2+half_order];
    vzy += yy_5(0,-2,0)*c[-2+half_order];

    pxz += yy_0(0,0,-2)*c[-2+half_order];
    pyz += yy_1(0,0,-2)*c[-2+half_order];
    pzz += yy_2(0,0,-2)*c[-2+half_order];

    vxz += yy_3(0,0,-2)*c[-2+half_order];
    vyz += yy_4(0,0,-2)*c[-2+half_order];
    vzz += yy_5(0,0,-2)*c[-2+half_order];

    pxx += yy_0(-1,0,0)*c[-1+half_order];
    pyx += yy_1(-1,0,0)*c[-1+half_order];
    pzx += yy_2(-1,0,0)*c[-1+half_order];

    vxx += yy_3(-1,0,0)*c[-1+half_order];
    vyx += yy_4(-1,0,0)*c[-1+half_order];
    vzx += yy_5(-1,0,0)*c[-1+half_order];

    pxy += yy_0(0,-1,0)*c[-1+half_order];
    pyy += yy_1(0,-1,0)*c[-1+half_order];
    pzy += yy_2(0,-1,0)*c[-1+half_order];

    vxy += yy_3(0,-1,0)*c[-1+half_order];
    vyy += yy_4(0,-1,0)*c[-1+half_order];
    vzy += yy_5(0,-1,0)*c[-1+half_order];

    pxz += yy_0(0,0,-1)*c[-1+half_order];
    pyz += yy_1(0,0,-1)*c[-1+half_order];
    pzz += yy_2(0,0,-1)*c[-1+half_order];

    vxz += yy_3(0,0,-1)*c[-1+half_order];
    vyz += yy_4(0,0,-1)*c[-1+half_order];
    vzz += yy_5(0,0,-1)*c[-1+half_order];

    pxx += yy_0(0,0,0)*c[half_order];
    pyx += yy_1(0,0,0)*c[half_order];
    pzx += yy_2(0,0,0)*c[half_order];

    vxx += yy_3(0,0,0)*c[half_order];
    vyx += yy_4(0,0,0)*c[half_order];
    vzx += yy_5(0,0,0)*c[half_order];

    pxy += yy_0(0,0,0)*c[half_order];
    pyy += yy_1(0,0,0)*c[half_order];
    pzy += yy_2(0,0,0)*c[half_order];

    vxy += yy_3(0,0,0)*c[half_order];
    vyy += yy_4(0,0,0)*c[half_order];
    vzy += yy_5(0,0,0)*c[half_order];

    pxz += yy_0(0,0,0)*c[half_order];
    pyz += yy_1(0,0,0)*c[half_order];
    pzz += yy_2(0,0,0)*c[half_order];

    vxz += yy_3(0,0,0)*c[half_order];
    vyz += yy_4(0,0,0)*c[half_order];
    vzz += yy_5(0,0,0)*c[half_order];

    pxx += yy_0(1,0,0)*c[1+half_order];
    pyx += yy_1(1,0,0)*c[1+half_order];
    pzx += yy_2(1,0,0)*c[1+half_order];

    vxx += yy_3(1,0,0)*c[1+half_order];
    vyx += yy_4(1,0,0)*c[1+half_order];
    vzx += yy_5(1,0,0)*c[1+half_order];

    pxy += yy_0(0,1,0)*c[1+half_order];
    pyy += yy_1(0,1,0)*c[1+half_order];
    pzy += yy_2(0,1,0)*c[1+half_order];

    vxy += yy_3(0,1,0)*c[1+half_order];
    vyy += yy_4(0,1,0)*c[1+half_order];
    vzy += yy_5(0,1,0)*c[1+half_order];

    pxz += yy_0(0,0,1)*c[1+half_order];
    pyz += yy_1(0,0,1)*c[1+half_order];
    pzz += yy_2(0,0,1)*c[1+half_order];

    vxz += yy_3(0,0,1)*c[1+half_order];
    vyz += yy_4(0,0,1)*c[1+half_order];
    vzz += yy_5(0,0,1)*c[1+half_order];

    pxx += yy_0(2,0,0)*c[2+half_order];
    pyx += yy_1(2,0,0)*c[2+half_order];
    pzx += yy_2(2,0,0)*c[2+half_order];

    vxx += yy_3(2,0,0)*c[2+half_order];
    vyx += yy_4(2,0,0)*c[2+half_order];
    vzx += yy_5(2,0,0)*c[2+half_order];

    pxy += yy_0(0,2,0)*c[2+half_order];
    pyy += yy_1(0,2,0)*c[2+half_order];
    pzy += yy_2(0,2,0)*c[2+half_order];

    vxy += yy_3(0,2,0)*c[2+half_order];
    vyy += yy_4(0,2,0)*c[2+half_order];
    vzy += yy_5(0,2,0)*c[2+half_order];

    pxz += yy_0(0,0,2)*c[2+half_order];
    pyz += yy_1(0,0,2)*c[2+half_order];
    pzz += yy_2(0,0,2)*c[2+half_order];

    vxz += yy_3(0,0,2)*c[2+half_order];
    vyz += yy_4(0,0,2)*c[2+half_order];
    vzz += yy_5(0,0,2)*c[2+half_order];

    pxx += yy_0(3,0,0)*c[3+half_order];
    pyx += yy_1(3,0,0)*c[3+half_order];
    pzx += yy_2(3,0,0)*c[3+half_order];

    vxx += yy_3(3,0,0)*c[3+half_order];
    vyx += yy_4(3,0,0)*c[3+half_order];
    vzx += yy_5(3,0,0)*c[3+half_order];

    pxy += yy_0(0,3,0)*c[3+half_order];
    pyy += yy_1(0,3,0)*c[3+half_order];
    pzy += yy_2(0,3,0)*c[3+half_order];

    vxy += yy_3(0,3,0)*c[3+half_order];
    vyy += yy_4(0,3,0)*c[3+half_order];
    vzy += yy_5(0,3,0)*c[3+half_order];

    pxz += yy_0(0,0,3)*c[3+half_order];
    pyz += yy_1(0,0,3)*c[3+half_order];
    pzz += yy_2(0,0,3)*c[3+half_order];

    vxz += yy_3(0,0,3)*c[3+half_order];
    vyz += yy_4(0,0,3)*c[3+half_order];
    vzz += yy_5(0,0,3)*c[3+half_order];

    pxx += yy_0(4,0,0)*c[4+half_order];
    pyx += yy_1(4,0,0)*c[4+half_order];
    pzx += yy_2(4,0,0)*c[4+half_order];

    vxx += yy_3(4,0,0)*c[4+half_order];
    vyx += yy_4(4,0,0)*c[4+half_order];
    vzx += yy_5(4,0,0)*c[4+half_order];

    pxy += yy_0(0,4,0)*c[4+half_order];
    pyy += yy_1(0,4,0)*c[4+half_order];
    pzy += yy_2(0,4,0)*c[4+half_order];

    vxy += yy_3(0,4,0)*c[4+half_order];
    vyy += yy_4(0,4,0)*c[4+half_order];
    vzy += yy_5(0,4,0)*c[4+half_order];

    pxz += yy_0(0,0,4)*c[4+half_order];
    pyz += yy_1(0,0,4)*c[4+half_order];
    pzz += yy_2(0,0,4)*c[4+half_order];

    vxz += yy_3(0,0,4)*c[4+half_order];
    vyz += yy_4(0,0,4)*c[4+half_order];
    vzz += yy_5(0,0,4)*c[4+half_order];

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
    
    float vxx_div_rho = vxx/rho(0,0,0);
    float sigmax_px = sigmax*px;
    float sum_pxx_pyx_pxz = pxx+pyx+pxz;
    float sum_pxx_pyx_pxz_mu = sum_pxx_pyx_pxz * mu(0,0,0);
    float sigmax_vx = sigmax * vx;

    float ytemp0 =(vxx_div_rho - sigmax_px) * *dt;
    float ytemp3 =(sum_pxx_pyx_pxz_mu - sigmax_vx)* *dt;
    
    float vyy_div_rho = vyy/rho(0,0,0);
    float sigmay_py = sigmay * py;
    float sum_pxy_pyy_pyz = pxy+pyy+pyz;
    float sum_pxy_pyy_pyz_mu = sum_pxy_pyy_pyz * mu(0,0,0);
    float sigmay_vy = sigmay * vy;

    float ytemp1 =(vyy_div_rho - sigmay_py) * *dt;
    float ytemp4 =(sum_pxy_pyy_pyz_mu - sigmay_vy) * *dt;
    
    float vzz_div_rho = vzz/rho(0,0,0);
    float sigmaz_pz = sigmaz*pz;
    float sum_pxz_pyz_pzz = pxz+pyz+pzz;
    float sum_pxz_pyz_pzz_mu = sum_pxz_pyz_pzz * mu(0,0,0);
    float sigmaz_vz = sigmaz*vz;

    float ytemp2 =(vzz_div_rho - sigmaz_pz)* *dt;
    float ytemp5 =(sum_pxz_pyz_pzz_mu - sigmaz_vz)* *dt;

    float ytemp0_scale1 = ytemp0* *scale1;
    float ytemp1_scale1 = ytemp1* *scale1;
    float ytemp2_scale1 = ytemp2* *scale1;
    float ytemp3_scale1 = ytemp3* *scale1;
    float ytemp4_scale1 = ytemp4* *scale1;
    float ytemp5_scale1 = ytemp5* *scale1;

    dyy_0(0,0,0) = yy_0(0,0,0) + ytemp0_scale1;
    dyy_3(0,0,0) = yy_3(0,0,0) + ytemp3_scale1;
    dyy_1(0,0,0) = yy_1(0,0,0) + ytemp1_scale1;
    dyy_4(0,0,0) = yy_4(0,0,0) + ytemp4_scale1;
    dyy_2(0,0,0) = yy_2(0,0,0) + ytemp2_scale1;
    dyy_5(0,0,0) = yy_5(0,0,0) + ytemp5_scale1;

    float ytemp0_scale2 = ytemp0 * *scale2;
    float ytemp1_scale2 = ytemp1 * *scale2;
    float ytemp2_scale2 = ytemp2 * *scale2;
    float ytemp3_scale2 = ytemp3 * *scale2;
    float ytemp4_scale2 = ytemp4 * *scale2;
    float ytemp5_scale2 = ytemp5 * *scale2;

    sum_0(0,0,0) += ytemp0_scale2;
    sum_3(0,0,0) += ytemp3_scale2;
    sum_1(0,0,0) += ytemp1_scale2;
    sum_4(0,0,0) += ytemp4_scale2;
    sum_2(0,0,0) += ytemp2_scale2;
    sum_5(0,0,0) += ytemp5_scale2;
    
}

void fd3d_pml_kernel2(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho, const ACC<float>& mu, 
        const ACC<float>& yy_0, const ACC<float>& yy_1, const ACC<float>& yy_2, const ACC<float>& yy_3, const ACC<float>& yy_4, const ACC<float>& yy_5, 
        const ACC<float>& dyyIn_0, const ACC<float>& dyyIn_1, const ACC<float>& dyyIn_2, const ACC<float>& dyyIn_3, const ACC<float>& dyyIn_4, const ACC<float>& dyyIn_5, 
        ACC<float>& dyyOut_0, ACC<float>& dyyOut_1, ACC<float>& dyyOut_2, ACC<float>& dyyOut_3, ACC<float>& dyyOut_4, ACC<float>& dyyOut_5, 
        ACC<float>& sum_0, ACC<float>& sum_1, ACC<float>& sum_2, ACC<float>& sum_3, ACC<float>& sum_4, ACC<float>& sum_5) {
    
    // #include "../coeffs/coeffs8.h"
    //  float* c = &coeffs[half_order+half_order*(order+1)];
    const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
    // float invdx = 1.0 / dx;
    // float invdy = 1.0 / dy;
    // float invdz = 1.0 / dz;
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

    pxx += dyyIn_0(-4,0,0)*c[-4+half_order];
    pyx += dyyIn_1(-4,0,0)*c[-4+half_order];
    pzx += dyyIn_2(-4,0,0)*c[-4+half_order];
    
    vxx += dyyIn_3(-4,0,0)*c[-4+half_order];
    vyx += dyyIn_4(-4,0,0)*c[-4+half_order];
    vzx += dyyIn_5(-4,0,0)*c[-4+half_order];
    
    pxy += dyyIn_0(0,-4,0)*c[-4+half_order];
    pyy += dyyIn_1(0,-4,0)*c[-4+half_order];
    pzy += dyyIn_2(0,-4,0)*c[-4+half_order];
    
    vxy += dyyIn_3(0,-4,0)*c[-4+half_order];
    vyy += dyyIn_4(0,-4,0)*c[-4+half_order];
    vzy += dyyIn_5(0,-4,0)*c[-4+half_order];
    
    pxz += dyyIn_0(0,0,-4)*c[-4+half_order];
    pyz += dyyIn_1(0,0,-4)*c[-4+half_order];
    pzz += dyyIn_2(0,0,-4)*c[-4+half_order];
    
    vxz += dyyIn_3(0,0,-4)*c[-4+half_order];
    vyz += dyyIn_4(0,0,-4)*c[-4+half_order];
    vzz += dyyIn_5(0,0,-4)*c[-4+half_order];

    // # i = -3
    pxx += dyyIn_0(-3,0,0)*c[-3+half_order];
    pyx += dyyIn_1(-3,0,0)*c[-3+half_order];
    pzx += dyyIn_2(-3,0,0)*c[-3+half_order];
    
    vxx += dyyIn_3(-3,0,0)*c[-3+half_order];
    vyx += dyyIn_4(-3,0,0)*c[-3+half_order];
    vzx += dyyIn_5(-3,0,0)*c[-3+half_order];
    
    pxy += dyyIn_0(0,-3,0)*c[-3+half_order];
    pyy += dyyIn_1(0,-3,0)*c[-3+half_order];
    pzy += dyyIn_2(0,-3,0)*c[-3+half_order];
    
    vxy += dyyIn_3(0,-3,0)*c[-3+half_order];
    vyy += dyyIn_4(0,-3,0)*c[-3+half_order];
    vzy += dyyIn_5(0,-3,0)*c[-3+half_order];
    
    pxz += dyyIn_0(0,0,-3)*c[-3+half_order];
    pyz += dyyIn_1(0,0,-3)*c[-3+half_order];
    pzz += dyyIn_2(0,0,-3)*c[-3+half_order];
    
    vxz += dyyIn_3(0,0,-3)*c[-3+half_order];
    vyz += dyyIn_4(0,0,-3)*c[-3+half_order];
    vzz += dyyIn_5(0,0,-3)*c[-3+half_order];

    // i = - 2
    
    pxx += dyyIn_0(-2,0,0)*c[-2+half_order];
    pyx += dyyIn_1(-2,0,0)*c[-2+half_order];
    pzx += dyyIn_2(-2,0,0)*c[-2+half_order];
    
    vxx += dyyIn_3(-2,0,0)*c[-2+half_order];
    vyx += dyyIn_4(-2,0,0)*c[-2+half_order];
    vzx += dyyIn_5(-2,0,0)*c[-2+half_order];
    
    pxy += dyyIn_0(0,-2,0)*c[-2+half_order];
    pyy += dyyIn_1(0,-2,0)*c[-2+half_order];
    pzy += dyyIn_2(0,-2,0)*c[-2+half_order];
    
    vxy += dyyIn_3(0,-2,0)*c[-2+half_order];
    vyy += dyyIn_4(0,-2,0)*c[-2+half_order];
    vzy += dyyIn_5(0,-2,0)*c[-2+half_order];
    
    pxz += dyyIn_0(0,0,-2)*c[-2+half_order];
    pyz += dyyIn_1(0,0,-2)*c[-2+half_order];
    pzz += dyyIn_2(0,0,-2)*c[-2+half_order];
    
    vxz += dyyIn_3(0,0,-2)*c[-2+half_order];
    vyz += dyyIn_4(0,0,-2)*c[-2+half_order];
    vzz += dyyIn_5(0,0,-2)*c[-2+half_order];

    // i = - 1
    
    pxx += dyyIn_0(-1,0,0)*c[-1+half_order];
    pyx += dyyIn_1(-1,0,0)*c[-1+half_order];
    pzx += dyyIn_2(-1,0,0)*c[-1+half_order];
    
    vxx += dyyIn_3(-1,0,0)*c[-1+half_order];
    vyx += dyyIn_4(-1,0,0)*c[-1+half_order];
    vzx += dyyIn_5(-1,0,0)*c[-1+half_order];
    
    pxy += dyyIn_0(0,-1,0)*c[-1+half_order];
    pyy += dyyIn_1(0,-1,0)*c[-1+half_order];
    pzy += dyyIn_2(0,-1,0)*c[-1+half_order];
    
    vxy += dyyIn_3(0,-1,0)*c[-1+half_order];
    vyy += dyyIn_4(0,-1,0)*c[-1+half_order];
    vzy += dyyIn_5(0,-1,0)*c[-1+half_order];
    
    pxz += dyyIn_0(0,0,-1)*c[-1+half_order];
    pyz += dyyIn_1(0,0,-1)*c[-1+half_order];
    pzz += dyyIn_2(0,0,-1)*c[-1+half_order];
    
    vxz += dyyIn_3(0,0,-1)*c[-1+half_order];
    vyz += dyyIn_4(0,0,-1)*c[-1+half_order];
    vzz += dyyIn_5(0,0,-1)*c[-1+half_order];

    // i = 0
    
    pxx += dyyIn_0(0,0,0)*c[half_order];
    pyx += dyyIn_1(0,0,0)*c[half_order];
    pzx += dyyIn_2(0,0,0)*c[half_order];
    
    vxx += dyyIn_3(0,0,0)*c[half_order];
    vyx += dyyIn_4(0,0,0)*c[half_order];
    vzx += dyyIn_5(0,0,0)*c[half_order];
    
    pxy += dyyIn_0(0,0,0)*c[half_order];
    pyy += dyyIn_1(0,0,0)*c[half_order];
    pzy += dyyIn_2(0,0,0)*c[half_order];
    
    vxy += dyyIn_3(0,0,0)*c[half_order];
    vyy += dyyIn_4(0,0,0)*c[half_order];
    vzy += dyyIn_5(0,0,0)*c[half_order];
    
    pxz += dyyIn_0(0,0,0)*c[half_order];
    pyz += dyyIn_1(0,0,0)*c[half_order];
    pzz += dyyIn_2(0,0,0)*c[half_order];
    
    vxz += dyyIn_3(0,0,0)*c[half_order];
    vyz += dyyIn_4(0,0,0)*c[half_order];
    vzz += dyyIn_5(0,0,0)*c[half_order];

    // i = 1
    
    pxx += dyyIn_0(1,0,0)*c[1+half_order];
    pyx += dyyIn_1(1,0,0)*c[1+half_order];
    pzx += dyyIn_2(1,0,0)*c[1+half_order];
    
    vxx += dyyIn_3(1,0,0)*c[1+half_order];
    vyx += dyyIn_4(1,0,0)*c[1+half_order];
    vzx += dyyIn_5(1,0,0)*c[1+half_order];
    
    pxy += dyyIn_0(0,1,0)*c[1+half_order];
    pyy += dyyIn_1(0,1,0)*c[1+half_order];
    pzy += dyyIn_2(0,1,0)*c[1+half_order];
    
    vxy += dyyIn_3(0,1,0)*c[1+half_order];
    vyy += dyyIn_4(0,1,0)*c[1+half_order];
    vzy += dyyIn_5(0,1,0)*c[1+half_order];
    
    pxz += dyyIn_0(0,0,1)*c[1+half_order];
    pyz += dyyIn_1(0,0,1)*c[1+half_order];
    pzz += dyyIn_2(0,0,1)*c[1+half_order];
    
    vxz += dyyIn_3(0,0,1)*c[1+half_order];
    vyz += dyyIn_4(0,0,1)*c[1+half_order];
    vzz += dyyIn_5(0,0,1)*c[1+half_order];

    // i = 2
    
    pxx += dyyIn_0(2,0,0)*c[2+half_order];
    pyx += dyyIn_1(2,0,0)*c[2+half_order];
    pzx += dyyIn_2(2,0,0)*c[2+half_order];
    
    vxx += dyyIn_3(2,0,0)*c[2+half_order];
    vyx += dyyIn_4(2,0,0)*c[2+half_order];
    vzx += dyyIn_5(2,0,0)*c[2+half_order];
    
    pxy += dyyIn_0(0,2,0)*c[2+half_order];
    pyy += dyyIn_1(0,2,0)*c[2+half_order];
    pzy += dyyIn_2(0,2,0)*c[2+half_order];
    
    vxy += dyyIn_3(0,2,0)*c[2+half_order];
    vyy += dyyIn_4(0,2,0)*c[2+half_order];
    vzy += dyyIn_5(0,2,0)*c[2+half_order];
    
    pxz += dyyIn_0(0,0,2)*c[2+half_order];
    pyz += dyyIn_1(0,0,2)*c[2+half_order];
    pzz += dyyIn_2(0,0,2)*c[2+half_order];
    
    vxz += dyyIn_3(0,0,2)*c[2+half_order];
    vyz += dyyIn_4(0,0,2)*c[2+half_order];
    vzz += dyyIn_5(0,0,2)*c[2+half_order];

    // i = 3
    
    pxx += dyyIn_0(3,0,0)*c[3+half_order];
    pyx += dyyIn_1(3,0,0)*c[3+half_order];
    pzx += dyyIn_2(3,0,0)*c[3+half_order];
    
    vxx += dyyIn_3(3,0,0)*c[3+half_order];
    vyx += dyyIn_4(3,0,0)*c[3+half_order];
    vzx += dyyIn_5(3,0,0)*c[3+half_order];
    
    pxy += dyyIn_0(0,3,0)*c[3+half_order];
    pyy += dyyIn_1(0,3,0)*c[3+half_order];
    pzy += dyyIn_2(0,3,0)*c[3+half_order];
    
    vxy += dyyIn_3(0,3,0)*c[3+half_order];
    vyy += dyyIn_4(0,3,0)*c[3+half_order];
    vzy += dyyIn_5(0,3,0)*c[3+half_order];
    
    pxz += dyyIn_0(0,0,3)*c[3+half_order];
    pyz += dyyIn_1(0,0,3)*c[3+half_order];
    pzz += dyyIn_2(0,0,3)*c[3+half_order];
    
    vxz += dyyIn_3(0,0,3)*c[3+half_order];
    vyz += dyyIn_4(0,0,3)*c[3+half_order];
    vzz += dyyIn_5(0,0,3)*c[3+half_order];

    // i = 4
    
    pxx += dyyIn_0(4,0,0)*c[4+half_order];
    pyx += dyyIn_1(4,0,0)*c[4+half_order];
    pzx += dyyIn_2(4,0,0)*c[4+half_order];
    
    vxx += dyyIn_3(4,0,0)*c[4+half_order];
    vyx += dyyIn_4(4,0,0)*c[4+half_order];
    vzx += dyyIn_5(4,0,0)*c[4+half_order];
    
    pxy += dyyIn_0(0,4,0)*c[4+half_order];
    pyy += dyyIn_1(0,4,0)*c[4+half_order];
    pzy += dyyIn_2(0,4,0)*c[4+half_order];
    
    vxy += dyyIn_3(0,4,0)*c[4+half_order];
    vyy += dyyIn_4(0,4,0)*c[4+half_order];
    vzy += dyyIn_5(0,4,0)*c[4+half_order];
    
    pxz += dyyIn_0(0,0,4)*c[4+half_order];
    pyz += dyyIn_1(0,0,4)*c[4+half_order];
    pzz += dyyIn_2(0,0,4)*c[4+half_order];
    
    vxz += dyyIn_3(0,0,4)*c[4+half_order];
    vyz += dyyIn_4(0,0,4)*c[4+half_order];
    vzz += dyyIn_5(0,0,4)*c[4+half_order];

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
    //  float* c = &coeffs[half_order+half_order*(order+1)];
    const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
    // float invdx = 1.0 / dx;
    // float invdy = 1.0 / dy;
    // float invdz = 1.0 / dz;
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

    pxx += dyyIn_0(-4,0,0)*c[-4+half_order];
    pyx += dyyIn_1(-4,0,0)*c[-4+half_order];
    pzx += dyyIn_2(-4,0,0)*c[-4+half_order];
    
    vxx += dyyIn_3(-4,0,0)*c[-4+half_order];
    vyx += dyyIn_4(-4,0,0)*c[-4+half_order];
    vzx += dyyIn_5(-4,0,0)*c[-4+half_order];
    
    pxy += dyyIn_0(0,-4,0)*c[-4+half_order];
    pyy += dyyIn_1(0,-4,0)*c[-4+half_order];
    pzy += dyyIn_2(0,-4,0)*c[-4+half_order];
    
    vxy += dyyIn_3(0,-4,0)*c[-4+half_order];
    vyy += dyyIn_4(0,-4,0)*c[-4+half_order];
    vzy += dyyIn_5(0,-4,0)*c[-4+half_order];
    
    pxz += dyyIn_0(0,0,-4)*c[-4+half_order];
    pyz += dyyIn_1(0,0,-4)*c[-4+half_order];
    pzz += dyyIn_2(0,0,-4)*c[-4+half_order];
    
    vxz += dyyIn_3(0,0,-4)*c[-4+half_order];
    vyz += dyyIn_4(0,0,-4)*c[-4+half_order];
    vzz += dyyIn_5(0,0,-4)*c[-4+half_order];

    // # i = -3
    pxx += dyyIn_0(-3,0,0)*c[-3+half_order];
    pyx += dyyIn_1(-3,0,0)*c[-3+half_order];
    pzx += dyyIn_2(-3,0,0)*c[-3+half_order];
    
    vxx += dyyIn_3(-3,0,0)*c[-3+half_order];
    vyx += dyyIn_4(-3,0,0)*c[-3+half_order];
    vzx += dyyIn_5(-3,0,0)*c[-3+half_order];
    
    pxy += dyyIn_0(0,-3,0)*c[-3+half_order];
    pyy += dyyIn_1(0,-3,0)*c[-3+half_order];
    pzy += dyyIn_2(0,-3,0)*c[-3+half_order];
    
    vxy += dyyIn_3(0,-3,0)*c[-3+half_order];
    vyy += dyyIn_4(0,-3,0)*c[-3+half_order];
    vzy += dyyIn_5(0,-3,0)*c[-3+half_order];
    
    pxz += dyyIn_0(0,0,-3)*c[-3+half_order];
    pyz += dyyIn_1(0,0,-3)*c[-3+half_order];
    pzz += dyyIn_2(0,0,-3)*c[-3+half_order];
    
    vxz += dyyIn_3(0,0,-3)*c[-3+half_order];
    vyz += dyyIn_4(0,0,-3)*c[-3+half_order];
    vzz += dyyIn_5(0,0,-3)*c[-3+half_order];

    // i = - 2
    
    pxx += dyyIn_0(-2,0,0)*c[-2+half_order];
    pyx += dyyIn_1(-2,0,0)*c[-2+half_order];
    pzx += dyyIn_2(-2,0,0)*c[-2+half_order];
    
    vxx += dyyIn_3(-2,0,0)*c[-2+half_order];
    vyx += dyyIn_4(-2,0,0)*c[-2+half_order];
    vzx += dyyIn_5(-2,0,0)*c[-2+half_order];
    
    pxy += dyyIn_0(0,-2,0)*c[-2+half_order];
    pyy += dyyIn_1(0,-2,0)*c[-2+half_order];
    pzy += dyyIn_2(0,-2,0)*c[-2+half_order];
    
    vxy += dyyIn_3(0,-2,0)*c[-2+half_order];
    vyy += dyyIn_4(0,-2,0)*c[-2+half_order];
    vzy += dyyIn_5(0,-2,0)*c[-2+half_order];
    
    pxz += dyyIn_0(0,0,-2)*c[-2+half_order];
    pyz += dyyIn_1(0,0,-2)*c[-2+half_order];
    pzz += dyyIn_2(0,0,-2)*c[-2+half_order];
    
    vxz += dyyIn_3(0,0,-2)*c[-2+half_order];
    vyz += dyyIn_4(0,0,-2)*c[-2+half_order];
    vzz += dyyIn_5(0,0,-2)*c[-2+half_order];

    // i = - 1
    
    pxx += dyyIn_0(-1,0,0)*c[-1+half_order];
    pyx += dyyIn_1(-1,0,0)*c[-1+half_order];
    pzx += dyyIn_2(-1,0,0)*c[-1+half_order];
    
    vxx += dyyIn_3(-1,0,0)*c[-1+half_order];
    vyx += dyyIn_4(-1,0,0)*c[-1+half_order];
    vzx += dyyIn_5(-1,0,0)*c[-1+half_order];
    
    pxy += dyyIn_0(0,-1,0)*c[-1+half_order];
    pyy += dyyIn_1(0,-1,0)*c[-1+half_order];
    pzy += dyyIn_2(0,-1,0)*c[-1+half_order];
    
    vxy += dyyIn_3(0,-1,0)*c[-1+half_order];
    vyy += dyyIn_4(0,-1,0)*c[-1+half_order];
    vzy += dyyIn_5(0,-1,0)*c[-1+half_order];
    
    pxz += dyyIn_0(0,0,-1)*c[-1+half_order];
    pyz += dyyIn_1(0,0,-1)*c[-1+half_order];
    pzz += dyyIn_2(0,0,-1)*c[-1+half_order];
    
    vxz += dyyIn_3(0,0,-1)*c[-1+half_order];
    vyz += dyyIn_4(0,0,-1)*c[-1+half_order];
    vzz += dyyIn_5(0,0,-1)*c[-1+half_order];

    // i = 0
    
    pxx += dyyIn_0(0,0,0)*c[half_order];
    pyx += dyyIn_1(0,0,0)*c[half_order];
    pzx += dyyIn_2(0,0,0)*c[half_order];
    
    vxx += dyyIn_3(0,0,0)*c[half_order];
    vyx += dyyIn_4(0,0,0)*c[half_order];
    vzx += dyyIn_5(0,0,0)*c[half_order];
    
    pxy += dyyIn_0(0,0,0)*c[half_order];
    pyy += dyyIn_1(0,0,0)*c[half_order];
    pzy += dyyIn_2(0,0,0)*c[half_order];
    
    vxy += dyyIn_3(0,0,0)*c[half_order];
    vyy += dyyIn_4(0,0,0)*c[half_order];
    vzy += dyyIn_5(0,0,0)*c[half_order];
    
    pxz += dyyIn_0(0,0,0)*c[half_order];
    pyz += dyyIn_1(0,0,0)*c[half_order];
    pzz += dyyIn_2(0,0,0)*c[half_order];
    
    vxz += dyyIn_3(0,0,0)*c[half_order];
    vyz += dyyIn_4(0,0,0)*c[half_order];
    vzz += dyyIn_5(0,0,0)*c[half_order];

    // i = 1
    
    pxx += dyyIn_0(1,0,0)*c[1+half_order];
    pyx += dyyIn_1(1,0,0)*c[1+half_order];
    pzx += dyyIn_2(1,0,0)*c[1+half_order];
    
    vxx += dyyIn_3(1,0,0)*c[1+half_order];
    vyx += dyyIn_4(1,0,0)*c[1+half_order];
    vzx += dyyIn_5(1,0,0)*c[1+half_order];
    
    pxy += dyyIn_0(0,1,0)*c[1+half_order];
    pyy += dyyIn_1(0,1,0)*c[1+half_order];
    pzy += dyyIn_2(0,1,0)*c[1+half_order];
    
    vxy += dyyIn_3(0,1,0)*c[1+half_order];
    vyy += dyyIn_4(0,1,0)*c[1+half_order];
    vzy += dyyIn_5(0,1,0)*c[1+half_order];
    
    pxz += dyyIn_0(0,0,1)*c[1+half_order];
    pyz += dyyIn_1(0,0,1)*c[1+half_order];
    pzz += dyyIn_2(0,0,1)*c[1+half_order];
    
    vxz += dyyIn_3(0,0,1)*c[1+half_order];
    vyz += dyyIn_4(0,0,1)*c[1+half_order];
    vzz += dyyIn_5(0,0,1)*c[1+half_order];

    // i = 2
    
    pxx += dyyIn_0(2,0,0)*c[2+half_order];
    pyx += dyyIn_1(2,0,0)*c[2+half_order];
    pzx += dyyIn_2(2,0,0)*c[2+half_order];
    
    vxx += dyyIn_3(2,0,0)*c[2+half_order];
    vyx += dyyIn_4(2,0,0)*c[2+half_order];
    vzx += dyyIn_5(2,0,0)*c[2+half_order];
    
    pxy += dyyIn_0(0,2,0)*c[2+half_order];
    pyy += dyyIn_1(0,2,0)*c[2+half_order];
    pzy += dyyIn_2(0,2,0)*c[2+half_order];
    
    vxy += dyyIn_3(0,2,0)*c[2+half_order];
    vyy += dyyIn_4(0,2,0)*c[2+half_order];
    vzy += dyyIn_5(0,2,0)*c[2+half_order];
    
    pxz += dyyIn_0(0,0,2)*c[2+half_order];
    pyz += dyyIn_1(0,0,2)*c[2+half_order];
    pzz += dyyIn_2(0,0,2)*c[2+half_order];
    
    vxz += dyyIn_3(0,0,2)*c[2+half_order];
    vyz += dyyIn_4(0,0,2)*c[2+half_order];
    vzz += dyyIn_5(0,0,2)*c[2+half_order];

    // i = 3
    
    pxx += dyyIn_0(3,0,0)*c[3+half_order];
    pyx += dyyIn_1(3,0,0)*c[3+half_order];
    pzx += dyyIn_2(3,0,0)*c[3+half_order];
    
    vxx += dyyIn_3(3,0,0)*c[3+half_order];
    vyx += dyyIn_4(3,0,0)*c[3+half_order];
    vzx += dyyIn_5(3,0,0)*c[3+half_order];
    
    pxy += dyyIn_0(0,3,0)*c[3+half_order];
    pyy += dyyIn_1(0,3,0)*c[3+half_order];
    pzy += dyyIn_2(0,3,0)*c[3+half_order];
    
    vxy += dyyIn_3(0,3,0)*c[3+half_order];
    vyy += dyyIn_4(0,3,0)*c[3+half_order];
    vzy += dyyIn_5(0,3,0)*c[3+half_order];
    
    pxz += dyyIn_0(0,0,3)*c[3+half_order];
    pyz += dyyIn_1(0,0,3)*c[3+half_order];
    pzz += dyyIn_2(0,0,3)*c[3+half_order];
    
    vxz += dyyIn_3(0,0,3)*c[3+half_order];
    vyz += dyyIn_4(0,0,3)*c[3+half_order];
    vzz += dyyIn_5(0,0,3)*c[3+half_order];

    // i = 4
    
    pxx += dyyIn_0(4,0,0)*c[4+half_order];
    pyx += dyyIn_1(4,0,0)*c[4+half_order];
    pzx += dyyIn_2(4,0,0)*c[4+half_order];
    
    vxx += dyyIn_3(4,0,0)*c[4+half_order];
    vyx += dyyIn_4(4,0,0)*c[4+half_order];
    vzx += dyyIn_5(4,0,0)*c[4+half_order];
    
    pxy += dyyIn_0(0,4,0)*c[4+half_order];
    pyy += dyyIn_1(0,4,0)*c[4+half_order];
    pzy += dyyIn_2(0,4,0)*c[4+half_order];
    
    vxy += dyyIn_3(0,4,0)*c[4+half_order];
    vyy += dyyIn_4(0,4,0)*c[4+half_order];
    vzy += dyyIn_5(0,4,0)*c[4+half_order];
    
    pxz += dyyIn_0(0,0,4)*c[4+half_order];
    pyz += dyyIn_1(0,0,4)*c[4+half_order];
    pzz += dyyIn_2(0,0,4)*c[4+half_order];
    
    vxz += dyyIn_3(0,0,4)*c[4+half_order];
    vyz += dyyIn_4(0,0,4)*c[4+half_order];
    vzz += dyyIn_5(0,0,4)*c[4+half_order];


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
