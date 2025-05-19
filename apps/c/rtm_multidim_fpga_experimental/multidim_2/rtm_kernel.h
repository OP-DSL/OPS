
#pragma once

void rtm_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, ACC<float>& rho_mu, 
    ACC<float>& yy) {
    float x = 1.0*((float)(idx[0]-nx/2)/nx);
    float y = 1.0*((float)(idx[1]-ny/2)/ny);
    float z = 1.0*((float)(idx[2]-nz/2)/nz);
    //printf("x,y,z = %f %f %f\n",x,y,z);
    const float C = 1.0f;
    const float r0 = 0.001f;
    rho_mu(0,0,0,0) = 1000.0f; /* density */
    rho_mu(1, 0,0,0) = 0.001f; /* bulk modulus */
    unsigned short index = idx[0] + grid_size_x * idx[1] + grid_size_x * grid_size_y * idx[2];
    index =  2 * index;
    float val_0 = index;
    float val_1 = index + 1;//(1./3.)*C*exp(-(x*x+y*y+z*z)/r0); //idx[0] + idx[1] + idx[2];//
    yy(0,0,0,0) = val_0; //val; 
    yy(1,0,0,0) = val_1; //val;
}

void kernel_copy(const ACC<float> &in, ACC<float> &out) {
  out(0,0,0) = in(0,0,0);
}

void kernel_copy_d2(const ACC<float> &in, ACC<float> &out) {
  out(0,0,0,0) = in(0,0,0,0);
  out(1,0,0,0) = in(1,0,0,0);
}

void simple_forward_k1(const ACC<float>& yy_0_1, const ACC<float>& yy_2_3, const ACC<float>& yy_4_5, 
    ACC<float>& n_yy_0_1, ACC<float>& n_yy_2_3, ACC<float>& n_yy_4_5)
{
    n_yy_0_1(0,0,0,0) = yy_0_1(0,0,0,0);
    n_yy_0_1(1,0,0,0) = yy_0_1(1,0,0,0);
    n_yy_2_3(0,0,0,0) = yy_2_3(0,0,0,0);
    n_yy_2_3(1,0,0,0) = yy_2_3(1,0,0,0);
    n_yy_4_5(0,0,0,0) = yy_4_5(0,0,0,0);
    n_yy_4_5(1,0,0,0) = yy_4_5(0,0,0,0);
}

void fd3d_pml_kernel1(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho_mu, 
    const ACC<float>& yy_0_1, const ACC<float>& yy_2_3, const ACC<float>& yy_4_5, 
    ACC<float>& dyy_0_1, ACC<float>& dyy_2_3, ACC<float>& dyy_4_5,
    ACC<float>& sum_0_1, ACC<float>& sum_2_3, ACC<float>& sum_4_5) {
    
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

    float sigma = rho_mu(1,0,0,0)/rho_mu(0,0,0,0);
    float sigma_10_percent = sigma * 0.1f;
    float sigmax=0.0;
    float sigmay=0.0;
    float sigmaz=0.0;

    if(idx[0]<=xbeg+pml_width){

        sigmax = (xpmlbeg-idx[0])*sigma_10_percent;///pml_width;
    }
    else if(idx[0]>=xend-pml_width){
        sigmax=(idx[0]-xpmlend)*sigma_10_percent;///pml_width;
    }
    if(idx[1]<=ybeg+pml_width){
        sigmay=(ypmlbeg-idx[1])*sigma_10_percent;///pml_width;
    }
    else if(idx[1]>=yend-pml_width){
        sigmay=(idx[1]-ypmlend)*sigma_10_percent;///pml_width;
    }
    if(idx[2]<=zbeg+pml_width){
        sigmaz=(zpmlbeg-idx[2])*sigma_10_percent;///pml_width;
    }
    else if(idx[2]>=zend-pml_width){
        sigmaz=(idx[2]-zpmlend)*sigma_10_percent;///pml_width;
    }


                        //sigmax=0.0;
                        //sigmay=0.0;
    
    float px = yy_0_1(0,0,0,0);
    float py = yy_0_1(1,0,0,0);
    float pz = yy_2_3(0,0,0,0);
    
    float vx = yy_2_3(1,0,0,0);
    float vy = yy_4_5(0,0,0,0);
    float vz = yy_4_5(1,0,0,0);
    
    float sigmax_mul_px = sigmax * px;
    float sigmay_mul_py = sigmay * py;
    float sigmaz_mul_pz = sigmaz * pz;
    float sigmax_mul_vx = sigmax * vx;
    float sigmay_mul_vy = sigmay * vy;
    float sigmaz_mul_vz = sigmaz * vz;

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

    const unsigned short h_min_4 = half_order - 4;
    float pxx0 = yy_0_1(0,-4,0,0)*c[h_min_4];
    float pyx0 = yy_0_1(1,-4,0,0)*c[h_min_4];
    float pzx0 = yy_2_3(0,-4,0,0)*c[h_min_4];

    float vxx0 = yy_2_3(1,-4,0,0)*c[h_min_4];
    float vyx0 = yy_4_5(0,-4,0,0)*c[h_min_4];
    float vzx0 = yy_4_5(1,-4,0,0)*c[h_min_4];

    float pxy0 = yy_0_1(0,0,-4,0)*c[h_min_4];
    float pyy0 = yy_0_1(1,0,-4,0)*c[h_min_4];
    float pzy0 = yy_2_3(0,0,-4,0)*c[h_min_4];

    float vxy0 = yy_2_3(1,0,-4,0)*c[h_min_4];
    float vyy0 = yy_4_5(0,0,-4,0)*c[h_min_4];
    float vzy0 = yy_4_5(1,0,-4,0)*c[h_min_4];

    float pxz0 = yy_0_1(0,0,0,-4)*c[h_min_4];
    float pyz0 = yy_0_1(1,0,0,-4)*c[h_min_4];
    float pzz0 = yy_2_3(0,0,0,-4)*c[h_min_4];

    float vxz0 = yy_2_3(1,0,0,-4)*c[h_min_4];
    float vyz0 = yy_4_5(0,0,0,-4)*c[h_min_4];
    float vzz0 = yy_4_5(1,0,0,-4)*c[h_min_4];

    const unsigned short h_min_3 = half_order - 3;
    float pxx1 = yy_0_1(0,-3,0,0)*c[h_min_3];
    float pyx1 = yy_0_1(1,-3,0,0)*c[h_min_3];
    float pzx1 = yy_2_3(0,-3,0,0)*c[h_min_3];

    float vxx1 = yy_2_3(1,-3,0,0)*c[h_min_3];
    float vyx1 = yy_4_5(0,-3,0,0)*c[h_min_3];
    float vzx1 = yy_4_5(1,-3,0,0)*c[h_min_3];

    float pxy1 = yy_0_1(0,0,-3,0)*c[h_min_3];
    float pyy1 = yy_0_1(1,0,-3,0)*c[h_min_3];
    float pzy1 = yy_2_3(0,0,-3,0)*c[h_min_3];

    float vxy1 = yy_2_3(1,0,-3,0)*c[h_min_3];
    float vyy1 = yy_4_5(0,0,-3,0)*c[h_min_3];
    float vzy1 = yy_4_5(1,0,-3,0)*c[h_min_3];

    float pxz1 = yy_0_1(0,0,0,-3)*c[h_min_3];
    float pyz1 = yy_0_1(1,0,0,-3)*c[h_min_3];
    float pzz1 = yy_2_3(0,0,0,-3)*c[h_min_3];

    float vxz1 = yy_2_3(1,0,0,-3)*c[h_min_3];
    float vyz1 = yy_4_5(0,0,0,-3)*c[h_min_3];
    float vzz1 = yy_4_5(1,0,0,-3)*c[h_min_3];

    const unsigned short h_min_2 = half_order - 2;
    float pxx2 = yy_0_1(0,-2,0,0)*c[h_min_2];
    float pyx2 = yy_0_1(1,-2,0,0)*c[h_min_2];
    float pzx2 = yy_2_3(0,-2,0,0)*c[h_min_2];

    float vxx2 = yy_2_3(1,-2,0,0)*c[h_min_2];
    float vyx2 = yy_4_5(0,-2,0,0)*c[h_min_2];
    float vzx2 = yy_4_5(1,-2,0,0)*c[h_min_2];

    float pxy2 = yy_0_1(0,0,-2,0)*c[h_min_2];
    float pyy2 = yy_0_1(1,0,-2,0)*c[h_min_2];
    float pzy2 = yy_2_3(0,0,-2,0)*c[h_min_2];

    float vxy2 = yy_2_3(1,0,-2,0)*c[h_min_2];
    float vyy2 = yy_4_5(0,0,-2,0)*c[h_min_2];
    float vzy2 = yy_4_5(1,0,-2,0)*c[h_min_2];

    float pxz2 = yy_0_1(0,0,0,-2)*c[h_min_2];
    float pyz2 = yy_0_1(1,0,0,-2)*c[h_min_2];
    float pzz2 = yy_2_3(0,0,0,-2)*c[h_min_2];

    float vxz2 = yy_2_3(1,0,0,-2)*c[h_min_2];
    float vyz2 = yy_4_5(0,0,0,-2)*c[h_min_2];
    float vzz2 = yy_4_5(1,0,0,-2)*c[h_min_2];

    const unsigned short h_min_1 = half_order - 1;
    float pxx3 = yy_0_1(0,-1,0,0)*c[h_min_1];
    float pyx3 = yy_0_1(1,-1,0,0)*c[h_min_1];
    float pzx3 = yy_2_3(0,-1,0,0)*c[h_min_1];

    float vxx3 = yy_2_3(1,-1,0,0)*c[h_min_1];
    float vyx3 = yy_4_5(0,-1,0,0)*c[h_min_1];
    float vzx3 = yy_4_5(1,-1,0,0)*c[h_min_1];

    float pxy3 = yy_0_1(0,0,-1,0)*c[h_min_1];
    float pyy3 = yy_0_1(1,0,-1,0)*c[h_min_1];
    float pzy3 = yy_2_3(0,0,-1,0)*c[h_min_1];

    float vxy3 = yy_2_3(1,0,-1,0)*c[h_min_1];
    float vyy3 = yy_4_5(0,0,-1,0)*c[h_min_1];
    float vzy3 = yy_4_5(1,0,-1,0)*c[h_min_1];

    float pxz3 = yy_0_1(0,0,0,-1)*c[h_min_1];
    float pyz3 = yy_0_1(1,0,0,-1)*c[h_min_1];
    float pzz3 = yy_2_3(0,0,0,-1)*c[h_min_1];

    float vxz3 = yy_2_3(1,0,0,-1)*c[h_min_1];
    float vyz3 = yy_4_5(0,0,0,-1)*c[h_min_1];
    float vzz3 = yy_4_5(1,0,0,-1)*c[h_min_1];

    float pxx4 = yy_0_1(0,0,0,0)*c[half_order];
    float pyx4 = yy_0_1(1,0,0,0)*c[half_order];
    float pzx4 = yy_2_3(0,0,0,0)*c[half_order];

    float vxx4 = yy_2_3(1,0,0,0)*c[half_order];
    float vyx4 = yy_4_5(0,0,0,0)*c[half_order];
    float vzx4 = yy_4_5(1,0,0,0)*c[half_order];

    float pxy4 = yy_0_1(0,0,0,0)*c[half_order];
    float pyy4 = yy_0_1(1,0,0,0)*c[half_order];
    float pzy4 = yy_2_3(0,0,0,0)*c[half_order];

    float vxy4 = yy_2_3(1,0,0,0)*c[half_order];
    float vyy4 = yy_4_5(0,0,0,0)*c[half_order];
    float vzy4 = yy_4_5(1,0,0,0)*c[half_order];

    float pxz4 = yy_0_1(0,0,0,0)*c[half_order];
    float pyz4 = yy_0_1(1,0,0,0)*c[half_order];
    float pzz4 = yy_2_3(0,0,0,0)*c[half_order];

    float vxz4 = yy_2_3(1,0,0,0)*c[half_order];
    float vyz4 = yy_4_5(0,0,0,0)*c[half_order];
    float vzz4 = yy_4_5(1,0,0,0)*c[half_order];

    const unsigned short h_pl_1 = half_order + 1;
    float pxx5 = yy_0_1(0,1,0,0)*c[h_pl_1];
    float pyx5 = yy_0_1(1,1,0,0)*c[h_pl_1];
    float pzx5 = yy_2_3(0,1,0,0)*c[h_pl_1];

    float vxx5 = yy_2_3(1,1,0,0)*c[h_pl_1];
    float vyx5 = yy_4_5(0,1,0,0)*c[h_pl_1];
    float vzx5 = yy_4_5(1,1,0,0)*c[h_pl_1];

    float pxy5 = yy_0_1(0,0,1,0)*c[h_pl_1];
    float pyy5 = yy_0_1(1,0,1,0)*c[h_pl_1];
    float pzy5 = yy_2_3(0,0,1,0)*c[h_pl_1];

    float vxy5 = yy_2_3(1,0,1,0)*c[h_pl_1];
    float vyy5 = yy_4_5(0,0,1,0)*c[h_pl_1];
    float vzy5 = yy_4_5(1,0,1,0)*c[h_pl_1];

    float pxz5 = yy_0_1(0,0,0,1)*c[h_pl_1];
    float pyz5 = yy_0_1(1,0,0,1)*c[h_pl_1];
    float pzz5 = yy_2_3(0,0,0,1)*c[h_pl_1];

    float vxz5 = yy_2_3(1,0,0,1)*c[h_pl_1];
    float vyz5 = yy_4_5(0,0,0,1)*c[h_pl_1];
    float vzz5 = yy_4_5(1,0,0,1)*c[h_pl_1];

    const unsigned short h_pl_2 = half_order + 2;
    float pxx6 = yy_0_1(0,2,0,0)*c[h_pl_2];
    float pyx6 = yy_0_1(1,2,0,0)*c[h_pl_2];
    float pzx6 = yy_2_3(0,2,0,0)*c[h_pl_2];

    float vxx6 = yy_2_3(1,2,0,0)*c[h_pl_2];
    float vyx6 = yy_4_5(0,2,0,0)*c[h_pl_2];
    float vzx6 = yy_4_5(1,2,0,0)*c[h_pl_2];

    float pxy6 = yy_0_1(0,0,2,0)*c[h_pl_2];
    float pyy6 = yy_0_1(1,0,2,0)*c[h_pl_2];
    float pzy6 = yy_2_3(0,0,2,0)*c[h_pl_2];

    float vxy6 = yy_2_3(1,0,2,0)*c[h_pl_2];
    float vyy6 = yy_4_5(0,0,2,0)*c[h_pl_2];
    float vzy6 = yy_4_5(1,0,2,0)*c[h_pl_2];

    float pxz6 = yy_0_1(0,0,0,2)*c[h_pl_2];
    float pyz6 = yy_0_1(1,0,0,2)*c[h_pl_2];
    float pzz6 = yy_2_3(0,0,0,2)*c[h_pl_2];

    float vxz6 = yy_2_3(1,0,0,2)*c[h_pl_2];
    float vyz6 = yy_4_5(0,0,0,2)*c[h_pl_2];
    float vzz6 = yy_4_5(1,0,0,2)*c[h_pl_2];

    const unsigned short h_pl_3 = half_order + 3;
    float pxx7 = yy_0_1(0,3,0,0)*c[h_pl_3];
    float pyx7 = yy_0_1(1,3,0,0)*c[h_pl_3];
    float pzx7 = yy_2_3(0,3,0,0)*c[h_pl_3];

    float vxx7 = yy_2_3(1,3,0,0)*c[h_pl_3];
    float vyx7 = yy_4_5(0,3,0,0)*c[h_pl_3];
    float vzx7 = yy_4_5(1,3,0,0)*c[h_pl_3];

    float pxy7 = yy_0_1(0,0,3,0)*c[h_pl_3];
    float pyy7 = yy_0_1(1,0,3,0)*c[h_pl_3];
    float pzy7 = yy_2_3(0,0,3,0)*c[h_pl_3];

    float vxy7 = yy_2_3(1,0,3,0)*c[h_pl_3];
    float vyy7 = yy_4_5(0,0,3,0)*c[h_pl_3];
    float vzy7 = yy_4_5(1,0,3,0)*c[h_pl_3];

    float pxz7 = yy_0_1(0,0,0,3)*c[h_pl_3];
    float pyz7 = yy_0_1(1,0,0,3)*c[h_pl_3];
    float pzz7 = yy_2_3(0,0,0,3)*c[h_pl_3];

    float vxz7 = yy_2_3(1,0,0,3)*c[h_pl_3];
    float vyz7 = yy_4_5(0,0,0,3)*c[h_pl_3];
    float vzz7 = yy_4_5(1,0,0,3)*c[h_pl_3];

    const unsigned short h_pl_4 = half_order + 4;
    float pxx8 = yy_0_1(0,4,0,0)*c[h_pl_4];
    float pyx8 = yy_0_1(1,4,0,0)*c[h_pl_4];
    float pzx8 = yy_2_3(0,4,0,0)*c[h_pl_4];

    float vxx8 = yy_2_3(1,4,0,0)*c[h_pl_4];
    float vyx8 = yy_4_5(0,4,0,0)*c[h_pl_4];
    float vzx8 = yy_4_5(1,4,0,0)*c[h_pl_4];

    float pxy8 = yy_0_1(0,0,4,0)*c[h_pl_4];
    float pyy8 = yy_0_1(1,0,4,0)*c[h_pl_4];
    float pzy8 = yy_2_3(0,0,4,0)*c[h_pl_4];

    float vxy8 = yy_2_3(1,0,4,0)*c[h_pl_4];
    float vyy8 = yy_4_5(0,0,4,0)*c[h_pl_4];
    float vzy8 = yy_4_5(1,0,4,0)*c[h_pl_4];

    float pxz8 = yy_0_1(0,0,0,4)*c[h_pl_4];
    float pyz8 = yy_0_1(1,0,0,4)*c[h_pl_4];
    float pzz8 = yy_2_3(0,0,0,4)*c[h_pl_4];

    float vxz8 = yy_2_3(1,0,0,4)*c[h_pl_4];
    float vyz8 = yy_4_5(0,0,0,4)*c[h_pl_4];
    float vzz8 = yy_4_5(1,0,0,4)*c[h_pl_4];

    float pxx9 = pxx0 + pxx1;
    float pxx10 = pxx2 + pxx3;
    float pxx11 = pxx4 + pxx5;
    float pxx12 = pxx6 + pxx7;
    float pxx13 = pxx8 + pxx9;
    float pxx14 = pxx10 + pxx11;
    float pxx15 = pxx12 + pxx13;
    pxx = pxx14 + pxx15;

    float pyx9 = pyx0 + pyx1;
    float pyx10 = pyx2 + pyx3;
    float pyx11 = pyx4 + pyx5;
    float pyx12 = pyx6 + pyx7;
    float pyx13 = pyx8 + pyx9;
    float pyx14 = pyx10 + pyx11;
    float pyx15 = pyx12 + pyx13;
    pyx = pyx14 + pyx15;

    float pzx9 = pzx0 + pzx1;
    float pzx10 = pzx2 + pzx3;
    float pzx11 = pzx4 + pzx5;
    float pzx12 = pzx6 + pzx7;
    float pzx13 = pzx8 + pzx9;
    float pzx14 = pzx10 + pzx11;
    float pzx15 = pzx12 + pzx13;
    pzx = pzx14 + pzx15;

    float vxx9 = vxx0 + vxx1;
    float vxx10 = vxx2 + vxx3;
    float vxx11 = vxx4 + vxx5;
    float vxx12 = vxx6 + vxx7;
    float vxx13 = vxx8 + vxx9;
    float vxx14 = vxx10 + vxx11;
    float vxx15 = vxx12 + vxx13;
    vxx = vxx14 + vxx15;

    float vyx9 = vyx0 + vyx1;
    float vyx10 = vyx2 + vyx3;
    float vyx11 = vyx4 + vyx5;
    float vyx12 = vyx6 + vyx7;
    float vyx13 = vyx8 + vyx9;
    float vyx14 = vyx10 + vyx11;
    float vyx15 = vyx12 + vyx13;
    vyx = vyx14 + vyx15;

    float vzx9 = vzx0 + vzx1;
    float vzx10 = vzx2 + vzx3;
    float vzx11 = vzx4 + vzx5;
    float vzx12 = vzx6 + vzx7;
    float vzx13 = vzx8 + vzx9;
    float vzx14 = vzx10 + vzx11;
    float vzx15 = vzx12 + vzx13;
    vzx = vzx14 + vzx15;

    float pxy9 = pxy0 + pxy1;
    float pxy10 = pxy2 + pxy3;
    float pxy11 = pxy4 + pxy5;
    float pxy12 = pxy6 + pxy7;
    float pxy13 = pxy8 + pxy9;
    float pxy14 = pxy10 + pxy11;
    float pxy15 = pxy12 + pxy13;
    pxy = pxy14 + pxy15;
    
    float pyy9 = pyy0 + pyy1;
    float pyy10 = pyy2 + pyy3;
    float pyy11 = pyy4 + pyy5;
    float pyy12 = pyy6 + pyy7;
    float pyy13 = pyy8 + pyy9;
    float pyy14 = pyy10 + pyy11;
    float pyy15 = pyy12 + pyy13;
    pyy = pyy14 + pyy15;

    float pzy9 = pzy0 + pzy1;
    float pzy10 = pzy2 + pzy3;
    float pzy11 = pzy4 + pzy5;
    float pzy12 = pzy6 + pzy7;
    float pzy13 = pzy8 + pzy9;
    float pzy14 = pzy10 + pzy11;
    float pzy15 = pzy12 + pzy13;
    pzy = pzy14 + pzy15;

    float vxy9 = vxy0 + vxy1;
    float vxy10 = vxy2 + vxy3;
    float vxy11 = vxy4 + vxy5;
    float vxy12 = vxy6 + vxy7;
    float vxy13 = vxy8 + vxy9;
    float vxy14 = vxy10 + vxy11;
    float vxy15 = vxy12 + vxy13;
    vxy = vxy14 + vxy15;

    float vyy9 = vyy0 + vyy1;
    float vyy10 = vyy2 + vyy3;
    float vyy11 = vyy4 + vyy5;
    float vyy12 = vyy6 + vyy7;
    float vyy13 = vyy8 + vyy9;
    float vyy14 = vyy10 + vyy11;
    float vyy15 = vyy12 + vyy13;
    vyy = vyy14 + vyy15;

    float pxz9 = pxz0 + pxz1;
    float pxz10 = pxz2 + pxz3;
    float pxz11 = pxz4 + pxz5;
    float pxz12 = pxz6 + pxz7;
    float pxz13 = pxz8 + pxz9;
    float pxz14 = pxz10 + pxz11;
    float pxz15 = pxz12 + pxz13;
    pxz = pxz14 + pxz15;

    float pyz9 = pyz0 + pyz1;
    float pyz10 = pyz2 + pyz3;
    float pyz11 = pyz4 + pyz5;
    float pyz12 = pyz6 + pyz7;
    float pyz13 = pyz8 + pyz9;
    float pyz14 = pyz10 + pyz11;
    float pyz15 = pyz12 + pyz13;
    pyz = pyz14 + pyz15;

    float pzz9 = pzz0 + pzz1;
    float pzz10 = pzz2 + pzz3;
    float pzz11 = pzz4 + pzz5;
    float pzz12 = pzz6 + pzz7;
    float pzz13 = pzz8 + pzz9;
    float pzz14 = pzz10 + pzz11;
    float pzz15 = pzz12 + pzz13;
    pzz = pzz14 + pzz15;

    float vxz9 = vxz0 + vxz1;
    float vxz10 = vxz2 + vxz3;
    float vxz11 = vxz4 + vxz5;
    float vxz12 = vxz6 + vxz7;
    float vxz13 = vxz8 + vxz9;
    float vxz14 = vxz10 + vxz11;
    float vxz15 = vxz12 + vxz13;
    vxz = vxz14 + vxz15;

    float vyz9 = vyz0 + vyz1;
    float vyz10 = vyz2 + vyz3;
    float vyz11 = vyz4 + vyz5;
    float vyz12 = vyz6 + vyz7;
    float vyz13 = vyz8 + vyz9;
    float vyz14 = vyz10 + vyz11;
    float vyz15 = vyz12 + vyz13;
    vyz = vyz14 + vyz15;

    float vzz9 = vzz0 + vzz1;
    float vzz10 = vzz2 + vzz3;
    float vzz11 = vzz4 + vzz5;
    float vzz12 = vzz6 + vzz7;
    float vzz13 = vzz8 + vzz9;
    float vzz14 = vzz10 + vzz11;
    float vzz15 = vzz12 + vzz13;
    vzz = vzz14 + vzz15;

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
    
    float vxx_div_rho = vxx/rho_mu(0,0,0,0);
    float vyy_div_rho = vyy/rho_mu(0,0,0,0);
    float vzz_div_rho = vzz/rho_mu(0,0,0,0);

    float add_pxx_pyx_pxz0 = pxx+pyx;
    float add_pxx_pyx_pxz = add_pxx_pyx_pxz0 + pxz;
    float add_pxx_pyx_pxz_mul_mu = add_pxx_pyx_pxz*rho_mu(1,0,0,0);
    float add_pxy_pyy_pyz0 = pxy+pyy;
    float add_pxy_pyy_pyz = add_pxy_pyy_pyz0 + pyz;
    float add_pxy_pyy_pyz_mul_mu = add_pxy_pyy_pyz*rho_mu(1,0,0,0);
    float add_pxz_pyz_pzz0 = pxz+pyz;
    float add_pxz_pyz_pzz = add_pxz_pyz_pzz0 + pzz;
    float add_pxz_pyz_pzz_mul_mu = add_pxz_pyz_pzz*rho_mu(1,0,0,0);

    float ytemp0 =(vxx_div_rho - sigmax_mul_px) * *dt;
    float ytemp3 =(add_pxx_pyx_pxz_mul_mu - sigmax_mul_vx)* *dt;
    
    float ytemp1 =(vyy_div_rho - sigmay_mul_py)* *dt;
    float ytemp4 =(add_pxy_pyy_pyz_mul_mu - sigmay_mul_vy)* *dt;
    
    float ytemp2 =(vzz_div_rho - sigmaz_mul_pz)* *dt;
    float ytemp5 =(add_pxz_pyz_pzz_mul_mu - sigmaz_mul_vz)* *dt;

    float scale1_ytemp0 = ytemp0 * *scale1;
    float scale1_ytemp1 = ytemp1 * *scale1;
    float scale1_ytemp2 = ytemp2 * *scale1;
    float scale1_ytemp3 = ytemp3 * *scale1;
    float scale1_ytemp4 = ytemp4 * *scale1;
    float scale1_ytemp5 = ytemp5 * *scale1;


    dyy_0_1(0,0,0,0) = yy_0_1(0,0,0,0) + scale1_ytemp0;
    dyy_2_3(1,0,0,0) = yy_2_3(1,0,0,0) + scale1_ytemp3;
    dyy_0_1(1,0,0,0) = yy_0_1(1,0,0,0) + scale1_ytemp1;
    dyy_4_5(0,0,0,0) = yy_4_5(0,0,0,0) + scale1_ytemp4;
    dyy_2_3(0,0,0,0) = yy_2_3(0,0,0,0) + scale1_ytemp2;
    dyy_4_5(1,0,0,0) = yy_4_5(1,0,0,0) + scale1_ytemp5;


    sum_0_1(0,0,0,0) = ytemp0 * *scale2;
    sum_2_3(1,0,0,0) = ytemp3 * *scale2;
    sum_0_1(1,0,0,0) = ytemp1 * *scale2;
    sum_4_5(0,0,0,0) = ytemp4 * *scale2;
    sum_2_3(0,0,0,0) = ytemp2 * *scale2;
    sum_4_5(1,0,0,0) = ytemp5 * *scale2;
}

void fd3d_pml_kernel2(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho_mu, 
        const ACC<float>& yy_0_1,  const ACC<float>& yy_2_3, const ACC<float>& yy_4_5,
        const ACC<float>& dyyIn_0_1, const ACC<float>& dyyIn_2_3, const ACC<float>& dyyIn_4_5,
        ACC<float>& dyyOut_0_1, ACC<float>& dyyOut_2_3, ACC<float>& dyyOut_4_5,
        ACC<float>& sum_0_1, ACC<float>& sum_2_3, ACC<float>& sum_4_5) {
    
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

    float sigma = rho_mu(1,0,0,0)/rho_mu(0,0,0,0);
    float sigma_10_percent = sigma * 0.1f;
    float sigmax=0.0;
    float sigmay=0.0;
    float sigmaz=0.0;


    if(idx[0]<=xbeg+pml_width){

        sigmax = (xpmlbeg-idx[0])*sigma_10_percent;///pml_width;
    }
    else if(idx[0]>=xend-pml_width){
        sigmax=(idx[0]-xpmlend)*sigma_10_percent;///pml_width;
    }
    if(idx[1]<=ybeg+pml_width){
        sigmay=(ypmlbeg-idx[1])*sigma_10_percent;///pml_width;
    }
    else if(idx[1]>=yend-pml_width){
        sigmay=(idx[1]-ypmlend)*sigma_10_percent;///pml_width;
    }
    if(idx[2]<=zbeg+pml_width){
        sigmaz=(zpmlbeg-idx[2])*sigma_10_percent;///pml_width;
    }
    else if(idx[2]>=zend-pml_width){
        sigmaz=(idx[2]-zpmlend)*sigma_10_percent;///pml_width;
    }

            //sigmax=0.0;
            //sigmay=0.0;
    
    float px = dyyIn_0_1(0,0,0,0);
    float py = dyyIn_0_1(1,0,0,0);
    float pz = dyyIn_2_3(0,0,0,0);
    
    float vx = dyyIn_2_3(1,0,0,0);
    float vy = dyyIn_4_5(0,0,0,0);
    float vz = dyyIn_4_5(1,0,0,0);
    
    float yy_0_add_sum_0 = yy_0_1(0,0,0,0) + sum_0_1(0,0,0,0);
    float yy_1_add_sum_1 = yy_0_1(1,0,0,0) + sum_0_1(1,0,0,0);
    float yy_2_add_sum_2 = yy_2_3(0,0,0,0) + sum_2_3(0,0,0,0);
    float yy_3_add_sum_3 = yy_2_3(1,0,0,0) + sum_2_3(1,0,0,0);
    float yy_4_add_sum_4 = yy_4_5(0,0,0,0) + sum_4_5(0,0,0,0);
    float yy_5_add_sum_5 = yy_4_5(1,0,0,0) + sum_4_5(1,0,0,0);

    float sigmax_mul_px = sigmax * px;
    float sigmay_mul_py = sigmay * py;
    float sigmaz_mul_pz = sigmaz * pz;
    float sigmax_mul_vx = sigmax * vx;
    float sigmay_mul_vy = sigmay * vy;
    float sigmaz_mul_vz = sigmaz * vz;
    
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

    const unsigned short h_min_4 = half_order - 4;
    float pxx0 = dyyIn_0_1(0,-4,0,0)*c[h_min_4];
    float pyx0 = dyyIn_0_1(1,-4,0,0)*c[h_min_4];
    float pzx0 = dyyIn_2_3(0,-4,0,0)*c[h_min_4];

    float vxx0 = dyyIn_2_3(1,-4,0,0)*c[h_min_4];
    float vyx0 = dyyIn_4_5(0,-4,0,0)*c[h_min_4];
    float vzx0 = dyyIn_4_5(1,-4,0,0)*c[h_min_4];

    float pxy0 = dyyIn_0_1(0,0,-4,0)*c[h_min_4];
    float pyy0 = dyyIn_0_1(1,0,-4,0)*c[h_min_4];
    float pzy0 = dyyIn_2_3(0,0,-4,0)*c[h_min_4];

    float vxy0 = dyyIn_2_3(1,0,-4,0)*c[h_min_4];
    float vyy0 = dyyIn_4_5(0,0,-4,0)*c[h_min_4];
    float vzy0 = dyyIn_4_5(1,0,-4,0)*c[h_min_4];

    float pxz0 = dyyIn_0_1(0,0,0,-4)*c[h_min_4];
    float pyz0 = dyyIn_0_1(1,0,0,-4)*c[h_min_4];
    float pzz0 = dyyIn_2_3(0,0,0,-4)*c[h_min_4];

    float vxz0 = dyyIn_2_3(1,0,0,-4)*c[h_min_4];
    float vyz0 = dyyIn_4_5(0,0,0,-4)*c[h_min_4];
    float vzz0 = dyyIn_4_5(1,0,0,-4)*c[h_min_4];

    const unsigned short h_min_3 = half_order - 3;
    float pxx1 = dyyIn_0_1(0,-3,0,0)*c[h_min_3];
    float pyx1 = dyyIn_0_1(1,-3,0,0)*c[h_min_3];
    float pzx1 = dyyIn_2_3(0,-3,0,0)*c[h_min_3];

    float vxx1 = dyyIn_2_3(1,-3,0,0)*c[h_min_3];
    float vyx1 = dyyIn_4_5(0,-3,0,0)*c[h_min_3];
    float vzx1 = dyyIn_4_5(1,-3,0,0)*c[h_min_3];

    float pxy1 = dyyIn_0_1(0,0,-3,0)*c[h_min_3];
    float pyy1 = dyyIn_0_1(1,0,-3,0)*c[h_min_3];
    float pzy1 = dyyIn_2_3(0,0,-3,0)*c[h_min_3];

    float vxy1 = dyyIn_2_3(1,0,-3,0)*c[h_min_3];
    float vyy1 = dyyIn_4_5(0,0,-3,0)*c[h_min_3];
    float vzy1 = dyyIn_4_5(1,0,-3,0)*c[h_min_3];

    float pxz1 = dyyIn_0_1(0,0,0,-3)*c[h_min_3];
    float pyz1 = dyyIn_0_1(1,0,0,-3)*c[h_min_3];
    float pzz1 = dyyIn_2_3(0,0,0,-3)*c[h_min_3];

    float vxz1 = dyyIn_2_3(1,0,0,-3)*c[h_min_3];
    float vyz1 = dyyIn_4_5(0,0,0,-3)*c[h_min_3];
    float vzz1 = dyyIn_4_5(1,0,0,-3)*c[h_min_3];

    const unsigned short h_min_2 = half_order - 2;
    float pxx2 = dyyIn_0_1(0,-2,0,0)*c[h_min_2];
    float pyx2 = dyyIn_0_1(1,-2,0,0)*c[h_min_2];
    float pzx2 = dyyIn_2_3(0,-2,0,0)*c[h_min_2];

    float vxx2 = dyyIn_2_3(1,-2,0,0)*c[h_min_2];
    float vyx2 = dyyIn_4_5(0,-2,0,0)*c[h_min_2];
    float vzx2 = dyyIn_4_5(1,-2,0,0)*c[h_min_2];

    float pxy2 = dyyIn_0_1(0,0,-2,0)*c[h_min_2];
    float pyy2 = dyyIn_0_1(1,0,-2,0)*c[h_min_2];
    float pzy2 = dyyIn_2_3(0,0,-2,0)*c[h_min_2];

    float vxy2 = dyyIn_2_3(1,0,-2,0)*c[h_min_2];
    float vyy2 = dyyIn_4_5(0,0,-2,0)*c[h_min_2];
    float vzy2 = dyyIn_4_5(1,0,-2,0)*c[h_min_2];

    float pxz2 = dyyIn_0_1(0,0,0,-2)*c[h_min_2];
    float pyz2 = dyyIn_0_1(1,0,0,-2)*c[h_min_2];
    float pzz2 = dyyIn_2_3(0,0,0,-2)*c[h_min_2];

    float vxz2 = dyyIn_2_3(1,0,0,-2)*c[h_min_2];
    float vyz2 = dyyIn_4_5(0,0,0,-2)*c[h_min_2];
    float vzz2 = dyyIn_4_5(1,0,0,-2)*c[h_min_2];

    const unsigned short h_min_1 = half_order - 1;
    float pxx3 = dyyIn_0_1(0,-1,0,0)*c[h_min_1];
    float pyx3 = dyyIn_0_1(1,-1,0,0)*c[h_min_1];
    float pzx3 = dyyIn_2_3(0,-1,0,0)*c[h_min_1];

    float vxx3 = dyyIn_2_3(1,-1,0,0)*c[h_min_1];
    float vyx3 = dyyIn_4_5(0,-1,0,0)*c[h_min_1];
    float vzx3 = dyyIn_4_5(1,-1,0,0)*c[h_min_1];

    float pxy3 = dyyIn_0_1(0,0,-1,0)*c[h_min_1];
    float pyy3 = dyyIn_0_1(1,0,-1,0)*c[h_min_1];
    float pzy3 = dyyIn_2_3(0,0,-1,0)*c[h_min_1];

    float vxy3 = dyyIn_2_3(1,0,-1,0)*c[h_min_1];
    float vyy3 = dyyIn_4_5(0,0,-1,0)*c[h_min_1];
    float vzy3 = dyyIn_4_5(1,0,-1,0)*c[h_min_1];

    float pxz3 = dyyIn_0_1(0,0,0,-1)*c[h_min_1];
    float pyz3 = dyyIn_0_1(1,0,0,-1)*c[h_min_1];
    float pzz3 = dyyIn_2_3(0,0,0,-1)*c[h_min_1];

    float vxz3 = dyyIn_2_3(1,0,0,-1)*c[h_min_1];
    float vyz3 = dyyIn_4_5(0,0,0,-1)*c[h_min_1];
    float vzz3 = dyyIn_4_5(1,0,0,-1)*c[h_min_1];

    float pxx4 = dyyIn_0_1(0,0,0,0)*c[half_order];
    float pyx4 = dyyIn_0_1(1,0,0,0)*c[half_order];
    float pzx4 = dyyIn_2_3(0,0,0,0)*c[half_order];

    float vxx4 = dyyIn_2_3(1,0,0,0)*c[half_order];
    float vyx4 = dyyIn_4_5(0,0,0,0)*c[half_order];
    float vzx4 = dyyIn_4_5(1,0,0,0)*c[half_order];

    float pxy4 = dyyIn_0_1(0,0,0,0)*c[half_order];
    float pyy4 = dyyIn_0_1(1,0,0,0)*c[half_order];
    float pzy4 = dyyIn_2_3(0,0,0,0)*c[half_order];

    float vxy4 = dyyIn_2_3(1,0,0,0)*c[half_order];
    float vyy4 = dyyIn_4_5(0,0,0,0)*c[half_order];
    float vzy4 = dyyIn_4_5(1,0,0,0)*c[half_order];

    float pxz4 = dyyIn_0_1(0,0,0,0)*c[half_order];
    float pyz4 = dyyIn_0_1(1,0,0,0)*c[half_order];
    float pzz4 = dyyIn_2_3(0,0,0,0)*c[half_order];

    float vxz4 = dyyIn_2_3(1,0,0,0)*c[half_order];
    float vyz4 = dyyIn_4_5(0,0,0,0)*c[half_order];
    float vzz4 = dyyIn_4_5(1,0,0,0)*c[half_order];

    const unsigned short h_pl_1 = half_order + 1;
    float pxx5 = dyyIn_0_1(0,1,0,0)*c[h_pl_1];
    float pyx5 = dyyIn_0_1(1,1,0,0)*c[h_pl_1];
    float pzx5 = dyyIn_2_3(0,1,0,0)*c[h_pl_1];

    float vxx5 = dyyIn_2_3(1,1,0,0)*c[h_pl_1];
    float vyx5 = dyyIn_4_5(0,1,0,0)*c[h_pl_1];
    float vzx5 = dyyIn_4_5(1,1,0,0)*c[h_pl_1];

    float pxy5 = dyyIn_0_1(0,0,1,0)*c[h_pl_1];
    float pyy5 = dyyIn_0_1(1,0,1,0)*c[h_pl_1];
    float pzy5 = dyyIn_2_3(0,0,1,0)*c[h_pl_1];

    float vxy5 = dyyIn_2_3(1,0,1,0)*c[h_pl_1];
    float vyy5 = dyyIn_4_5(0,0,1,0)*c[h_pl_1];
    float vzy5 = dyyIn_4_5(1,0,1,0)*c[h_pl_1];

    float pxz5 = dyyIn_0_1(0,0,0,1)*c[h_pl_1];
    float pyz5 = dyyIn_0_1(1,0,0,1)*c[h_pl_1];
    float pzz5 = dyyIn_2_3(0,0,0,1)*c[h_pl_1];

    float vxz5 = dyyIn_2_3(1,0,0,1)*c[h_pl_1];
    float vyz5 = dyyIn_4_5(0,0,0,1)*c[h_pl_1];
    float vzz5 = dyyIn_4_5(1,0,0,1)*c[h_pl_1];

    const unsigned short h_pl_2 = half_order + 2;
    float pxx6 = dyyIn_0_1(0,2,0,0)*c[h_pl_2];
    float pyx6 = dyyIn_0_1(1,2,0,0)*c[h_pl_2];
    float pzx6 = dyyIn_2_3(0,2,0,0)*c[h_pl_2];

    float vxx6 = dyyIn_2_3(1,2,0,0)*c[h_pl_2];
    float vyx6 = dyyIn_4_5(0,2,0,0)*c[h_pl_2];
    float vzx6 = dyyIn_4_5(1,2,0,0)*c[h_pl_2];

    float pxy6 = dyyIn_0_1(0,0,2,0)*c[h_pl_2];
    float pyy6 = dyyIn_0_1(1,0,2,0)*c[h_pl_2];
    float pzy6 = dyyIn_2_3(0,0,2,0)*c[h_pl_2];

    float vxy6 = dyyIn_2_3(1,0,2,0)*c[h_pl_2];
    float vyy6 = dyyIn_4_5(0,0,2,0)*c[h_pl_2];
    float vzy6 = dyyIn_4_5(1,0,2,0)*c[h_pl_2];

    float pxz6 = dyyIn_0_1(0,0,0,2)*c[h_pl_2];
    float pyz6 = dyyIn_0_1(1,0,0,2)*c[h_pl_2];
    float pzz6 = dyyIn_2_3(0,0,0,2)*c[h_pl_2];

    float vxz6 = dyyIn_2_3(1,0,0,2)*c[h_pl_2];
    float vyz6 = dyyIn_4_5(0,0,0,2)*c[h_pl_2];
    float vzz6 = dyyIn_4_5(1,0,0,2)*c[h_pl_2];

    const unsigned short h_pl_3 = half_order + 3;
    float pxx7 = dyyIn_0_1(0,3,0,0)*c[h_pl_3];
    float pyx7 = dyyIn_0_1(1,3,0,0)*c[h_pl_3];
    float pzx7 = dyyIn_2_3(0,3,0,0)*c[h_pl_3];

    float vxx7 = dyyIn_2_3(1,3,0,0)*c[h_pl_3];
    float vyx7 = dyyIn_4_5(0,3,0,0)*c[h_pl_3];
    float vzx7 = dyyIn_4_5(1,3,0,0)*c[h_pl_3];

    float pxy7 = dyyIn_0_1(0,0,3,0)*c[h_pl_3];
    float pyy7 = dyyIn_0_1(1,0,3,0)*c[h_pl_3];
    float pzy7 = dyyIn_2_3(0,0,3,0)*c[h_pl_3];

    float vxy7 = dyyIn_2_3(1,0,3,0)*c[h_pl_3];
    float vyy7 = dyyIn_4_5(0,0,3,0)*c[h_pl_3];
    float vzy7 = dyyIn_4_5(1,0,3,0)*c[h_pl_3];

    float pxz7 = dyyIn_0_1(0,0,0,3)*c[h_pl_3];
    float pyz7 = dyyIn_0_1(1,0,0,3)*c[h_pl_3];
    float pzz7 = dyyIn_2_3(0,0,0,3)*c[h_pl_3];

    float vxz7 = dyyIn_2_3(1,0,0,3)*c[h_pl_3];
    float vyz7 = dyyIn_4_5(0,0,0,3)*c[h_pl_3];
    float vzz7 = dyyIn_4_5(1,0,0,3)*c[h_pl_3];

    const unsigned short h_pl_4 = half_order + 4;
    float pxx8 = dyyIn_0_1(0,4,0,0)*c[h_pl_4];
    float pyx8 = dyyIn_0_1(1,4,0,0)*c[h_pl_4];
    float pzx8 = dyyIn_2_3(0,4,0,0)*c[h_pl_4];

    float vxx8 = dyyIn_2_3(1,4,0,0)*c[h_pl_4];
    float vyx8 = dyyIn_4_5(0,4,0,0)*c[h_pl_4];
    float vzx8 = dyyIn_4_5(1,4,0,0)*c[h_pl_4];

    float pxy8 = dyyIn_0_1(0,0,4,0)*c[h_pl_4];
    float pyy8 = dyyIn_0_1(1,0,4,0)*c[h_pl_4];
    float pzy8 = dyyIn_2_3(0,0,4,0)*c[h_pl_4];

    float vxy8 = dyyIn_2_3(1,0,4,0)*c[h_pl_4];
    float vyy8 = dyyIn_4_5(0,0,4,0)*c[h_pl_4];
    float vzy8 = dyyIn_4_5(1,0,4,0)*c[h_pl_4];

    float pxz8 = dyyIn_0_1(0,0,0,4)*c[h_pl_4];
    float pyz8 = dyyIn_0_1(1,0,0,4)*c[h_pl_4];
    float pzz8 = dyyIn_2_3(0,0,0,4)*c[h_pl_4];

    float vxz8 = dyyIn_2_3(1,0,0,4)*c[h_pl_4];
    float vyz8 = dyyIn_4_5(0,0,0,4)*c[h_pl_4];
    float vzz8 = dyyIn_4_5(1,0,0,4)*c[h_pl_4];

    float pxx9 = pxx0 + pxx1;
    float pxx10 = pxx2 + pxx3;
    float pxx11 = pxx4 + pxx5;
    float pxx12 = pxx6 + pxx7;
    float pxx13 = pxx8 + pxx9;
    float pxx14 = pxx10 + pxx11;
    float pxx15 = pxx12 + pxx13;
    pxx = pxx14 + pxx15;

    float pyx9 = pyx0 + pyx1;
    float pyx10 = pyx2 + pyx3;
    float pyx11 = pyx4 + pyx5;
    float pyx12 = pyx6 + pyx7;
    float pyx13 = pyx8 + pyx9;
    float pyx14 = pyx10 + pyx11;
    float pyx15 = pyx12 + pyx13;
    pyx = pyx14 + pyx15;

    float pzx9 = pzx0 + pzx1;
    float pzx10 = pzx2 + pzx3;
    float pzx11 = pzx4 + pzx5;
    float pzx12 = pzx6 + pzx7;
    float pzx13 = pzx8 + pzx9;
    float pzx14 = pzx10 + pzx11;
    float pzx15 = pzx12 + pzx13;
    pzx = pzx14 + pzx15;

    float vxx9 = vxx0 + vxx1;
    float vxx10 = vxx2 + vxx3;
    float vxx11 = vxx4 + vxx5;
    float vxx12 = vxx6 + vxx7;
    float vxx13 = vxx8 + vxx9;
    float vxx14 = vxx10 + vxx11;
    float vxx15 = vxx12 + vxx13;
    vxx = vxx14 + vxx15;

    float vyx9 = vyx0 + vyx1;
    float vyx10 = vyx2 + vyx3;
    float vyx11 = vyx4 + vyx5;
    float vyx12 = vyx6 + vyx7;
    float vyx13 = vyx8 + vyx9;
    float vyx14 = vyx10 + vyx11;
    float vyx15 = vyx12 + vyx13;
    vyx = vyx14 + vyx15;

    float vzx9 = vzx0 + vzx1;
    float vzx10 = vzx2 + vzx3;
    float vzx11 = vzx4 + vzx5;
    float vzx12 = vzx6 + vzx7;
    float vzx13 = vzx8 + vzx9;
    float vzx14 = vzx10 + vzx11;
    float vzx15 = vzx12 + vzx13;
    vzx = vzx14 + vzx15;

    float pxy9 = pxy0 + pxy1;
    float pxy10 = pxy2 + pxy3;
    float pxy11 = pxy4 + pxy5;
    float pxy12 = pxy6 + pxy7;
    float pxy13 = pxy8 + pxy9;
    float pxy14 = pxy10 + pxy11;
    float pxy15 = pxy12 + pxy13;
    pxy = pxy14 + pxy15;
    
    float pyy9 = pyy0 + pyy1;
    float pyy10 = pyy2 + pyy3;
    float pyy11 = pyy4 + pyy5;
    float pyy12 = pyy6 + pyy7;
    float pyy13 = pyy8 + pyy9;
    float pyy14 = pyy10 + pyy11;
    float pyy15 = pyy12 + pyy13;
    pyy = pyy14 + pyy15;

    float pzy9 = pzy0 + pzy1;
    float pzy10 = pzy2 + pzy3;
    float pzy11 = pzy4 + pzy5;
    float pzy12 = pzy6 + pzy7;
    float pzy13 = pzy8 + pzy9;
    float pzy14 = pzy10 + pzy11;
    float pzy15 = pzy12 + pzy13;
    pzy = pzy14 + pzy15;

    float vxy9 = vxy0 + vxy1;
    float vxy10 = vxy2 + vxy3;
    float vxy11 = vxy4 + vxy5;
    float vxy12 = vxy6 + vxy7;
    float vxy13 = vxy8 + vxy9;
    float vxy14 = vxy10 + vxy11;
    float vxy15 = vxy12 + vxy13;
    vxy = vxy14 + vxy15;

    float vyy9 = vyy0 + vyy1;
    float vyy10 = vyy2 + vyy3;
    float vyy11 = vyy4 + vyy5;
    float vyy12 = vyy6 + vyy7;
    float vyy13 = vyy8 + vyy9;
    float vyy14 = vyy10 + vyy11;
    float vyy15 = vyy12 + vyy13;
    vyy = vyy14 + vyy15;

    float pxz9 = pxz0 + pxz1;
    float pxz10 = pxz2 + pxz3;
    float pxz11 = pxz4 + pxz5;
    float pxz12 = pxz6 + pxz7;
    float pxz13 = pxz8 + pxz9;
    float pxz14 = pxz10 + pxz11;
    float pxz15 = pxz12 + pxz13;
    pxz = pxz14 + pxz15;

    float pyz9 = pyz0 + pyz1;
    float pyz10 = pyz2 + pyz3;
    float pyz11 = pyz4 + pyz5;
    float pyz12 = pyz6 + pyz7;
    float pyz13 = pyz8 + pyz9;
    float pyz14 = pyz10 + pyz11;
    float pyz15 = pyz12 + pyz13;
    pyz = pyz14 + pyz15;

    float pzz9 = pzz0 + pzz1;
    float pzz10 = pzz2 + pzz3;
    float pzz11 = pzz4 + pzz5;
    float pzz12 = pzz6 + pzz7;
    float pzz13 = pzz8 + pzz9;
    float pzz14 = pzz10 + pzz11;
    float pzz15 = pzz12 + pzz13;
    pzz = pzz14 + pzz15;

    float vxz9 = vxz0 + vxz1;
    float vxz10 = vxz2 + vxz3;
    float vxz11 = vxz4 + vxz5;
    float vxz12 = vxz6 + vxz7;
    float vxz13 = vxz8 + vxz9;
    float vxz14 = vxz10 + vxz11;
    float vxz15 = vxz12 + vxz13;
    vxz = vxz14 + vxz15;

    float vyz9 = vyz0 + vyz1;
    float vyz10 = vyz2 + vyz3;
    float vyz11 = vyz4 + vyz5;
    float vyz12 = vyz6 + vyz7;
    float vyz13 = vyz8 + vyz9;
    float vyz14 = vyz10 + vyz11;
    float vyz15 = vyz12 + vyz13;
    vyz = vyz14 + vyz15;

    float vzz9 = vzz0 + vzz1;
    float vzz10 = vzz2 + vzz3;
    float vzz11 = vzz4 + vzz5;
    float vzz12 = vzz6 + vzz7;
    float vzz13 = vzz8 + vzz9;
    float vzz14 = vzz10 + vzz11;
    float vzz15 = vzz12 + vzz13;
    vzz = vzz14 + vzz15;

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
    
    float vxx_div_rho = vxx/rho_mu(0,0,0,0);
    float vyy_div_rho = vyy/rho_mu(0,0,0,0);
    float vzz_div_rho = vzz/rho_mu(0,0,0,0);

    float add_pxx_pyx_pxz0 = pxx+pyx;
    float add_pxx_pyx_pxz = add_pxx_pyx_pxz0 + pxz;
    float add_pxx_pyx_pxz_mul_mu = add_pxx_pyx_pxz*rho_mu(1,0,0,0);
    float add_pxy_pyy_pyz0 = pxy+pyy;
    float add_pxy_pyy_pyz = add_pxy_pyy_pyz0 + pyz;
    float add_pxy_pyy_pyz_mul_mu = add_pxy_pyy_pyz*rho_mu(1,0,0,0);
    float add_pxz_pyz_pzz0 = pxz+pyz;
    float add_pxz_pyz_pzz = add_pxz_pyz_pzz0 + pzz;
    float add_pxz_pyz_pzz_mul_mu = add_pxz_pyz_pzz*rho_mu(1,0,0,0);

    float ytemp0 =(vxx_div_rho - sigmax_mul_px) * *dt;
    float ytemp3 =(add_pxx_pyx_pxz_mul_mu - sigmax_mul_vx)* *dt;
    
    float ytemp1 =(vyy_div_rho - sigmay_mul_py)* *dt;
    float ytemp4 =(add_pxy_pyy_pyz_mul_mu - sigmay_mul_vy)* *dt;
    
    float ytemp2 =(vzz_div_rho - sigmaz_mul_pz)* *dt;
    float ytemp5 =(add_pxz_pyz_pzz_mul_mu - sigmaz_mul_vz)* *dt;

    float scale1_ytemp0 = ytemp0 * *scale1;
    float scale1_ytemp1 = ytemp1 * *scale1;
    float scale1_ytemp2 = ytemp2 * *scale1;
    float scale1_ytemp3 = ytemp3 * *scale1;
    float scale1_ytemp4 = ytemp4 * *scale1;
    float scale1_ytemp5 = ytemp5 * *scale1;


    dyyOut_0_1(0,0,0,0) = yy_0_add_sum_0 + scale1_ytemp0;
    dyyOut_2_3(1,0,0,0) = yy_3_add_sum_3 + scale1_ytemp3;
    dyyOut_0_1(1,0,0,0) = yy_1_add_sum_1 + scale1_ytemp1;
    dyyOut_4_5(0,0,0,0) = yy_4_add_sum_4 + scale1_ytemp4;
    dyyOut_2_3(0,0,0,0) = yy_2_add_sum_2 + scale1_ytemp2;
    dyyOut_4_5(1,0,0,0) = yy_5_add_sum_5 + scale1_ytemp5;


    sum_0_1(0,0,0,0) += ytemp0 * *scale2;
    sum_2_3(1,0,0,0) += ytemp3 * *scale2;
    sum_0_1(1,0,0,0) += ytemp1 * *scale2;
    sum_4_5(0,0,0,0) += ytemp4 * *scale2;
    sum_2_3(0,0,0,0) += ytemp2 * *scale2;
    sum_4_5(1,0,0,0) += ytemp5 * *scale2;
}

void fd3d_pml_kernel3(const int *dispx, const int *dispy, const int *dispz, const int *idx, float* dt,  float* scale1, float* scale2, const ACC<float>& rho_mu, 
        const ACC<float>& yy_0_1, const ACC<float>& yy_2_3, const ACC<float>& yy_4_5,
        const ACC<float>& dyyIn_0_1, const ACC<float>& dyyIn_2_3, const ACC<float>& dyyIn_4_5,
        ACC<float>& dyyOut_0_1, ACC<float>& dyyOut_2_3, ACC<float>& dyyOut_4_5,
        const ACC<float>& sum_0_1, const ACC<float>& sum_2_3, const ACC<float>& sum_4_5) {
  
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

    float sigma = rho_mu(1,0,0,0)/rho_mu(0,0,0,0);
    float sigma_10_percent = sigma * 0.1f;
    float sigmax=0.0;
    float sigmay=0.0;
    float sigmaz=0.0;
    
    if(idx[0]<=xbeg+pml_width){

        sigmax = (xpmlbeg-idx[0])*sigma_10_percent;///pml_width;
    }
    else if(idx[0]>=xend-pml_width){
        sigmax=(idx[0]-xpmlend)*sigma_10_percent;///pml_width;
    }
    if(idx[1]<=ybeg+pml_width){
        sigmay=(ypmlbeg-idx[1])*sigma_10_percent;///pml_width;
    }
    else if(idx[1]>=yend-pml_width){
        sigmay=(idx[1]-ypmlend)*sigma_10_percent;///pml_width;
    }
    if(idx[2]<=zbeg+pml_width){
        sigmaz=(zpmlbeg-idx[2])*sigma_10_percent;///pml_width;
    }
    else if(idx[2]>=zend-pml_width){
        sigmaz=(idx[2]-zpmlend)*sigma_10_percent;///pml_width;
    }

            //sigmax=0.0;
            //sigmay=0.0;
    
    float px = dyyIn_0_1(0,0,0,0);
    float py = dyyIn_0_1(1,0,0,0);
    float pz = dyyIn_2_3(0,0,0,0);
    
    float vx = dyyIn_2_3(1,0,0,0);
    float vy = dyyIn_4_5(0,0,0,0);
    float vz = dyyIn_4_5(1,0,0,0);
    
    float yy_0_add_sum_0 = yy_0_1(0,0,0,0) + sum_0_1(0,0,0,0);
    float yy_1_add_sum_1 = yy_0_1(1,0,0,0) + sum_0_1(1,0,0,0);
    float yy_2_add_sum_2 = yy_2_3(0,0,0,0) + sum_2_3(0,0,0,0);
    float yy_3_add_sum_3 = yy_2_3(1,0,0,0) + sum_2_3(1,0,0,0);
    float yy_4_add_sum_4 = yy_4_5(0,0,0,0) + sum_4_5(0,0,0,0);
    float yy_5_add_sum_5 = yy_4_5(1,0,0,0) + sum_4_5(1,0,0,0);

    float sigmax_mul_px = sigmax * px;
    float sigmay_mul_py = sigmay * py;
    float sigmaz_mul_pz = sigmaz * pz;
    float sigmax_mul_vx = sigmax * vx;
    float sigmay_mul_vy = sigmay * vy;
    float sigmaz_mul_vz = sigmaz * vz;

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

    const unsigned short h_min_4 = half_order - 4;
    float pxx0 = dyyIn_0_1(0,-4,0,0)*c[h_min_4];
    float pyx0 = dyyIn_0_1(1,-4,0,0)*c[h_min_4];
    float pzx0 = dyyIn_2_3(0,-4,0,0)*c[h_min_4];

    float vxx0 = dyyIn_2_3(1,-4,0,0)*c[h_min_4];
    float vyx0 = dyyIn_4_5(0,-4,0,0)*c[h_min_4];
    float vzx0 = dyyIn_4_5(1,-4,0,0)*c[h_min_4];

    float pxy0 = dyyIn_0_1(0,0,-4,0)*c[h_min_4];
    float pyy0 = dyyIn_0_1(1,0,-4,0)*c[h_min_4];
    float pzy0 = dyyIn_2_3(0,0,-4,0)*c[h_min_4];

    float vxy0 = dyyIn_2_3(1,0,-4,0)*c[h_min_4];
    float vyy0 = dyyIn_4_5(0,0,-4,0)*c[h_min_4];
    float vzy0 = dyyIn_4_5(1,0,-4,0)*c[h_min_4];

    float pxz0 = dyyIn_0_1(0,0,0,-4)*c[h_min_4];
    float pyz0 = dyyIn_0_1(1,0,0,-4)*c[h_min_4];
    float pzz0 = dyyIn_2_3(0,0,0,-4)*c[h_min_4];

    float vxz0 = dyyIn_2_3(1,0,0,-4)*c[h_min_4];
    float vyz0 = dyyIn_4_5(0,0,0,-4)*c[h_min_4];
    float vzz0 = dyyIn_4_5(1,0,0,-4)*c[h_min_4];

    const unsigned short h_min_3 = half_order - 3;
    float pxx1 = dyyIn_0_1(0,-3,0,0)*c[h_min_3];
    float pyx1 = dyyIn_0_1(1,-3,0,0)*c[h_min_3];
    float pzx1 = dyyIn_2_3(0,-3,0,0)*c[h_min_3];

    float vxx1 = dyyIn_2_3(1,-3,0,0)*c[h_min_3];
    float vyx1 = dyyIn_4_5(0,-3,0,0)*c[h_min_3];
    float vzx1 = dyyIn_4_5(1,-3,0,0)*c[h_min_3];

    float pxy1 = dyyIn_0_1(0,0,-3,0)*c[h_min_3];
    float pyy1 = dyyIn_0_1(1,0,-3,0)*c[h_min_3];
    float pzy1 = dyyIn_2_3(0,0,-3,0)*c[h_min_3];

    float vxy1 = dyyIn_2_3(1,0,-3,0)*c[h_min_3];
    float vyy1 = dyyIn_4_5(0,0,-3,0)*c[h_min_3];
    float vzy1 = dyyIn_4_5(1,0,-3,0)*c[h_min_3];

    float pxz1 = dyyIn_0_1(0,0,0,-3)*c[h_min_3];
    float pyz1 = dyyIn_0_1(1,0,0,-3)*c[h_min_3];
    float pzz1 = dyyIn_2_3(0,0,0,-3)*c[h_min_3];

    float vxz1 = dyyIn_2_3(1,0,0,-3)*c[h_min_3];
    float vyz1 = dyyIn_4_5(0,0,0,-3)*c[h_min_3];
    float vzz1 = dyyIn_4_5(1,0,0,-3)*c[h_min_3];

    const unsigned short h_min_2 = half_order - 2;
    float pxx2 = dyyIn_0_1(0,-2,0,0)*c[h_min_2];
    float pyx2 = dyyIn_0_1(1,-2,0,0)*c[h_min_2];
    float pzx2 = dyyIn_2_3(0,-2,0,0)*c[h_min_2];

    float vxx2 = dyyIn_2_3(1,-2,0,0)*c[h_min_2];
    float vyx2 = dyyIn_4_5(0,-2,0,0)*c[h_min_2];
    float vzx2 = dyyIn_4_5(1,-2,0,0)*c[h_min_2];

    float pxy2 = dyyIn_0_1(0,0,-2,0)*c[h_min_2];
    float pyy2 = dyyIn_0_1(1,0,-2,0)*c[h_min_2];
    float pzy2 = dyyIn_2_3(0,0,-2,0)*c[h_min_2];

    float vxy2 = dyyIn_2_3(1,0,-2,0)*c[h_min_2];
    float vyy2 = dyyIn_4_5(0,0,-2,0)*c[h_min_2];
    float vzy2 = dyyIn_4_5(1,0,-2,0)*c[h_min_2];

    float pxz2 = dyyIn_0_1(0,0,0,-2)*c[h_min_2];
    float pyz2 = dyyIn_0_1(1,0,0,-2)*c[h_min_2];
    float pzz2 = dyyIn_2_3(0,0,0,-2)*c[h_min_2];

    float vxz2 = dyyIn_2_3(1,0,0,-2)*c[h_min_2];
    float vyz2 = dyyIn_4_5(0,0,0,-2)*c[h_min_2];
    float vzz2 = dyyIn_4_5(1,0,0,-2)*c[h_min_2];

    const unsigned short h_min_1 = half_order - 1;
    float pxx3 = dyyIn_0_1(0,-1,0,0)*c[h_min_1];
    float pyx3 = dyyIn_0_1(1,-1,0,0)*c[h_min_1];
    float pzx3 = dyyIn_2_3(0,-1,0,0)*c[h_min_1];

    float vxx3 = dyyIn_2_3(1,-1,0,0)*c[h_min_1];
    float vyx3 = dyyIn_4_5(0,-1,0,0)*c[h_min_1];
    float vzx3 = dyyIn_4_5(1,-1,0,0)*c[h_min_1];

    float pxy3 = dyyIn_0_1(0,0,-1,0)*c[h_min_1];
    float pyy3 = dyyIn_0_1(1,0,-1,0)*c[h_min_1];
    float pzy3 = dyyIn_2_3(0,0,-1,0)*c[h_min_1];

    float vxy3 = dyyIn_2_3(1,0,-1,0)*c[h_min_1];
    float vyy3 = dyyIn_4_5(0,0,-1,0)*c[h_min_1];
    float vzy3 = dyyIn_4_5(1,0,-1,0)*c[h_min_1];

    float pxz3 = dyyIn_0_1(0,0,0,-1)*c[h_min_1];
    float pyz3 = dyyIn_0_1(1,0,0,-1)*c[h_min_1];
    float pzz3 = dyyIn_2_3(0,0,0,-1)*c[h_min_1];

    float vxz3 = dyyIn_2_3(1,0,0,-1)*c[h_min_1];
    float vyz3 = dyyIn_4_5(0,0,0,-1)*c[h_min_1];
    float vzz3 = dyyIn_4_5(1,0,0,-1)*c[h_min_1];

    float pxx4 = dyyIn_0_1(0,0,0,0)*c[half_order];
    float pyx4 = dyyIn_0_1(1,0,0,0)*c[half_order];
    float pzx4 = dyyIn_2_3(0,0,0,0)*c[half_order];

    float vxx4 = dyyIn_2_3(1,0,0,0)*c[half_order];
    float vyx4 = dyyIn_4_5(0,0,0,0)*c[half_order];
    float vzx4 = dyyIn_4_5(1,0,0,0)*c[half_order];

    float pxy4 = dyyIn_0_1(0,0,0,0)*c[half_order];
    float pyy4 = dyyIn_0_1(1,0,0,0)*c[half_order];
    float pzy4 = dyyIn_2_3(0,0,0,0)*c[half_order];

    float vxy4 = dyyIn_2_3(1,0,0,0)*c[half_order];
    float vyy4 = dyyIn_4_5(0,0,0,0)*c[half_order];
    float vzy4 = dyyIn_4_5(1,0,0,0)*c[half_order];

    float pxz4 = dyyIn_0_1(0,0,0,0)*c[half_order];
    float pyz4 = dyyIn_0_1(1,0,0,0)*c[half_order];
    float pzz4 = dyyIn_2_3(0,0,0,0)*c[half_order];

    float vxz4 = dyyIn_2_3(1,0,0,0)*c[half_order];
    float vyz4 = dyyIn_4_5(0,0,0,0)*c[half_order];
    float vzz4 = dyyIn_4_5(1,0,0,0)*c[half_order];

    const unsigned short h_pl_1 = half_order + 1;
    float pxx5 = dyyIn_0_1(0,1,0,0)*c[h_pl_1];
    float pyx5 = dyyIn_0_1(1,1,0,0)*c[h_pl_1];
    float pzx5 = dyyIn_2_3(0,1,0,0)*c[h_pl_1];

    float vxx5 = dyyIn_2_3(1,1,0,0)*c[h_pl_1];
    float vyx5 = dyyIn_4_5(0,1,0,0)*c[h_pl_1];
    float vzx5 = dyyIn_4_5(1,1,0,0)*c[h_pl_1];

    float pxy5 = dyyIn_0_1(0,0,1,0)*c[h_pl_1];
    float pyy5 = dyyIn_0_1(1,0,1,0)*c[h_pl_1];
    float pzy5 = dyyIn_2_3(0,0,1,0)*c[h_pl_1];

    float vxy5 = dyyIn_2_3(1,0,1,0)*c[h_pl_1];
    float vyy5 = dyyIn_4_5(0,0,1,0)*c[h_pl_1];
    float vzy5 = dyyIn_4_5(1,0,1,0)*c[h_pl_1];

    float pxz5 = dyyIn_0_1(0,0,0,1)*c[h_pl_1];
    float pyz5 = dyyIn_0_1(1,0,0,1)*c[h_pl_1];
    float pzz5 = dyyIn_2_3(0,0,0,1)*c[h_pl_1];

    float vxz5 = dyyIn_2_3(1,0,0,1)*c[h_pl_1];
    float vyz5 = dyyIn_4_5(0,0,0,1)*c[h_pl_1];
    float vzz5 = dyyIn_4_5(1,0,0,1)*c[h_pl_1];

    const unsigned short h_pl_2 = half_order + 2;
    float pxx6 = dyyIn_0_1(0,2,0,0)*c[h_pl_2];
    float pyx6 = dyyIn_0_1(1,2,0,0)*c[h_pl_2];
    float pzx6 = dyyIn_2_3(0,2,0,0)*c[h_pl_2];

    float vxx6 = dyyIn_2_3(1,2,0,0)*c[h_pl_2];
    float vyx6 = dyyIn_4_5(0,2,0,0)*c[h_pl_2];
    float vzx6 = dyyIn_4_5(1,2,0,0)*c[h_pl_2];

    float pxy6 = dyyIn_0_1(0,0,2,0)*c[h_pl_2];
    float pyy6 = dyyIn_0_1(1,0,2,0)*c[h_pl_2];
    float pzy6 = dyyIn_2_3(0,0,2,0)*c[h_pl_2];

    float vxy6 = dyyIn_2_3(1,0,2,0)*c[h_pl_2];
    float vyy6 = dyyIn_4_5(0,0,2,0)*c[h_pl_2];
    float vzy6 = dyyIn_4_5(1,0,2,0)*c[h_pl_2];

    float pxz6 = dyyIn_0_1(0,0,0,2)*c[h_pl_2];
    float pyz6 = dyyIn_0_1(1,0,0,2)*c[h_pl_2];
    float pzz6 = dyyIn_2_3(0,0,0,2)*c[h_pl_2];

    float vxz6 = dyyIn_2_3(1,0,0,2)*c[h_pl_2];
    float vyz6 = dyyIn_4_5(0,0,0,2)*c[h_pl_2];
    float vzz6 = dyyIn_4_5(1,0,0,2)*c[h_pl_2];

    const unsigned short h_pl_3 = half_order + 3;
    float pxx7 = dyyIn_0_1(0,3,0,0)*c[h_pl_3];
    float pyx7 = dyyIn_0_1(1,3,0,0)*c[h_pl_3];
    float pzx7 = dyyIn_2_3(0,3,0,0)*c[h_pl_3];

    float vxx7 = dyyIn_2_3(1,3,0,0)*c[h_pl_3];
    float vyx7 = dyyIn_4_5(0,3,0,0)*c[h_pl_3];
    float vzx7 = dyyIn_4_5(1,3,0,0)*c[h_pl_3];

    float pxy7 = dyyIn_0_1(0,0,3,0)*c[h_pl_3];
    float pyy7 = dyyIn_0_1(1,0,3,0)*c[h_pl_3];
    float pzy7 = dyyIn_2_3(0,0,3,0)*c[h_pl_3];

    float vxy7 = dyyIn_2_3(1,0,3,0)*c[h_pl_3];
    float vyy7 = dyyIn_4_5(0,0,3,0)*c[h_pl_3];
    float vzy7 = dyyIn_4_5(1,0,3,0)*c[h_pl_3];

    float pxz7 = dyyIn_0_1(0,0,0,3)*c[h_pl_3];
    float pyz7 = dyyIn_0_1(1,0,0,3)*c[h_pl_3];
    float pzz7 = dyyIn_2_3(0,0,0,3)*c[h_pl_3];

    float vxz7 = dyyIn_2_3(1,0,0,3)*c[h_pl_3];
    float vyz7 = dyyIn_4_5(0,0,0,3)*c[h_pl_3];
    float vzz7 = dyyIn_4_5(1,0,0,3)*c[h_pl_3];

    const unsigned short h_pl_4 = half_order + 4;
    float pxx8 = dyyIn_0_1(0,4,0,0)*c[h_pl_4];
    float pyx8 = dyyIn_0_1(1,4,0,0)*c[h_pl_4];
    float pzx8 = dyyIn_2_3(0,4,0,0)*c[h_pl_4];

    float vxx8 = dyyIn_2_3(1,4,0,0)*c[h_pl_4];
    float vyx8 = dyyIn_4_5(0,4,0,0)*c[h_pl_4];
    float vzx8 = dyyIn_4_5(1,4,0,0)*c[h_pl_4];

    float pxy8 = dyyIn_0_1(0,0,4,0)*c[h_pl_4];
    float pyy8 = dyyIn_0_1(1,0,4,0)*c[h_pl_4];
    float pzy8 = dyyIn_2_3(0,0,4,0)*c[h_pl_4];

    float vxy8 = dyyIn_2_3(1,0,4,0)*c[h_pl_4];
    float vyy8 = dyyIn_4_5(0,0,4,0)*c[h_pl_4];
    float vzy8 = dyyIn_4_5(1,0,4,0)*c[h_pl_4];

    float pxz8 = dyyIn_0_1(0,0,0,4)*c[h_pl_4];
    float pyz8 = dyyIn_0_1(1,0,0,4)*c[h_pl_4];
    float pzz8 = dyyIn_2_3(0,0,0,4)*c[h_pl_4];

    float vxz8 = dyyIn_2_3(1,0,0,4)*c[h_pl_4];
    float vyz8 = dyyIn_4_5(0,0,0,4)*c[h_pl_4];
    float vzz8 = dyyIn_4_5(1,0,0,4)*c[h_pl_4];

    float pxx9 = pxx0 + pxx1;
    float pxx10 = pxx2 + pxx3;
    float pxx11 = pxx4 + pxx5;
    float pxx12 = pxx6 + pxx7;
    float pxx13 = pxx8 + pxx9;
    float pxx14 = pxx10 + pxx11;
    float pxx15 = pxx12 + pxx13;
    pxx = pxx14 + pxx15;

    float pyx9 = pyx0 + pyx1;
    float pyx10 = pyx2 + pyx3;
    float pyx11 = pyx4 + pyx5;
    float pyx12 = pyx6 + pyx7;
    float pyx13 = pyx8 + pyx9;
    float pyx14 = pyx10 + pyx11;
    float pyx15 = pyx12 + pyx13;
    pyx = pyx14 + pyx15;

    float pzx9 = pzx0 + pzx1;
    float pzx10 = pzx2 + pzx3;
    float pzx11 = pzx4 + pzx5;
    float pzx12 = pzx6 + pzx7;
    float pzx13 = pzx8 + pzx9;
    float pzx14 = pzx10 + pzx11;
    float pzx15 = pzx12 + pzx13;
    pzx = pzx14 + pzx15;

    float vxx9 = vxx0 + vxx1;
    float vxx10 = vxx2 + vxx3;
    float vxx11 = vxx4 + vxx5;
    float vxx12 = vxx6 + vxx7;
    float vxx13 = vxx8 + vxx9;
    float vxx14 = vxx10 + vxx11;
    float vxx15 = vxx12 + vxx13;
    vxx = vxx14 + vxx15;

    float vyx9 = vyx0 + vyx1;
    float vyx10 = vyx2 + vyx3;
    float vyx11 = vyx4 + vyx5;
    float vyx12 = vyx6 + vyx7;
    float vyx13 = vyx8 + vyx9;
    float vyx14 = vyx10 + vyx11;
    float vyx15 = vyx12 + vyx13;
    vyx = vyx14 + vyx15;

    float vzx9 = vzx0 + vzx1;
    float vzx10 = vzx2 + vzx3;
    float vzx11 = vzx4 + vzx5;
    float vzx12 = vzx6 + vzx7;
    float vzx13 = vzx8 + vzx9;
    float vzx14 = vzx10 + vzx11;
    float vzx15 = vzx12 + vzx13;
    vzx = vzx14 + vzx15;

    float pxy9 = pxy0 + pxy1;
    float pxy10 = pxy2 + pxy3;
    float pxy11 = pxy4 + pxy5;
    float pxy12 = pxy6 + pxy7;
    float pxy13 = pxy8 + pxy9;
    float pxy14 = pxy10 + pxy11;
    float pxy15 = pxy12 + pxy13;
    pxy = pxy14 + pxy15;
    
    float pyy9 = pyy0 + pyy1;
    float pyy10 = pyy2 + pyy3;
    float pyy11 = pyy4 + pyy5;
    float pyy12 = pyy6 + pyy7;
    float pyy13 = pyy8 + pyy9;
    float pyy14 = pyy10 + pyy11;
    float pyy15 = pyy12 + pyy13;
    pyy = pyy14 + pyy15;

    float pzy9 = pzy0 + pzy1;
    float pzy10 = pzy2 + pzy3;
    float pzy11 = pzy4 + pzy5;
    float pzy12 = pzy6 + pzy7;
    float pzy13 = pzy8 + pzy9;
    float pzy14 = pzy10 + pzy11;
    float pzy15 = pzy12 + pzy13;
    pzy = pzy14 + pzy15;

    float vxy9 = vxy0 + vxy1;
    float vxy10 = vxy2 + vxy3;
    float vxy11 = vxy4 + vxy5;
    float vxy12 = vxy6 + vxy7;
    float vxy13 = vxy8 + vxy9;
    float vxy14 = vxy10 + vxy11;
    float vxy15 = vxy12 + vxy13;
    vxy = vxy14 + vxy15;

    float vyy9 = vyy0 + vyy1;
    float vyy10 = vyy2 + vyy3;
    float vyy11 = vyy4 + vyy5;
    float vyy12 = vyy6 + vyy7;
    float vyy13 = vyy8 + vyy9;
    float vyy14 = vyy10 + vyy11;
    float vyy15 = vyy12 + vyy13;
    vyy = vyy14 + vyy15;

    float pxz9 = pxz0 + pxz1;
    float pxz10 = pxz2 + pxz3;
    float pxz11 = pxz4 + pxz5;
    float pxz12 = pxz6 + pxz7;
    float pxz13 = pxz8 + pxz9;
    float pxz14 = pxz10 + pxz11;
    float pxz15 = pxz12 + pxz13;
    pxz = pxz14 + pxz15;

    float pyz9 = pyz0 + pyz1;
    float pyz10 = pyz2 + pyz3;
    float pyz11 = pyz4 + pyz5;
    float pyz12 = pyz6 + pyz7;
    float pyz13 = pyz8 + pyz9;
    float pyz14 = pyz10 + pyz11;
    float pyz15 = pyz12 + pyz13;
    pyz = pyz14 + pyz15;

    float pzz9 = pzz0 + pzz1;
    float pzz10 = pzz2 + pzz3;
    float pzz11 = pzz4 + pzz5;
    float pzz12 = pzz6 + pzz7;
    float pzz13 = pzz8 + pzz9;
    float pzz14 = pzz10 + pzz11;
    float pzz15 = pzz12 + pzz13;
    pzz = pzz14 + pzz15;

    float vxz9 = vxz0 + vxz1;
    float vxz10 = vxz2 + vxz3;
    float vxz11 = vxz4 + vxz5;
    float vxz12 = vxz6 + vxz7;
    float vxz13 = vxz8 + vxz9;
    float vxz14 = vxz10 + vxz11;
    float vxz15 = vxz12 + vxz13;
    vxz = vxz14 + vxz15;

    float vyz9 = vyz0 + vyz1;
    float vyz10 = vyz2 + vyz3;
    float vyz11 = vyz4 + vyz5;
    float vyz12 = vyz6 + vyz7;
    float vyz13 = vyz8 + vyz9;
    float vyz14 = vyz10 + vyz11;
    float vyz15 = vyz12 + vyz13;
    vyz = vyz14 + vyz15;

    float vzz9 = vzz0 + vzz1;
    float vzz10 = vzz2 + vzz3;
    float vzz11 = vzz4 + vzz5;
    float vzz12 = vzz6 + vzz7;
    float vzz13 = vzz8 + vzz9;
    float vzz14 = vzz10 + vzz11;
    float vzz15 = vzz12 + vzz13;
    vzz = vzz14 + vzz15;


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
    
    float vxx_div_rho = vxx/rho_mu(0,0,0,0);
    float vyy_div_rho = vyy/rho_mu(0,0,0,0);
    float vzz_div_rho = vzz/rho_mu(0,0,0,0);

    float add_pxx_pyx_pxz0 = pxx+pyx;
    float add_pxx_pyx_pxz = add_pxx_pyx_pxz0 + pxz;
    float add_pxx_pyx_pxz_mul_mu = add_pxx_pyx_pxz*rho_mu(1,0,0,0);
    float add_pxy_pyy_pyz0 = pxy+pyy;
    float add_pxy_pyy_pyz = add_pxy_pyy_pyz0 + pyz;
    float add_pxy_pyy_pyz_mul_mu = add_pxy_pyy_pyz*rho_mu(1,0,0,0);
    float add_pxz_pyz_pzz0 = pxz+pyz;
    float add_pxz_pyz_pzz = add_pxz_pyz_pzz0 + pzz;
    float add_pxz_pyz_pzz_mul_mu = add_pxz_pyz_pzz*rho_mu(1,0,0,0);

    float ytemp0 =(vxx_div_rho - sigmax_mul_px) * *dt;
    float ytemp3 =(add_pxx_pyx_pxz_mul_mu - sigmax_mul_vx)* *dt;
    
    float ytemp1 =(vyy_div_rho - sigmay_mul_py)* *dt;
    float ytemp4 =(add_pxy_pyy_pyz_mul_mu - sigmay_mul_vy)* *dt;
    
    float ytemp2 =(vzz_div_rho - sigmaz_mul_pz)* *dt;
    float ytemp5 =(add_pxz_pyz_pzz_mul_mu - sigmaz_mul_vz)* *dt;

    float scale2_ytemp0 = ytemp0 * *scale2;
    float scale2_ytemp1 = ytemp1 * *scale2;
    float scale2_ytemp2 = ytemp2 * *scale2;
    float scale2_ytemp3 = ytemp3 * *scale2;
    float scale2_ytemp4 = ytemp4 * *scale2;
    float scale2_ytemp5 = ytemp5 * *scale2;


    dyyOut_0_1(0,0,0,0) = yy_0_add_sum_0 + scale2_ytemp0;
    dyyOut_2_3(1,0,0,0) = yy_3_add_sum_3 + scale2_ytemp3;
    dyyOut_0_1(1,0,0,0) = yy_1_add_sum_1 + scale2_ytemp1;
    dyyOut_4_5(0,0,0,0) = yy_4_add_sum_4 + scale2_ytemp4;
    dyyOut_2_3(0,0,0,0) = yy_2_add_sum_2 + scale2_ytemp2;
    dyyOut_4_5(1,0,0,0) = yy_5_add_sum_5 + scale2_ytemp5;
    
}
