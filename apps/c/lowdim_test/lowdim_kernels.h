#ifndef LOWDIM_KERNEL_H
#define LOWDIM_KERNEL_H

void set_val(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}

void set_valXY(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}

void set_valYZ(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}

void set_valXZ(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}

void set_valX(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}
void set_valY(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}
void set_valZ(ACC<double> &dat, const double *val)
{
    dat(0,0,0) = *val;
}




//void set3D(ACC<double> &dat,  const int* sizes, const int *idx)
//{    
//    dat(0,0,0) = idx[0] + idx[1]*sizes[0] + idx[2]*sizes[0]*sizes[1];
//}

void set3D(ACC<double> &dat,  const int* sizes, const int *idx)
{
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};

    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];

    dat(0,0,0) = shifted_idx[0] + shifted_idx[1]*sizes[0] + shifted_idx[2]*sizes[0]*sizes[1];
}

void check2D_XY_max(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[2]= {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    expectedValue =(sizes[2]-1)*sizes[1]*sizes[0] + sizes[0]*shifted_idx[1] + shifted_idx[0];

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check2D_XZ_max(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = shifted_idx[2]*sizes[0]*sizes[1] + (sizes[1]-1)*sizes[0] + shifted_idx[0];

  
    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check2D_YZ_max(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = shifted_idx[2]*sizes[0]*sizes[1] + shifted_idx[1]*sizes[0] + (sizes[0]-1);

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}




void check2D_XY_min(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue =(0)*sizes[1]*sizes[0] + sizes[0]*shifted_idx[1] + shifted_idx[0];

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check2D_XZ_min(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = shifted_idx[2]*sizes[0]*sizes[1] + (0)*sizes[0] + shifted_idx[0];

  
    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check2D_YZ_min(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = shifted_idx[2]*sizes[0]*sizes[1] + shifted_idx[1]*sizes[0] + (0);

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}




void check2D_XY_inc(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[2]= {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    int start_value= shifted_idx[0] + shifted_idx[1]*sizes[0];
    int end_value= start_value+sizes[0]*sizes[1]*(sizes[2]-1);
    expectedValue = (sizes[2]/2)*(start_value+end_value);

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check2D_XZ_inc(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[2]= {idx[0] + sizes[0]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    int start_value= shifted_idx[0];
    int end_value= start_value+sizes[1]*(sizes[2]-1);
    expectedValue = (sizes[1]/2)*(start_value+end_value) + sizes[0]*sizes[1]*(shifted_idx[1])*sizes[2];

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check2D_YZ_inc(ACC<double> &dat2D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[2]= {idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    int start_value= shifted_idx[1]*sizes[0]*sizes[1]+shifted_idx[0]*sizes[0];
    int end_value= start_value+(sizes[2]-1);
    expectedValue = (sizes[0]/2)*(start_value+end_value);

    if (dat2D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}


void check1D_X_max(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = sizes[1]*sizes[2]*(sizes[0]-1) + sizes[1]*(sizes[2]-1) + shifted_idx[0];

    
    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check1D_Y_max(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = sizes[1]*sizes[2]*(sizes[0]-1) + sizes[1]*(shifted_idx[1]) + sizes[2]-1;

    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}


void check1D_Z_max(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = sizes[0]*sizes[1]*(shifted_idx[2]+1)-1;

    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}



void check1D_X_min(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = shifted_idx[0];

    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check1D_Y_min(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = sizes[1]*shifted_idx[1];

    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}


void check1D_Z_min(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];
    expectedValue = sizes[0]*sizes[1]*shifted_idx[2];

    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}



void check1D_X_inc(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];

    double first= shifted_idx[0];
    double last= sizes[0]*(sizes[1]-1)*(sizes[2])+(sizes[1]-1)*(sizes[2])+shifted_idx[0];
    expectedValue = (first+last)*(sizes[1]*sizes[2])/2;

    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void check1D_Y_inc(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];

    //sum up all numbers where shifted_idx[1] is at the tens place
    double within_sum=(shifted_idx[1]*sizes[0]+shifted_idx[1]*sizes[0]+sizes[1]-1)*sizes[1]/2;
    double hundreds_sum=(0+(sizes[1]-1))*sizes[1]/2*sizes[0]*sizes[1]*sizes[2];
    expectedValue = within_sum*sizes[2]+hundreds_sum;
    
    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}


void check1D_Z_inc(ACC<double> &dat1D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 0;
    int shifted_idx[3] = {idx[0] + sizes[0]/3, idx[1] + sizes[1]/3, idx[2] + sizes[2]/3};
    shifted_idx[0] %= sizes[0];
    shifted_idx[1] %= sizes[1];
    shifted_idx[2] %= sizes[2];

    double first = sizes[1]*sizes[2]*shifted_idx[2];
    double last = first+sizes[0]*sizes[1]-1;
    expectedValue = (first+last)*sizes[0]*sizes[1]/2;
    
    if (dat1D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}




void set_valXY_idx(ACC<double> &dat, const double *val,  const int* sizes, const int *idx)
{
    dat(0,0,0) = (*val)*(idx[0]*sizes[1]+idx[1]);
}

void set_valYZ_idx(ACC<double> &dat, const double *val,  const int* sizes, const int *idx)
{
    dat(0,0,0) = (*val)*(idx[1]*sizes[2]+idx[2]);
}

void set_valXZ_idx(ACC<double> &dat, const double *val,  const int* sizes, const int *idx)
{
    dat(0,0,0) = (*val)*(idx[0]*sizes[2]+idx[2]);
}

void set_valX_idx(ACC<double> &dat, const double *val,  const int *idx)
{
    dat(0,0,0) = (*val)*idx[0];
}
void set_valY_idx(ACC<double> &dat, const double *val,  const int *idx)
{
    dat(0,0,0) = (*val)*idx[1];
}
void set_valZ_idx(ACC<double> &dat, const double *val,  const int *idx)
{
    dat(0,0,0) = (*val)*idx[2];
}


void calc(ACC<double> &dat3D, const ACC<double> &dat2D_xy,  const ACC<double> &dat2D_yz, const ACC<double> &dat2D_xz,
    const ACC<double> &dat1D_x,  const ACC<double> &dat1D_y, const ACC<double> &dat1D_z)
{
  dat3D(0,0,0) = dat2D_xy(0,0,0) +
                           dat2D_yz(0,0,0) +
                           dat2D_xz(0,0,0) +
                           dat1D_x(0,0,0) +
                           dat1D_y(0,0,0) +
                           dat1D_z(0,0,0);
}


void check_3D(ACC<double> &dat3D, const int* sizes, const int *idx, int *error_count)
{
    double expectedValue = 1.0*(idx[0]*sizes[1]+idx[1]) +
                           100.0*(idx[1]*sizes[2]+idx[2]) +
                           10000.0*(idx[0]*sizes[2]+idx[2]) +
                           0.01*idx[0] +
                           0.0001*idx[1] +
                           0.000001*idx[2];
    
    if (dat3D(0,0,0) != expectedValue) {
        *error_count += 1;
    }
}

void reduct22D_max(const ACC<double> &dat3D, ACC<double> &dat2D_xz, ACC<double> &dat2D_xy, ACC<double> &dat2D_yz)
{
    dat2D_xz.combine_max(0,0,0,dat3D(0,0,0));
    dat2D_xy.combine_max(0,0,0,dat3D(0,0,0));
    dat2D_yz.combine_max(0,0,0,dat3D(0,0,0));
}

void reduct22D_min(const ACC<double> &dat3D, ACC<double> &dat2D_xz, ACC<double> &dat2D_xy, ACC<double> &dat2D_yz)
{
    dat2D_xz.combine_min(0,0,0,dat3D(0,0,0));
    dat2D_xy.combine_min(0,0,0,dat3D(0,0,0));
    dat2D_yz.combine_min(0,0,0,dat3D(0,0,0));
}

void reduct22D_inc(const ACC<double> &dat3D, ACC<double> &dat2D_xz, ACC<double> &dat2D_xy, ACC<double> &dat2D_yz)
{
    dat2D_xz.combine_inc(0,0,0,dat3D(0,0,0));
    dat2D_xy.combine_inc(0,0,0,dat3D(0,0,0));
    dat2D_yz.combine_inc(0,0,0,dat3D(0,0,0));
}

void reduct21D_max(const ACC<double> &dat3D, ACC<double> &dat1D_x, ACC<double> &dat1D_y, ACC<double> &dat1D_z)
{
    dat1D_x.combine_max(0,0,0,dat3D(0,0,0));
    dat1D_y.combine_max(0,0,0,dat3D(0,0,0));
    dat1D_z.combine_max(0,0,0,dat3D(0,0,0));
}
void reduct21D_min(const ACC<double> &dat3D, ACC<double> &dat1D_x, ACC<double> &dat1D_y, ACC<double> &dat1D_z)
{
    dat1D_x.combine_min(0,0,0,dat3D(0,0,0));
    dat1D_y.combine_min(0,0,0,dat3D(0,0,0));
    dat1D_z.combine_min(0,0,0,dat3D(0,0,0));
}
void reduct21D_inc(const ACC<double> &dat3D, ACC<double> &dat1D_x, ACC<double> &dat1D_y, ACC<double> &dat1D_z)
{
    dat1D_x.combine_inc(0,0,0,dat3D(0,0,0));
    dat1D_y.combine_inc(0,0,0,dat3D(0,0,0));
    dat1D_z.combine_inc(0,0,0,dat3D(0,0,0));
}
#endif
