#ifndef ADVEC_CELL_KERNEL_H
#define ADVEC_CELL_KERNEL_H

void advec_cell_xdir_kernel1( double **pre_vol, double **post_vol, double **volume,
                        double **vol_flux_x, double **vol_flux_y) {

  **pre_vol = **volume + ( (*vol_flux_x[1]) - (*vol_flux_x[0]) +
                           (*vol_flux_y[1]) - (*vol_flux_y[0]));
  **post_vol = **pre_vol - ( (*vol_flux_x[1]) - (*vol_flux_x[0]));
}


void advec_cell_xdir_kernel2( double **pre_vol, double **post_vol, double **volume,
                        double **vol_flux_x) {

  **pre_vol = **volume + (*vol_flux_x[1]) - (*vol_flux_x[0]);
  **post_vol = **volume;

}

void advec_cell_xdir_kernel3( double **vol_flux_x, double **pre_vol, int** xx,
                              double **vertexdx,
                              double **density1, double **energy1 ,
                              double **mass_flux_x, double** ener_flux) {

  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int x_max=field->x_max;

  int upwind,donor,downwind,dif;

  //pre_vol accessed with: {0,0, 1,0, -1,0, -2,0};
  //vertexdx accessed with: {0,0, 1,0, -1,0, -2,0};
  //density1, energy1 accessed with: {0,0, 1,0, -1,0, -2,0};
  //xx accessed with: {0,0 ,1,0}

  if(**vol_flux_x > 0.0) {
    upwind   = 3; //j-2
    donor    = 2; //j-1
    downwind = 0; //j
    dif      = donor;
  }
  else if (*xx[1] < x_max+2) {
    upwind   = 1; //j+1
    donor    = 0; //j
    downwind = 2; //j-1
    dif      = upwind;
  } else /*{ //*xx[1] >= x_max+2 , then need 0
    upwind   = 0; //xmax+2
    donor    = 0; //j
    downwind = 2; //j-1
    dif      = upwind;
  }*/
  return;

  sigmat = fabs(**vol_flux_x)/(*pre_vol[donor]);
  sigma3 = (1.0 + sigmat)*((*vertexdx[0])/(*vertexdx[dif]));
  sigma4 = 2.0 - sigmat;

  sigma = sigmat;
  sigmav = sigmat;

  diffuw = (*density1[donor]) - (*density1[upwind]);
  diffdw = (*density1[downwind]) - (*density1[donor]);

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  **mass_flux_x = (**vol_flux_x) * ( (*density1[donor]) + limiter );

  sigmam = fabs(**mass_flux_x)/( (*density1[donor]) * (*pre_vol[donor]));
  diffuw = (*energy1[donor]) - (*energy1[upwind]);
  diffdw = (*energy1[downwind]) - (*energy1[donor]);

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  **ener_flux = (**mass_flux_x) * ( (*energy1[donor]) + limiter );
}



void advec_cell_xdir_kernel3a( double **vol_flux_x, double **pre_vol, int** xx,
                              double **vertexdx, double **vertexdx_xmax,
                              double **density1, double **density1_xmax,
                              double **energy1 , double **energy1_xmax,
                              double **mass_flux_x, double** ener_flux) {

  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int upwind,donor,downwind,dif;

  int x_max=field->x_max;

  //pre_vol accessed with: {0,0, 1,0, -1,0, -2,0};
  //vertexdx accessed with: {0,0, 1,0, -1,0, -2,0};
  //vertexdx_xmax accessed with: {xmax+2,0}; stride {0,0}

  //density1, energy1 accessed with: {0,0, 1,0, -1,0, -2,0};
  //density1_xmax, energy1_xmax accessed with: {xmax+2,0}; stride {0,1}
  //xx accessed with: {0,0 ,1,0}

  if(**vol_flux_x <= 0.0 && (*xx[1] >= x_max+2)) {
    upwind   = 0; //xmax+2
    donor    = 0; //j
    downwind = 2; //j-1
    dif      = 0; //xmax+2;

    sigmat = fabs(**vol_flux_x)/(*pre_vol[donor]);
    sigma3 = (1.0 + sigmat)*((*vertexdx[0])/(*vertexdx_xmax[dif]));
    sigma4 = 2.0 - sigmat;

    sigma = sigmat;
    sigmav = sigmat;

    diffuw = (*density1[donor]) - (*density1_xmax[upwind]);
    diffdw = (*density1[downwind]) - (*density1[donor]);

    if( (diffuw*diffdw) > 0.0)
      limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
      MIN( MIN(fabs(diffuw), fabs(diffdw)),
      one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
    else
      limiter=0.0;

    **mass_flux_x = (**vol_flux_x) * ( (*density1[donor]) + limiter );

    sigmam = fabs(**mass_flux_x)/( (*density1[donor]) * (*pre_vol[donor]));
    diffuw = (*energy1[donor]) - (*energy1_xmax[upwind]);
    diffdw = (*energy1[downwind]) - (*energy1[donor]);

    if( (diffuw*diffdw) > 0.0)
      limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
      MIN( MIN(fabs(diffuw), fabs(diffdw)),
      one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
    else
      limiter=0.0;

    **ener_flux = (**mass_flux_x) * ( (*energy1[donor]) + limiter );
  }
}

void advec_cell_xdir_kernel4( double **density1, double **energy1,
                         double** mass_flux_x, double** vol_flux_x,
                         double** pre_vol, double **post_vol,
                         double **pre_mass, double **post_mass,
                         double **advec_vol, double **post_ener,
                         double **ener_flux) {

  **pre_mass = (**density1) * (**pre_vol);
  **post_mass = (**pre_mass) + (*mass_flux_x[0]) - (*mass_flux_x[1]);
  **post_ener = ( (**energy1) * (**pre_mass) + (*ener_flux[0]) - (*ener_flux[1]))/ (**post_mass);
  **advec_vol = (**pre_vol) + (*vol_flux_x[0]) - (*vol_flux_x[1]);
  **density1 = (**post_mass)/(**advec_vol);
  **energy1 = **post_ener;
}



void advec_cell_ydir_kernel1( double **pre_vol, double **post_vol, double **volume,
                        double **vol_flux_x, double **vol_flux_y) {


  **pre_vol = (**volume) + ( (*vol_flux_y[1]) - (*vol_flux_y[0]) +
                             (*vol_flux_x[1]) - (*vol_flux_x[0]) );
  **post_vol = (**pre_vol) - ( (*vol_flux_y[1]) - (*vol_flux_y[0]) );
}

void advec_cell_ydir_kernel2( double **pre_vol, double **post_vol, double **volume,
                        double **vol_flux_y) {

  **pre_vol = **volume + (*vol_flux_y[1])- (*vol_flux_y[0]);
  **post_vol = **volume;

}

void advec_cell_ydir_kernel3( double **vol_flux_y, double **pre_vol, int** yy,
                              double **vertexdy,
                              double **density1, double **energy1 ,
                              double **mass_flux_y, double** ener_flux) {


  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int upwind = 99, donor, downwind, dif;

  //pre_vol accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //vertexdy accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //density1, energy1 accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //yy accessed with: {0,0 ,0,1}

  int y_max = field->y_max;

  if(**vol_flux_y > 0.0) {
    upwind   = 3; //k-2
    donor    = 2; //k-1
    downwind = 0; //k
    dif      = donor;
  }
  else if(*yy[1] < y_max+2) {
    upwind   = 1; //k+1
    donor    = 0; //k
    downwind = 2; //k-1
    dif      = upwind;
  } else /*{ //*yy[1] >= y_max+2 , then need 0
    upwind   = 0; //y_max
    donor    = 0; //k
    downwind = 2; //k-1
    dif      = upwind;
  }*/
  return;

  sigmat = fabs(**vol_flux_y)/(*pre_vol[donor]);
  sigma3 = (1.0 + sigmat) * ((*vertexdy[0])/(*vertexdy[dif]));
  sigma4 = 2.0 - sigmat;

  sigma = sigmat;
  sigmav = sigmat;

  diffuw = (*density1[donor]) - (*density1[upwind]);
  diffdw = (*density1[downwind]) - (*density1[donor]);

  if( (diffuw * diffdw) > 0.0 )
    limiter = (1.0 - sigmav) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw),fabs(diffdw)), one_by_six*(sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter = 0.0;

  **mass_flux_y = (**vol_flux_y) * (*density1[donor] + limiter);

  sigmam = fabs(**mass_flux_y)/((*density1[donor])* (*pre_vol[donor]));
  diffuw = (*energy1[donor]) - (*energy1[upwind]);
  diffdw = (*energy1[downwind]) - (*energy1[donor]);

  if( (diffuw * diffdw) > 0.0 )
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN(MIN(fabs(diffuw),fabs(diffdw)), one_by_six*(sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  **ener_flux = (**mass_flux_y) * ( *energy1[donor] + limiter);
}


void advec_cell_ydir_kernel3a( double **vol_flux_y, double **pre_vol, int** yy,
                              double **vertexdy, double **vertexdy_ymax,
                              double **density1, double **density1_ymax,
                              double **energy1 , double **energy1_ymax,
                              double **mass_flux_y, double** ener_flux) {

  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int upwind,donor,downwind,dif;

  int y_max=field->y_max;

  if(**vol_flux_y <= 0.0 && (*yy[1] >= y_max+2)) {

    upwind   = 0; //ymax+2
    donor    = 0; //k
    downwind = 2; //k-1
    dif      = 0; //ymax+2;

    sigmat = fabs(**vol_flux_y)/(*pre_vol[donor]);
    sigma3 = (1.0 + sigmat) * ((*vertexdy[0])/(*vertexdy_ymax[dif]));
    sigma4 = 2.0 - sigmat;

    sigma = sigmat;
    sigmav = sigmat;

    diffuw = (*density1[donor]) - (*density1_ymax[upwind]);
    diffdw = (*density1[downwind]) - (*density1[donor]);

    if( (diffuw * diffdw) > 0.0 )
      limiter = (1.0 - sigmav) * SIGN(1.0,diffdw) *
      MIN( MIN(fabs(diffuw),fabs(diffdw)), one_by_six*(sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
    else
      limiter = 0.0;

    **mass_flux_y = (**vol_flux_y) * (*density1[donor] + limiter);

    sigmam = fabs(**mass_flux_y)/((*density1[donor])* (*pre_vol[donor]));
    diffuw = (*energy1[donor]) - (*energy1_ymax[upwind]);
    diffdw = (*energy1[downwind]) - (*energy1[donor]);

    if( (diffuw * diffdw) > 0.0 )
      limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
      MIN(MIN(fabs(diffuw),fabs(diffdw)), one_by_six*(sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
    else
      limiter=0.0;

    **ener_flux = (**mass_flux_y) * ( *energy1[donor] + limiter);
  }

}

void advec_cell_ydir_kernel4( double **density1, double **energy1,
                         double** mass_flux_y, double** vol_flux_y,
                         double** pre_vol, double **post_vol,
                         double **pre_mass, double **post_mass,
                         double **advec_vol, double **post_ener,
                         double **ener_flux) {

  **pre_mass = (**density1) * (**pre_vol);
  **post_mass = (**pre_mass) + (*mass_flux_y[0]) - (*mass_flux_y[1]);
  **post_ener = ( (**energy1) * (**pre_mass) + (*ener_flux[0]) - (*ener_flux[1]))/ (**post_mass);
  **advec_vol = (**pre_vol) + (*vol_flux_y[0]) - (*vol_flux_y[1]);
  **density1 = (**post_mass)/(**advec_vol);
  **energy1 = **post_ener;
}

#endif
