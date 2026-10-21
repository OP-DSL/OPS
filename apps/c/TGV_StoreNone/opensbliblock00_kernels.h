#ifndef OPENSBLIBLOCK00_KERNEL_H
#define OPENSBLIBLOCK00_KERNEL_H
 void opensbliblock00Kernel032(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0,
const int *idx)
{
   double r = 0.0;
   double u2 = 0.0;
   double u0 = 0.0;
   double x1 = 0.0;
   double x2 = 0.0;
   double u1 = 0.0;
   double x0 = 0.0;
   double p = 0.0;
   x0 = Delta0block0*idx[0];

   x1 = Delta1block0*idx[1];

   x2 = Delta2block0*idx[2];

   u0 = sin(x0)*cos(x1)*cos(x2);

   u1 = -sin(x1)*cos(x0)*cos(x2);

   u2 = 0.0;

   p = (0.0625*cos(2.0*x0) + 0.0625*cos(2.0*x1))*(cos(2.0*x2) + 2.0) + 1.0/(pow(Minf, 2)*gama);

   r = pow(Minf, 2)*gama*p;

   rho_B0[OPS_ACC1(0,0,0)] = r;

   rhou0_B0[OPS_ACC2(0,0,0)] = r*u0;

   rhou1_B0[OPS_ACC4(0,0,0)] = r*u1;

   rhou2_B0[OPS_ACC3(0,0,0)] = r*u2;

   rhoE_B0[OPS_ACC0(0,0,0)] = p/(gama - 1) + 0.5*r*(pow(u0, 2) + pow(u1, 2) + pow(u2, 2));

}

void opensbliblock00Kernel026(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0)
{
   rho_B0[OPS_ACC1(-1,0,0)] = rho_B0[OPS_ACC1(1,0,0)];

   rhou0_B0[OPS_ACC2(-1,0,0)] = -1.0*rhou0_B0[OPS_ACC2(1,0,0)];

   rhou1_B0[OPS_ACC4(-1,0,0)] = rhou1_B0[OPS_ACC4(1,0,0)];

   rhou2_B0[OPS_ACC3(-1,0,0)] = rhou2_B0[OPS_ACC3(1,0,0)];

   rhoE_B0[OPS_ACC0(-1,0,0)] = rhoE_B0[OPS_ACC0(1,0,0)];

   rho_B0[OPS_ACC1(-2,0,0)] = rho_B0[OPS_ACC1(2,0,0)];

   rhou0_B0[OPS_ACC2(-2,0,0)] = -1.0*rhou0_B0[OPS_ACC2(2,0,0)];

   rhou1_B0[OPS_ACC4(-2,0,0)] = rhou1_B0[OPS_ACC4(2,0,0)];

   rhou2_B0[OPS_ACC3(-2,0,0)] = rhou2_B0[OPS_ACC3(2,0,0)];

   rhoE_B0[OPS_ACC0(-2,0,0)] = rhoE_B0[OPS_ACC0(2,0,0)];

}

void opensbliblock00Kernel027(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0)
{
   rho_B0[OPS_ACC1(1,0,0)] = rho_B0[OPS_ACC1(-1,0,0)];

   rhou0_B0[OPS_ACC2(1,0,0)] = -1.0*rhou0_B0[OPS_ACC2(-1,0,0)];

   rhou1_B0[OPS_ACC4(1,0,0)] = rhou1_B0[OPS_ACC4(-1,0,0)];

   rhou2_B0[OPS_ACC3(1,0,0)] = rhou2_B0[OPS_ACC3(-1,0,0)];

   rhoE_B0[OPS_ACC0(1,0,0)] = rhoE_B0[OPS_ACC0(-1,0,0)];

   rho_B0[OPS_ACC1(2,0,0)] = rho_B0[OPS_ACC1(-2,0,0)];

   rhou0_B0[OPS_ACC2(2,0,0)] = -1.0*rhou0_B0[OPS_ACC2(-2,0,0)];

   rhou1_B0[OPS_ACC4(2,0,0)] = rhou1_B0[OPS_ACC4(-2,0,0)];

   rhou2_B0[OPS_ACC3(2,0,0)] = rhou2_B0[OPS_ACC3(-2,0,0)];

   rhoE_B0[OPS_ACC0(2,0,0)] = rhoE_B0[OPS_ACC0(-2,0,0)];

}

void opensbliblock00Kernel028(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0)
{
   rho_B0[OPS_ACC1(0,-1,0)] = rho_B0[OPS_ACC1(0,1,0)];

   rhou0_B0[OPS_ACC2(0,-1,0)] = rhou0_B0[OPS_ACC2(0,1,0)];

   rhou1_B0[OPS_ACC4(0,-1,0)] = -1.0*rhou1_B0[OPS_ACC4(0,1,0)];

   rhou2_B0[OPS_ACC3(0,-1,0)] = rhou2_B0[OPS_ACC3(0,1,0)];

   rhoE_B0[OPS_ACC0(0,-1,0)] = rhoE_B0[OPS_ACC0(0,1,0)];

   rho_B0[OPS_ACC1(0,-2,0)] = rho_B0[OPS_ACC1(0,2,0)];

   rhou0_B0[OPS_ACC2(0,-2,0)] = rhou0_B0[OPS_ACC2(0,2,0)];

   rhou1_B0[OPS_ACC4(0,-2,0)] = -1.0*rhou1_B0[OPS_ACC4(0,2,0)];

   rhou2_B0[OPS_ACC3(0,-2,0)] = rhou2_B0[OPS_ACC3(0,2,0)];

   rhoE_B0[OPS_ACC0(0,-2,0)] = rhoE_B0[OPS_ACC0(0,2,0)];

}

void opensbliblock00Kernel029(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0)
{
   rho_B0[OPS_ACC1(0,1,0)] = rho_B0[OPS_ACC1(0,-1,0)];

   rhou0_B0[OPS_ACC2(0,1,0)] = rhou0_B0[OPS_ACC2(0,-1,0)];

   rhou1_B0[OPS_ACC4(0,1,0)] = -1.0*rhou1_B0[OPS_ACC4(0,-1,0)];

   rhou2_B0[OPS_ACC3(0,1,0)] = rhou2_B0[OPS_ACC3(0,-1,0)];

   rhoE_B0[OPS_ACC0(0,1,0)] = rhoE_B0[OPS_ACC0(0,-1,0)];

   rho_B0[OPS_ACC1(0,2,0)] = rho_B0[OPS_ACC1(0,-2,0)];

   rhou0_B0[OPS_ACC2(0,2,0)] = rhou0_B0[OPS_ACC2(0,-2,0)];

   rhou1_B0[OPS_ACC4(0,2,0)] = -1.0*rhou1_B0[OPS_ACC4(0,-2,0)];

   rhou2_B0[OPS_ACC3(0,2,0)] = rhou2_B0[OPS_ACC3(0,-2,0)];

   rhoE_B0[OPS_ACC0(0,2,0)] = rhoE_B0[OPS_ACC0(0,-2,0)];

}

void opensbliblock00Kernel030(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0)
{
   rho_B0[OPS_ACC1(0,0,-1)] = rho_B0[OPS_ACC1(0,0,1)];

   rhou0_B0[OPS_ACC2(0,0,-1)] = rhou0_B0[OPS_ACC2(0,0,1)];

   rhou1_B0[OPS_ACC4(0,0,-1)] = rhou1_B0[OPS_ACC4(0,0,1)];

   rhou2_B0[OPS_ACC3(0,0,-1)] = -1.0*rhou2_B0[OPS_ACC3(0,0,1)];

   rhoE_B0[OPS_ACC0(0,0,-1)] = rhoE_B0[OPS_ACC0(0,0,1)];

   rho_B0[OPS_ACC1(0,0,-2)] = rho_B0[OPS_ACC1(0,0,2)];

   rhou0_B0[OPS_ACC2(0,0,-2)] = rhou0_B0[OPS_ACC2(0,0,2)];

   rhou1_B0[OPS_ACC4(0,0,-2)] = rhou1_B0[OPS_ACC4(0,0,2)];

   rhou2_B0[OPS_ACC3(0,0,-2)] = -1.0*rhou2_B0[OPS_ACC3(0,0,2)];

   rhoE_B0[OPS_ACC0(0,0,-2)] = rhoE_B0[OPS_ACC0(0,0,2)];

}

void opensbliblock00Kernel031(double *rhoE_B0, double *rho_B0, double *rhou0_B0, double *rhou2_B0, double *rhou1_B0)
{
   rho_B0[OPS_ACC1(0,0,1)] = rho_B0[OPS_ACC1(0,0,-1)];

   rhou0_B0[OPS_ACC2(0,0,1)] = rhou0_B0[OPS_ACC2(0,0,-1)];

   rhou1_B0[OPS_ACC4(0,0,1)] = rhou1_B0[OPS_ACC4(0,0,-1)];

   rhou2_B0[OPS_ACC3(0,0,1)] = -1.0*rhou2_B0[OPS_ACC3(0,0,-1)];

   rhoE_B0[OPS_ACC0(0,0,1)] = rhoE_B0[OPS_ACC0(0,0,-1)];

   rho_B0[OPS_ACC1(0,0,2)] = rho_B0[OPS_ACC1(0,0,-2)];

   rhou0_B0[OPS_ACC2(0,0,2)] = rhou0_B0[OPS_ACC2(0,0,-2)];

   rhou1_B0[OPS_ACC4(0,0,2)] = rhou1_B0[OPS_ACC4(0,0,-2)];

   rhou2_B0[OPS_ACC3(0,0,2)] = -1.0*rhou2_B0[OPS_ACC3(0,0,-2)];

   rhoE_B0[OPS_ACC0(0,0,2)] = rhoE_B0[OPS_ACC0(0,0,-2)];

}

 void opensbliblock00Kernel033(const double *rhoE_B0, const double *rho_B0, const double *rhou0_B0, const double
*rhou2_B0, const double *rhou1_B0, double *rhoE_RKold_B0, double *rhou2_RKold_B0, double *rho_RKold_B0, double
*rhou1_RKold_B0, double *rhou0_RKold_B0)
{
   rho_RKold_B0[OPS_ACC7(0,0,0)] = rho_B0[OPS_ACC1(0,0,0)];

   rhou0_RKold_B0[OPS_ACC9(0,0,0)] = rhou0_B0[OPS_ACC2(0,0,0)];

   rhou1_RKold_B0[OPS_ACC8(0,0,0)] = rhou1_B0[OPS_ACC4(0,0,0)];

   rhou2_RKold_B0[OPS_ACC6(0,0,0)] = rhou2_B0[OPS_ACC3(0,0,0)];

   rhoE_RKold_B0[OPS_ACC5(0,0,0)] = rhoE_B0[OPS_ACC0(0,0,0)];

}

void opensbliblock00Kernel001(const double *rho_B0, const double *rhou0_B0, double *u0_B0)
{
   u0_B0[OPS_ACC2(0,0,0)] = rhou0_B0[OPS_ACC1(0,0,0)]/rho_B0[OPS_ACC0(0,0,0)];

}

void opensbliblock00Kernel005(const double *rho_B0, const double *rhou1_B0, double *u1_B0)
{
   u1_B0[OPS_ACC2(0,0,0)] = rhou1_B0[OPS_ACC1(0,0,0)]/rho_B0[OPS_ACC0(0,0,0)];

}

void opensbliblock00Kernel009(const double *rho_B0, const double *rhou2_B0, double *u2_B0)
{
   u2_B0[OPS_ACC2(0,0,0)] = rhou2_B0[OPS_ACC1(0,0,0)]/rho_B0[OPS_ACC0(0,0,0)];

}

 void opensbliblock00Kernel017(const double *rhoE_B0, const double *u2_B0, const double *u0_B0, const double *rho_B0,
const double *u1_B0, double *p_B0)
{
    p_B0[OPS_ACC5(0,0,0)] = (gama - 1)*(rhoE_B0[OPS_ACC0(0,0,0)] -
      rc6*rho_B0[OPS_ACC3(0,0,0)]*pow(u0_B0[OPS_ACC2(0,0,0)], 2) -
      rc6*rho_B0[OPS_ACC3(0,0,0)]*pow(u1_B0[OPS_ACC4(0,0,0)], 2) -
      rc6*rho_B0[OPS_ACC3(0,0,0)]*pow(u2_B0[OPS_ACC1(0,0,0)], 2));

}

void opensbliblock00Kernel018(const double *p_B0, const double *rho_B0, double *T_B0)
{
   T_B0[OPS_ACC2(0,0,0)] = pow(Minf, 2)*gama*p_B0[OPS_ACC0(0,0,0)]/rho_B0[OPS_ACC1(0,0,0)];

}

void opensbliblock00Kernel000(const double *u0_B0, double *wk0_B0)
{
    wk0_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_0*(u0_B0[OPS_ACC0(-2,0,0)] - 8*u0_B0[OPS_ACC0(-1,0,0)] +
      8*u0_B0[OPS_ACC0(1,0,0)] - u0_B0[OPS_ACC0(2,0,0)]);

}

void opensbliblock00Kernel002(const double *u0_B0, double *wk1_B0)
{
    wk1_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_1*(-8*u0_B0[OPS_ACC0(0,0,-1)] + u0_B0[OPS_ACC0(0,0,-2)] - u0_B0[OPS_ACC0(0,0,2)]
      + 8*u0_B0[OPS_ACC0(0,0,1)]);

}

void opensbliblock00Kernel003(const double *u0_B0, double *wk2_B0)
{
    wk2_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_2*(u0_B0[OPS_ACC0(0,-2,0)] - 8*u0_B0[OPS_ACC0(0,-1,0)] +
      8*u0_B0[OPS_ACC0(0,1,0)] - u0_B0[OPS_ACC0(0,2,0)]);

}

void opensbliblock00Kernel004(const double *u1_B0, double *wk3_B0)
{
    wk3_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_0*(-8*u1_B0[OPS_ACC0(-1,0,0)] + u1_B0[OPS_ACC0(-2,0,0)] +
      8*u1_B0[OPS_ACC0(1,0,0)] - u1_B0[OPS_ACC0(2,0,0)]);

}

void opensbliblock00Kernel006(const double *u1_B0, double *wk4_B0)
{
    wk4_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_1*(-8*u1_B0[OPS_ACC0(0,0,-1)] + u1_B0[OPS_ACC0(0,0,-2)] - u1_B0[OPS_ACC0(0,0,2)]
      + 8*u1_B0[OPS_ACC0(0,0,1)]);

}

void opensbliblock00Kernel007(const double *u1_B0, double *wk5_B0)
{
    wk5_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_2*(-8*u1_B0[OPS_ACC0(0,-1,0)] + u1_B0[OPS_ACC0(0,-2,0)] +
      8*u1_B0[OPS_ACC0(0,1,0)] - u1_B0[OPS_ACC0(0,2,0)]);

}

void opensbliblock00Kernel008(const double *u2_B0, double *wk6_B0)
{
    wk6_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_0*(u2_B0[OPS_ACC0(-2,0,0)] - u2_B0[OPS_ACC0(2,0,0)] + 8*u2_B0[OPS_ACC0(1,0,0)] -
      8*u2_B0[OPS_ACC0(-1,0,0)]);

}

void opensbliblock00Kernel010(const double *u2_B0, double *wk7_B0)
{
    wk7_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_1*(u2_B0[OPS_ACC0(0,0,-2)] + 8*u2_B0[OPS_ACC0(0,0,1)] - u2_B0[OPS_ACC0(0,0,2)] -
      8*u2_B0[OPS_ACC0(0,0,-1)]);

}

void opensbliblock00Kernel011(const double *u2_B0, double *wk8_B0)
{
    wk8_B0[OPS_ACC1(0,0,0)] = (rc7)*inv_2*(u2_B0[OPS_ACC0(0,-2,0)] - u2_B0[OPS_ACC0(0,2,0)] + 8*u2_B0[OPS_ACC0(0,1,0)] -
      8*u2_B0[OPS_ACC0(0,-1,0)]);

}

 void opensbliblock00Kernel024(const double *rhoE_B0, const double *u0_B0, const double *u2_B0, const double *u1_B0,
const double *wk7_B0, const double *p_B0, const double *rho_B0, const double *wk5_B0, const double *rhou0_B0, const
double *wk0_B0, const double *rhou2_B0, const double *rhou1_B0, double *Residual4_B0, double *Residual3_B0, double
*Residual1_B0, double *Residual0_B0, double *Residual2_B0)
{
   double localeval_35 = 0.0;
   double localeval_17 = 0.0;
   double localeval_23 = 0.0;
   double localeval_6 = 0.0;
   double localeval_15 = 0.0;
   double localeval_22 = 0.0;
   double localeval_10 = 0.0;
   double localeval_4 = 0.0;
   double localeval_31 = 0.0;
   double localeval_33 = 0.0;
   double localeval_26 = 0.0;
   double localeval_29 = 0.0;
   double localeval_8 = 0.0;
   double localeval_11 = 0.0;
   double localeval_12 = 0.0;
   double localeval_25 = 0.0;
   double localeval_13 = 0.0;
   double localeval_16 = 0.0;
   double localeval_5 = 0.0;
   double localeval_32 = 0.0;
   double localeval_18 = 0.0;
   double localeval_9 = 0.0;
   double localeval_7 = 0.0;
   double localeval_3 = 0.0;
   double localeval_2 = 0.0;
   double localeval_27 = 0.0;
   double localeval_14 = 0.0;
   double localeval_34 = 0.0;
   double localeval_0 = 0.0;
   double localeval_1 = 0.0;
   double localeval_20 = 0.0;
   double localeval_24 = 0.0;
   double localeval_28 = 0.0;
   double localeval_19 = 0.0;
   double localeval_21 = 0.0;
   double localeval_30 = 0.0;
    localeval_0 = (rc7)*inv_0*(-rhou2_B0[OPS_ACC10(2,0,0)]*u0_B0[OPS_ACC1(2,0,0)] +
      rhou2_B0[OPS_ACC10(-2,0,0)]*u0_B0[OPS_ACC1(-2,0,0)] - 8*rhou2_B0[OPS_ACC10(-1,0,0)]*u0_B0[OPS_ACC1(-1,0,0)] +
      8*rhou2_B0[OPS_ACC10(1,0,0)]*u0_B0[OPS_ACC1(1,0,0)]);

    localeval_1 = (rc7)*inv_0*(rhou1_B0[OPS_ACC11(-2,0,0)]*u0_B0[OPS_ACC1(-2,0,0)] -
      8*rhou1_B0[OPS_ACC11(-1,0,0)]*u0_B0[OPS_ACC1(-1,0,0)] - rhou1_B0[OPS_ACC11(2,0,0)]*u0_B0[OPS_ACC1(2,0,0)] +
      8*rhou1_B0[OPS_ACC11(1,0,0)]*u0_B0[OPS_ACC1(1,0,0)]);

    localeval_2 = (rc7)*inv_0*(8*rhou1_B0[OPS_ACC11(1,0,0)] + rhou1_B0[OPS_ACC11(-2,0,0)] -
      8*rhou1_B0[OPS_ACC11(-1,0,0)] - rhou1_B0[OPS_ACC11(2,0,0)]);

    localeval_3 = (rc7)*inv_0*(-rhoE_B0[OPS_ACC0(2,0,0)] + rhoE_B0[OPS_ACC0(-2,0,0)] - 8*rhoE_B0[OPS_ACC0(-1,0,0)] +
      8*rhoE_B0[OPS_ACC0(1,0,0)]);

    localeval_4 = (rc7)*inv_0*(rho_B0[OPS_ACC6(-2,0,0)]*u0_B0[OPS_ACC1(-2,0,0)] -
      8*rho_B0[OPS_ACC6(-1,0,0)]*u0_B0[OPS_ACC1(-1,0,0)] - rho_B0[OPS_ACC6(2,0,0)]*u0_B0[OPS_ACC1(2,0,0)] +
      8*rho_B0[OPS_ACC6(1,0,0)]*u0_B0[OPS_ACC1(1,0,0)]);

    localeval_5 = (rc7)*inv_0*(8*rhou0_B0[OPS_ACC8(1,0,0)] - 8*rhou0_B0[OPS_ACC8(-1,0,0)] + rhou0_B0[OPS_ACC8(-2,0,0)] -
      rhou0_B0[OPS_ACC8(2,0,0)]);

    localeval_6 = (rc7)*inv_0*(p_B0[OPS_ACC5(-2,0,0)]*u0_B0[OPS_ACC1(-2,0,0)] -
      8*p_B0[OPS_ACC5(-1,0,0)]*u0_B0[OPS_ACC1(-1,0,0)] + 8*p_B0[OPS_ACC5(1,0,0)]*u0_B0[OPS_ACC1(1,0,0)] -
      p_B0[OPS_ACC5(2,0,0)]*u0_B0[OPS_ACC1(2,0,0)]);

    localeval_7 = (rc7)*inv_0*(-8*rhou2_B0[OPS_ACC10(-1,0,0)] + rhou2_B0[OPS_ACC10(-2,0,0)] +
      8*rhou2_B0[OPS_ACC10(1,0,0)] - rhou2_B0[OPS_ACC10(2,0,0)]);

    localeval_8 = (rc7)*inv_0*(-rhoE_B0[OPS_ACC0(2,0,0)]*u0_B0[OPS_ACC1(2,0,0)] +
      rhoE_B0[OPS_ACC0(-2,0,0)]*u0_B0[OPS_ACC1(-2,0,0)] - 8*rhoE_B0[OPS_ACC0(-1,0,0)]*u0_B0[OPS_ACC1(-1,0,0)] +
      8*rhoE_B0[OPS_ACC0(1,0,0)]*u0_B0[OPS_ACC1(1,0,0)]);

    localeval_9 = (rc7)*inv_0*(-rho_B0[OPS_ACC6(2,0,0)] + rho_B0[OPS_ACC6(-2,0,0)] - 8*rho_B0[OPS_ACC6(-1,0,0)] +
      8*rho_B0[OPS_ACC6(1,0,0)]);

    localeval_10 = (rc7)*inv_0*(rhou0_B0[OPS_ACC8(-2,0,0)]*u0_B0[OPS_ACC1(-2,0,0)] -
      8*rhou0_B0[OPS_ACC8(-1,0,0)]*u0_B0[OPS_ACC1(-1,0,0)] + 8*rhou0_B0[OPS_ACC8(1,0,0)]*u0_B0[OPS_ACC1(1,0,0)] -
      rhou0_B0[OPS_ACC8(2,0,0)]*u0_B0[OPS_ACC1(2,0,0)]);

    localeval_11 = (rc7)*inv_0*(-p_B0[OPS_ACC5(2,0,0)] + 8*p_B0[OPS_ACC5(1,0,0)] - 8*p_B0[OPS_ACC5(-1,0,0)] +
      p_B0[OPS_ACC5(-2,0,0)]);

    localeval_12 = (rc7)*inv_2*(p_B0[OPS_ACC5(0,-2,0)]*u1_B0[OPS_ACC3(0,-2,0)] -
      8*p_B0[OPS_ACC5(0,-1,0)]*u1_B0[OPS_ACC3(0,-1,0)] + 8*p_B0[OPS_ACC5(0,1,0)]*u1_B0[OPS_ACC3(0,1,0)] -
      p_B0[OPS_ACC5(0,2,0)]*u1_B0[OPS_ACC3(0,2,0)]);

    localeval_13 = (rc7)*inv_2*(rho_B0[OPS_ACC6(0,-2,0)]*u1_B0[OPS_ACC3(0,-2,0)] -
      8*rho_B0[OPS_ACC6(0,-1,0)]*u1_B0[OPS_ACC3(0,-1,0)] - rho_B0[OPS_ACC6(0,2,0)]*u1_B0[OPS_ACC3(0,2,0)] +
      8*rho_B0[OPS_ACC6(0,1,0)]*u1_B0[OPS_ACC3(0,1,0)]);

    localeval_14 = (rc7)*inv_2*(-rhou2_B0[OPS_ACC10(0,2,0)]*u1_B0[OPS_ACC3(0,2,0)] +
      rhou2_B0[OPS_ACC10(0,-2,0)]*u1_B0[OPS_ACC3(0,-2,0)] - 8*rhou2_B0[OPS_ACC10(0,-1,0)]*u1_B0[OPS_ACC3(0,-1,0)] +
      8*rhou2_B0[OPS_ACC10(0,1,0)]*u1_B0[OPS_ACC3(0,1,0)]);

    localeval_15 = (rc7)*inv_2*(-rhoE_B0[OPS_ACC0(0,2,0)]*u1_B0[OPS_ACC3(0,2,0)] +
      rhoE_B0[OPS_ACC0(0,-2,0)]*u1_B0[OPS_ACC3(0,-2,0)] - 8*rhoE_B0[OPS_ACC0(0,-1,0)]*u1_B0[OPS_ACC3(0,-1,0)] +
      8*rhoE_B0[OPS_ACC0(0,1,0)]*u1_B0[OPS_ACC3(0,1,0)]);

    localeval_16 = (rc7)*inv_2*(rhou1_B0[OPS_ACC11(0,-2,0)]*u1_B0[OPS_ACC3(0,-2,0)] -
      8*rhou1_B0[OPS_ACC11(0,-1,0)]*u1_B0[OPS_ACC3(0,-1,0)] - rhou1_B0[OPS_ACC11(0,2,0)]*u1_B0[OPS_ACC3(0,2,0)] +
      8*rhou1_B0[OPS_ACC11(0,1,0)]*u1_B0[OPS_ACC3(0,1,0)]);

    localeval_17 = (rc7)*inv_2*(-8*rhou2_B0[OPS_ACC10(0,-1,0)] + rhou2_B0[OPS_ACC10(0,-2,0)] +
      8*rhou2_B0[OPS_ACC10(0,1,0)] - rhou2_B0[OPS_ACC10(0,2,0)]);

    localeval_18 = (rc7)*inv_2*(8*rhou1_B0[OPS_ACC11(0,1,0)] + rhou1_B0[OPS_ACC11(0,-2,0)] -
      8*rhou1_B0[OPS_ACC11(0,-1,0)] - rhou1_B0[OPS_ACC11(0,2,0)]);

    localeval_19 = (rc7)*inv_2*(-rho_B0[OPS_ACC6(0,2,0)] + rho_B0[OPS_ACC6(0,-2,0)] - 8*rho_B0[OPS_ACC6(0,-1,0)] +
      8*rho_B0[OPS_ACC6(0,1,0)]);

    localeval_20 = (rc7)*inv_2*(-rhoE_B0[OPS_ACC0(0,2,0)] + rhoE_B0[OPS_ACC0(0,-2,0)] - 8*rhoE_B0[OPS_ACC0(0,-1,0)] +
      8*rhoE_B0[OPS_ACC0(0,1,0)]);

    localeval_21 = (rc7)*inv_2*(-p_B0[OPS_ACC5(0,2,0)] + 8*p_B0[OPS_ACC5(0,1,0)] - 8*p_B0[OPS_ACC5(0,-1,0)] +
      p_B0[OPS_ACC5(0,-2,0)]);

    localeval_22 = (rc7)*inv_2*(8*rhou0_B0[OPS_ACC8(0,1,0)] - 8*rhou0_B0[OPS_ACC8(0,-1,0)] + rhou0_B0[OPS_ACC8(0,-2,0)]
      - rhou0_B0[OPS_ACC8(0,2,0)]);

    localeval_23 = (rc7)*inv_2*(rhou0_B0[OPS_ACC8(0,-2,0)]*u1_B0[OPS_ACC3(0,-2,0)] -
      8*rhou0_B0[OPS_ACC8(0,-1,0)]*u1_B0[OPS_ACC3(0,-1,0)] + 8*rhou0_B0[OPS_ACC8(0,1,0)]*u1_B0[OPS_ACC3(0,1,0)] -
      rhou0_B0[OPS_ACC8(0,2,0)]*u1_B0[OPS_ACC3(0,2,0)]);

    localeval_24 = (rc7)*inv_1*(rhou0_B0[OPS_ACC8(0,0,-2)]*u2_B0[OPS_ACC2(0,0,-2)] -
      8*rhou0_B0[OPS_ACC8(0,0,-1)]*u2_B0[OPS_ACC2(0,0,-1)] + 8*rhou0_B0[OPS_ACC8(0,0,1)]*u2_B0[OPS_ACC2(0,0,1)] -
      rhou0_B0[OPS_ACC8(0,0,2)]*u2_B0[OPS_ACC2(0,0,2)]);

    localeval_25 = (rc7)*inv_1*(-p_B0[OPS_ACC5(0,0,2)]*u2_B0[OPS_ACC2(0,0,2)] +
      p_B0[OPS_ACC5(0,0,-2)]*u2_B0[OPS_ACC2(0,0,-2)] - 8*p_B0[OPS_ACC5(0,0,-1)]*u2_B0[OPS_ACC2(0,0,-1)] +
      8*p_B0[OPS_ACC5(0,0,1)]*u2_B0[OPS_ACC2(0,0,1)]);

    localeval_26 = (rc7)*inv_1*(-8*rhou2_B0[OPS_ACC10(0,0,-1)] + rhou2_B0[OPS_ACC10(0,0,-2)] -
      rhou2_B0[OPS_ACC10(0,0,2)] + 8*rhou2_B0[OPS_ACC10(0,0,1)]);

    localeval_27 = (rc7)*inv_1*(-rho_B0[OPS_ACC6(0,0,2)] + rho_B0[OPS_ACC6(0,0,-2)] + 8*rho_B0[OPS_ACC6(0,0,1)] -
      8*rho_B0[OPS_ACC6(0,0,-1)]);

    localeval_28 = (rc7)*inv_1*(8*p_B0[OPS_ACC5(0,0,1)] - p_B0[OPS_ACC5(0,0,2)] - 8*p_B0[OPS_ACC5(0,0,-1)] +
      p_B0[OPS_ACC5(0,0,-2)]);

    localeval_29 = (rc7)*inv_1*(-rhoE_B0[OPS_ACC0(0,0,2)]*u2_B0[OPS_ACC2(0,0,2)] +
      8*rhoE_B0[OPS_ACC0(0,0,1)]*u2_B0[OPS_ACC2(0,0,1)] + rhoE_B0[OPS_ACC0(0,0,-2)]*u2_B0[OPS_ACC2(0,0,-2)] -
      8*rhoE_B0[OPS_ACC0(0,0,-1)]*u2_B0[OPS_ACC2(0,0,-1)]);

    localeval_30 = (rc7)*inv_1*(-rho_B0[OPS_ACC6(0,0,2)]*u2_B0[OPS_ACC2(0,0,2)] +
      rho_B0[OPS_ACC6(0,0,-2)]*u2_B0[OPS_ACC2(0,0,-2)] - 8*rho_B0[OPS_ACC6(0,0,-1)]*u2_B0[OPS_ACC2(0,0,-1)] +
      8*rho_B0[OPS_ACC6(0,0,1)]*u2_B0[OPS_ACC2(0,0,1)]);

    localeval_31 = (rc7)*inv_1*(rhou1_B0[OPS_ACC11(0,0,-2)]*u2_B0[OPS_ACC2(0,0,-2)] -
      8*rhou1_B0[OPS_ACC11(0,0,-1)]*u2_B0[OPS_ACC2(0,0,-1)] - rhou1_B0[OPS_ACC11(0,0,2)]*u2_B0[OPS_ACC2(0,0,2)] +
      8*rhou1_B0[OPS_ACC11(0,0,1)]*u2_B0[OPS_ACC2(0,0,1)]);

    localeval_32 = (rc7)*inv_1*(-8*rhou1_B0[OPS_ACC11(0,0,-1)] + 8*rhou1_B0[OPS_ACC11(0,0,1)] +
      rhou1_B0[OPS_ACC11(0,0,-2)] - rhou1_B0[OPS_ACC11(0,0,2)]);

    localeval_33 = (rc7)*inv_1*(-rhoE_B0[OPS_ACC0(0,0,2)] + 8*rhoE_B0[OPS_ACC0(0,0,1)] - 8*rhoE_B0[OPS_ACC0(0,0,-1)] +
      rhoE_B0[OPS_ACC0(0,0,-2)]);

    localeval_34 = (rc7)*inv_1*(rhou2_B0[OPS_ACC10(0,0,-2)]*u2_B0[OPS_ACC2(0,0,-2)] -
      8*rhou2_B0[OPS_ACC10(0,0,-1)]*u2_B0[OPS_ACC2(0,0,-1)] + 8*rhou2_B0[OPS_ACC10(0,0,1)]*u2_B0[OPS_ACC2(0,0,1)] -
      rhou2_B0[OPS_ACC10(0,0,2)]*u2_B0[OPS_ACC2(0,0,2)]);

    localeval_35 = (rc7)*inv_1*(-8*rhou0_B0[OPS_ACC8(0,0,-1)] + rhou0_B0[OPS_ACC8(0,0,-2)] + 8*rhou0_B0[OPS_ACC8(0,0,1)]
      - rhou0_B0[OPS_ACC8(0,0,2)]);

    Residual0_B0[OPS_ACC15(0,0,0)] = -rc6*localeval_13 - rc6*localeval_19*u1_B0[OPS_ACC3(0,0,0)] -
      rc6*localeval_27*u2_B0[OPS_ACC2(0,0,0)] - rc6*localeval_30 - rc6*localeval_4 -
      rc6*localeval_9*u0_B0[OPS_ACC1(0,0,0)] - rc6*(wk0_B0[OPS_ACC9(0,0,0)] + wk5_B0[OPS_ACC7(0,0,0)] +
      wk7_B0[OPS_ACC4(0,0,0)])*rho_B0[OPS_ACC6(0,0,0)];

    Residual1_B0[OPS_ACC14(0,0,0)] = -rc6*localeval_10 - localeval_11 - rc6*localeval_22*u1_B0[OPS_ACC3(0,0,0)] -
      rc6*localeval_23 - rc6*localeval_24 - rc6*localeval_35*u2_B0[OPS_ACC2(0,0,0)] -
      rc6*localeval_5*u0_B0[OPS_ACC1(0,0,0)] - rc6*(wk0_B0[OPS_ACC9(0,0,0)] + wk5_B0[OPS_ACC7(0,0,0)] +
      wk7_B0[OPS_ACC4(0,0,0)])*rhou0_B0[OPS_ACC8(0,0,0)];

    Residual2_B0[OPS_ACC16(0,0,0)] = -rc6*localeval_1 - rc6*localeval_16 - rc6*localeval_18*u1_B0[OPS_ACC3(0,0,0)] -
      rc6*localeval_2*u0_B0[OPS_ACC1(0,0,0)] - localeval_21 - rc6*localeval_31 - rc6*localeval_32*u2_B0[OPS_ACC2(0,0,0)]
      - rc6*(wk0_B0[OPS_ACC9(0,0,0)] + wk5_B0[OPS_ACC7(0,0,0)] + wk7_B0[OPS_ACC4(0,0,0)])*rhou1_B0[OPS_ACC11(0,0,0)];

    Residual3_B0[OPS_ACC13(0,0,0)] = -rc6*localeval_0 - rc6*localeval_14 - rc6*localeval_17*u1_B0[OPS_ACC3(0,0,0)] -
      rc6*localeval_26*u2_B0[OPS_ACC2(0,0,0)] - localeval_28 - rc6*localeval_34 - rc6*localeval_7*u0_B0[OPS_ACC1(0,0,0)]
      - rc6*(wk0_B0[OPS_ACC9(0,0,0)] + wk5_B0[OPS_ACC7(0,0,0)] + wk7_B0[OPS_ACC4(0,0,0)])*rhou2_B0[OPS_ACC10(0,0,0)];

    Residual4_B0[OPS_ACC12(0,0,0)] = -localeval_12 - rc6*localeval_15 - rc6*localeval_20*u1_B0[OPS_ACC3(0,0,0)] -
      localeval_25 - rc6*localeval_29 - rc6*localeval_3*u0_B0[OPS_ACC1(0,0,0)] - rc6*localeval_33*u2_B0[OPS_ACC2(0,0,0)]
      - localeval_6 - rc6*localeval_8 - rc6*(wk0_B0[OPS_ACC9(0,0,0)] + wk5_B0[OPS_ACC7(0,0,0)] +
      wk7_B0[OPS_ACC4(0,0,0)])*rhoE_B0[OPS_ACC0(0,0,0)];

}

 void opensbliblock00Kernel025(const double *wk8_B0, const double *wk3_B0, const double *wk2_B0, const double *u0_B0,
const double *wk1_B0, const double *u1_B0, const double *wk4_B0, const double *wk7_B0, const double *T_B0, const double
*u2_B0, const double *wk5_B0, const double *wk0_B0, const double *wk6_B0, double *Residual4_B0, double *Residual1_B0,
double *Residual3_B0, double *Residual2_B0)
{
   double localeval_16 = 0.0;
   double localeval_5 = 0.0;
   double localeval_9 = 0.0;
   double localeval_7 = 0.0;
   double localeval_6 = 0.0;
   double localeval_15 = 0.0;
   double localeval_3 = 0.0;
   double localeval_2 = 0.0;
   double localeval_14 = 0.0;
   double localeval_10 = 0.0;
   double localeval_0 = 0.0;
   double localeval_1 = 0.0;
   double localeval_4 = 0.0;
   double localeval_8 = 0.0;
   double localeval_11 = 0.0;
   double localeval_12 = 0.0;
   double localeval_13 = 0.0;
   double localeval_17 = 0.0;
    localeval_0 = (rc7)*inv_3*(16*T_B0[OPS_ACC8(1,0,0)] - 30*T_B0[OPS_ACC8(0,0,0)] - T_B0[OPS_ACC8(2,0,0)] -
      T_B0[OPS_ACC8(-2,0,0)] + 16*T_B0[OPS_ACC8(-1,0,0)]);

    localeval_1 = (rc7)*inv_4*(16*T_B0[OPS_ACC8(0,1,0)] - 30*T_B0[OPS_ACC8(0,0,0)] - T_B0[OPS_ACC8(0,-2,0)] -
      T_B0[OPS_ACC8(0,2,0)] + 16*T_B0[OPS_ACC8(0,-1,0)]);

    localeval_2 = (rc7)*inv_5*(-T_B0[OPS_ACC8(0,0,-2)] + 16*T_B0[OPS_ACC8(0,0,-1)] - 30*T_B0[OPS_ACC8(0,0,0)] -
      T_B0[OPS_ACC8(0,0,2)] + 16*T_B0[OPS_ACC8(0,0,1)]);

    localeval_3 = (rc7)*inv_4*(-u0_B0[OPS_ACC3(0,2,0)] + 16*u0_B0[OPS_ACC3(0,1,0)] - 30*u0_B0[OPS_ACC3(0,0,0)] +
      16*u0_B0[OPS_ACC3(0,-1,0)] - u0_B0[OPS_ACC3(0,-2,0)]);

    localeval_4 = (rc7)*inv_5*(16*u0_B0[OPS_ACC3(0,0,1)] - 30*u0_B0[OPS_ACC3(0,0,0)] + 16*u0_B0[OPS_ACC3(0,0,-1)] -
      u0_B0[OPS_ACC3(0,0,-2)] - u0_B0[OPS_ACC3(0,0,2)]);

    localeval_5 = (rc7)*inv_3*(-u0_B0[OPS_ACC3(2,0,0)] + 16*u0_B0[OPS_ACC3(1,0,0)] - 30*u0_B0[OPS_ACC3(0,0,0)] +
      16*u0_B0[OPS_ACC3(-1,0,0)] - u0_B0[OPS_ACC3(-2,0,0)]);

    localeval_6 = (rc7)*inv_4*(16*u1_B0[OPS_ACC5(0,1,0)] - u1_B0[OPS_ACC5(0,2,0)] - u1_B0[OPS_ACC5(0,-2,0)] +
      16*u1_B0[OPS_ACC5(0,-1,0)] - 30*u1_B0[OPS_ACC5(0,0,0)]);

    localeval_7 = (rc7)*inv_3*(16*u1_B0[OPS_ACC5(1,0,0)] - u1_B0[OPS_ACC5(2,0,0)] - u1_B0[OPS_ACC5(-2,0,0)] +
      16*u1_B0[OPS_ACC5(-1,0,0)] - 30*u1_B0[OPS_ACC5(0,0,0)]);

    localeval_8 = (rc7)*inv_5*(16*u1_B0[OPS_ACC5(0,0,1)] - u1_B0[OPS_ACC5(0,0,2)] - u1_B0[OPS_ACC5(0,0,-2)] +
      16*u1_B0[OPS_ACC5(0,0,-1)] - 30*u1_B0[OPS_ACC5(0,0,0)]);

    localeval_9 = (rc7)*inv_5*(-u2_B0[OPS_ACC9(0,0,2)] + 16*u2_B0[OPS_ACC9(0,0,-1)] - u2_B0[OPS_ACC9(0,0,-2)] -
      30*u2_B0[OPS_ACC9(0,0,0)] + 16*u2_B0[OPS_ACC9(0,0,1)]);

    localeval_10 = (rc7)*inv_3*(16*u2_B0[OPS_ACC9(-1,0,0)] - u2_B0[OPS_ACC9(-2,0,0)] - 30*u2_B0[OPS_ACC9(0,0,0)] +
      16*u2_B0[OPS_ACC9(1,0,0)] - u2_B0[OPS_ACC9(2,0,0)]);

    localeval_11 = (rc7)*inv_4*(16*u2_B0[OPS_ACC9(0,-1,0)] - u2_B0[OPS_ACC9(0,-2,0)] - 30*u2_B0[OPS_ACC9(0,0,0)] +
      16*u2_B0[OPS_ACC9(0,1,0)] - u2_B0[OPS_ACC9(0,2,0)]);

    localeval_12 = (rc7)*inv_2*(-wk0_B0[OPS_ACC11(0,2,0)] + wk0_B0[OPS_ACC11(0,-2,0)] - 8*wk0_B0[OPS_ACC11(0,-1,0)] +
      8*wk0_B0[OPS_ACC11(0,1,0)]);

    localeval_13 = (rc7)*inv_1*(wk0_B0[OPS_ACC11(0,0,-2)] - wk0_B0[OPS_ACC11(0,0,2)] + 8*wk0_B0[OPS_ACC11(0,0,1)] -
      8*wk0_B0[OPS_ACC11(0,0,-1)]);

    localeval_14 = (rc7)*inv_2*(-wk3_B0[OPS_ACC1(0,2,0)] + wk3_B0[OPS_ACC1(0,-2,0)] + 8*wk3_B0[OPS_ACC1(0,1,0)] -
      8*wk3_B0[OPS_ACC1(0,-1,0)]);

    localeval_15 = (rc7)*inv_1*(8*wk5_B0[OPS_ACC10(0,0,1)] - wk5_B0[OPS_ACC10(0,0,2)] + wk5_B0[OPS_ACC10(0,0,-2)] -
      8*wk5_B0[OPS_ACC10(0,0,-1)]);

    localeval_16 = (rc7)*inv_1*(-8*wk6_B0[OPS_ACC12(0,0,-1)] - wk6_B0[OPS_ACC12(0,0,2)] + wk6_B0[OPS_ACC12(0,0,-2)] +
      8*wk6_B0[OPS_ACC12(0,0,1)]);

    localeval_17 = (rc7)*inv_1*(-wk8_B0[OPS_ACC0(0,0,2)] + 8*wk8_B0[OPS_ACC0(0,0,1)] - 8*wk8_B0[OPS_ACC0(0,0,-1)] +
      wk8_B0[OPS_ACC0(0,0,-2)]);

    Residual1_B0[OPS_ACC14(0,0,0)] = 1.0*rcinv8*((rc9)*localeval_14 + (rc9)*localeval_16 + localeval_3 + localeval_4 +
      (rc10)*localeval_5) + Residual1_B0[OPS_ACC14(0,0,0)];

    Residual2_B0[OPS_ACC16(0,0,0)] = 1.0*rcinv8*((rc9)*localeval_12 + (rc9)*localeval_17 + (rc10)*localeval_6 +
      localeval_7 + localeval_8) + Residual2_B0[OPS_ACC16(0,0,0)];

    Residual3_B0[OPS_ACC15(0,0,0)] = 1.0*rcinv8*(localeval_10 + localeval_11 + (rc9)*localeval_13 + (rc9)*localeval_15 +
      (rc10)*localeval_9) + Residual3_B0[OPS_ACC15(0,0,0)];

    Residual4_B0[OPS_ACC13(0,0,0)] = 1.0*rcinv11*rcinv12*rcinv13*rcinv8*(localeval_0 + localeval_1 + localeval_2) +
      1.0*rcinv8*(wk1_B0[OPS_ACC4(0,0,0)] + wk6_B0[OPS_ACC12(0,0,0)])*wk1_B0[OPS_ACC4(0,0,0)] +
      1.0*rcinv8*(wk1_B0[OPS_ACC4(0,0,0)] + wk6_B0[OPS_ACC12(0,0,0)])*wk6_B0[OPS_ACC12(0,0,0)] +
      1.0*rcinv8*(wk2_B0[OPS_ACC2(0,0,0)] + wk3_B0[OPS_ACC1(0,0,0)])*wk2_B0[OPS_ACC2(0,0,0)] +
      1.0*rcinv8*(wk2_B0[OPS_ACC2(0,0,0)] + wk3_B0[OPS_ACC1(0,0,0)])*wk3_B0[OPS_ACC1(0,0,0)] +
      1.0*rcinv8*(wk4_B0[OPS_ACC6(0,0,0)] + wk8_B0[OPS_ACC0(0,0,0)])*wk4_B0[OPS_ACC6(0,0,0)] +
      1.0*rcinv8*(wk4_B0[OPS_ACC6(0,0,0)] + wk8_B0[OPS_ACC0(0,0,0)])*wk8_B0[OPS_ACC0(0,0,0)] +
      1.0*rcinv8*(-rc14*wk0_B0[OPS_ACC11(0,0,0)] - rc14*wk5_B0[OPS_ACC10(0,0,0)] +
      (rc10)*wk7_B0[OPS_ACC7(0,0,0)])*wk7_B0[OPS_ACC7(0,0,0)] + 1.0*rcinv8*(-rc14*wk0_B0[OPS_ACC11(0,0,0)] +
      (rc10)*wk5_B0[OPS_ACC10(0,0,0)] - rc14*wk7_B0[OPS_ACC7(0,0,0)])*wk5_B0[OPS_ACC10(0,0,0)] +
      1.0*rcinv8*((rc10)*wk0_B0[OPS_ACC11(0,0,0)] - rc14*wk5_B0[OPS_ACC10(0,0,0)] -
      rc14*wk7_B0[OPS_ACC7(0,0,0)])*wk0_B0[OPS_ACC11(0,0,0)] + 1.0*rcinv8*(localeval_10 + localeval_11 +
      (rc9)*localeval_13 + (rc9)*localeval_15 + (rc10)*localeval_9)*u2_B0[OPS_ACC9(0,0,0)] +
      1.0*rcinv8*((rc9)*localeval_12 + (rc9)*localeval_17 + (rc10)*localeval_6 + localeval_7 +
      localeval_8)*u1_B0[OPS_ACC5(0,0,0)] + 1.0*rcinv8*((rc9)*localeval_14 + (rc9)*localeval_16 + localeval_3 +
      localeval_4 + (rc10)*localeval_5)*u0_B0[OPS_ACC3(0,0,0)] + Residual4_B0[OPS_ACC13(0,0,0)];

}

 void opensbliblock00Kernel035(const double *rhoE_RKold_B0, const double *Residual3_B0, const double *rho_RKold_B0,
const double *Residual1_B0, const double *rhou0_RKold_B0, const double *Residual0_B0, const double *Residual4_B0, const
double *rhou2_RKold_B0, const double *rhou1_RKold_B0, const double *Residual2_B0, double *rhoE_B0, double *rho_B0,
double *rhou0_B0, double *rhou2_B0, double *rhou1_B0, const double *rknew)
{
   rho_B0[OPS_ACC11(0,0,0)] = dt*rknew[0]*Residual0_B0[OPS_ACC5(0,0,0)] + rho_RKold_B0[OPS_ACC2(0,0,0)];

   rhou0_B0[OPS_ACC12(0,0,0)] = dt*rknew[0]*Residual1_B0[OPS_ACC3(0,0,0)] + rhou0_RKold_B0[OPS_ACC4(0,0,0)];

   rhou1_B0[OPS_ACC14(0,0,0)] = dt*rknew[0]*Residual2_B0[OPS_ACC9(0,0,0)] + rhou1_RKold_B0[OPS_ACC8(0,0,0)];

   rhou2_B0[OPS_ACC13(0,0,0)] = dt*rknew[0]*Residual3_B0[OPS_ACC1(0,0,0)] + rhou2_RKold_B0[OPS_ACC7(0,0,0)];

   rhoE_B0[OPS_ACC10(0,0,0)] = dt*rknew[0]*Residual4_B0[OPS_ACC6(0,0,0)] + rhoE_RKold_B0[OPS_ACC0(0,0,0)];

}

 void opensbliblock00Kernel034(const double *Residual3_B0, const double *Residual1_B0, const double *Residual0_B0, const
double *Residual4_B0, const double *Residual2_B0, double *rhoE_RKold_B0, double *rho_RKold_B0, double *rhou0_RKold_B0,
double *rhou2_RKold_B0, double *rhou1_RKold_B0, const double *rkold)
{
   rho_RKold_B0[OPS_ACC6(0,0,0)] = dt*rkold[0]*Residual0_B0[OPS_ACC2(0,0,0)] + rho_RKold_B0[OPS_ACC6(0,0,0)];

   rhou0_RKold_B0[OPS_ACC7(0,0,0)] = dt*rkold[0]*Residual1_B0[OPS_ACC1(0,0,0)] + rhou0_RKold_B0[OPS_ACC7(0,0,0)];

   rhou1_RKold_B0[OPS_ACC9(0,0,0)] = dt*rkold[0]*Residual2_B0[OPS_ACC4(0,0,0)] + rhou1_RKold_B0[OPS_ACC9(0,0,0)];

   rhou2_RKold_B0[OPS_ACC8(0,0,0)] = dt*rkold[0]*Residual3_B0[OPS_ACC0(0,0,0)] + rhou2_RKold_B0[OPS_ACC8(0,0,0)];

   rhoE_RKold_B0[OPS_ACC5(0,0,0)] = dt*rkold[0]*Residual4_B0[OPS_ACC3(0,0,0)] + rhoE_RKold_B0[OPS_ACC5(0,0,0)];

}

#endif
