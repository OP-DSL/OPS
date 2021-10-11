# Developing an OPS Application

## Tutorial
##
## OPS User Kernels

In OPS, the elemental operation carried out per mesh/grid point is
specified as an outlined function called a *user kernel*. An example
taken from the Cloverleaf application is given below.

```c++
void accelerate_kernel( const ACC<double> &density0, const ACC<double> &volume,
                ACC<double> &stepbymass, const ACC<double> &xvel0, ACC<double> &xvel1,
                const ACC<double> &xarea, const ACC<double> &pressure,
                const ACC<double> &yvel0, ACC<double> &yvel1,
                const ACC<double> &yarea, const ACC<double> &viscosity) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( density0(-1,-1) * volume(-1,-1)
    + density0(0,-1) * volume(0,-1)
    + density0(0,0) * volume(0,0)
    + density0(-1,0) * volume(-1,0) ) * 0.25;

  stepbymass(0,0) = 0.5*dt/ nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1(0,0) = xvel0(0,0) - stepbymass(0,0) *
            ( xarea(0,0)  * ( pressure(0,0) - pressure(-1,0) ) +
              xarea(0,-1) * ( pressure(0,-1) - pressure(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1(0,0) = yvel0(0,0) - stepbymass(0,0) *
            ( yarea(0,0)  * ( pressure(0,0) - pressure(0,-1) ) +
              yarea(-1,0) * ( pressure(-1,0) - pressure(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1(0,0) = xvel1(0,0) - stepbymass(0,0) *
            ( xarea(0,0) * ( viscosity(0,0) - viscosity(-1,0) ) +
              xarea(0,-1) * ( viscosity(0,-1) - viscosity(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1(0,0) = yvel1(0,0) - stepbymass(0,0) *
            ( yarea(0,0) * ( viscosity(0,0) - viscosity(0,-1) ) +
              yarea(-1,0) * ( viscosity(-1,0) - viscosity(-1,-1) ) );

}
```

This user kernel is then used in an `ops_par_loop` function. The key aspect to note in the user kernel is the use of the ACC\<\> objects and their
parentheses operator. These specify the stencil in accessing the
elements of the respective data arrays.

```c++
int rangexy_inner_plus1[] = {x_min,x_max+1,y_min,y_max+1};

ops_par_loop(accelerate_kernel, "accelerate_kernel", clover_grid, 2, rangexy_inner_plus1,
     ops_arg_dat(density0, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
     ops_arg_dat(volume, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
     ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
     ops_arg_dat(xvel0, 1, S2D_00, "double", OPS_READ),
     ops_arg_dat(xvel1, 1, S2D_00, "double", OPS_INC),
     ops_arg_dat(xarea, 1, S2D_00_0M1, "double", OPS_READ),
     ops_arg_dat(pressure, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
     ops_arg_dat(yvel0, 1, S2D_00, "double", OPS_READ),
     ops_arg_dat(yvel1, 1, S2D_00, "double", OPS_INC),
     ops_arg_dat(yarea, 1, S2D_00_M10, "double", OPS_READ),
     ops_arg_dat(viscosity, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ));
```
## File I/O
## Supported Paralleizations
## Code-generation Flags
## Runtime Flags and Options
