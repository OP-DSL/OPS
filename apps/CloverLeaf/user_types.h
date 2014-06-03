#ifndef USER_TYPES
#define USER_TYPES

#define FIELD_DENSITY0    0
#define FIELD_DENSITY1    1
#define FIELD_ENERGY0     2
#define FIELD_ENERGY1     3
#define FIELD_PRESSURE    4
#define FIELD_VISCOSITY   5
#define FIELD_SOUNDSPEED  6
#define FIELD_XVEL0       7
#define FIELD_XVEL1       8
#define FIELD_YVEL0       9
#define FIELD_YVEL1       10
#define FIELD_VOL_FLUX_X  11
#define FIELD_VOL_FLUX_Y  12
#define FIELD_MASS_FLUX_X 13
#define FIELD_MASS_FLUX_Y 14
#define NUM_FIELDS        15

typedef struct
{
      int defined;  //logical
      double density,
             energy,
             xvel,
             yvel;
      int geometry;
      double xmin,
             xmax,
             ymin,
             ymax,
             radius;
} state_type;


typedef struct
{
  double  xmin, ymin, xmax, ymax;
  int x_cells, y_cells;
} grid_type;


typedef struct field_type
{
  int left, right, bottom, top ,left_boundary, right_boundary,
      bottom_boundary, top_boundary;
  int x_min, y_min, x_max ,y_max;
} field_type;

#ifdef __cplusplus
#ifndef __OPENCL_VERSION__
inline int type_error (const field_type * a, const char *type ) {
  (void)a; return (strcmp ( type, "field_type" ) && strcmp ( type, "field_type:soa" ));
}

inline int type_error (const grid_type * a, const char *type ) {
  (void)a; return (strcmp ( type, "grid_type" ) && strcmp ( type, "grid_type:soa" ));
}

inline int type_error (const state_type * a, const char *type ) {
  (void)a; return (strcmp ( type, "state_type" ) && strcmp ( type, "state_type:soa" ));
}
#endif

#endif

#endif
