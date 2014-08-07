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
#define FIELD_ZVEL0       11
#define FIELD_ZVEL1       12
#define FIELD_VOL_FLUX_X  13
#define FIELD_VOL_FLUX_Y  14
#define FIELD_VOL_FLUX_Z  15
#define FIELD_MASS_FLUX_X 16
#define FIELD_MASS_FLUX_Y 17
#define FIELD_MASS_FLUX_Z 18
#define NUM_FIELDS        19

typedef struct
{
      int defined;  //logical
      double density,
             energy,
             xvel,
             yvel,
             zvel;
      int geometry;
      double xmin,
             xmax,
             ymin,
             ymax,
             zmin,
             zmax,
             radius;
} state_type;


typedef struct
{
  double  xmin, ymin, xmax, ymax, zmin,zmax;
  int x_cells, y_cells, z_cells;
} grid_type;


typedef struct field_type
{
  int left, right, bottom, top, back, front, left_boundary, right_boundary,
      bottom_boundary, top_boundary, back_boundary, front_boundary;
  int x_min, y_min, x_max ,y_max, z_min, z_max;
} field_type;

#ifdef __cplusplus
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
