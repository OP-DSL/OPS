#ifndef USER_TYPES
#define USER_TYPES

#define FIELD_DENSITY 0
#define FIELD_ENERGY0 1
#define FIELD_ENERGY1 2
#define FIELD_U       3
#define FIELD_P       4
#define FIELD_SD      5
#define FIELD_R       6
#define NUM_FIELDS    7

#define fixed 1

#define CELL_DATA 0
#define VERTEX_DATA 1
#define X_FACE_DATA 2
#define Y_FACE_DATA 3

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

#define TL_PREC_NONE 1
#define TL_PREC_JAC_DIAG 2
#define TL_PREC_JAC_BLOCK 3

typedef struct
{
      int defined;  //logical
      double density,
             energy;
      int geometry;
      double xmin,
             xmax,
             ymin,
             ymax,
             radius;
} state_type;


typedef struct grid_type
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
