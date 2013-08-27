

#ifndef __OPS_LIB_CPP_H
#define __OPS_LIB_CPP_H

#include <ops_lib_core.h>


/*
* run-time type-checking routines
*/

inline int type_error (const double * a, const char *type ) {
  (void)a; return (strcmp ( type, "double" ) && strcmp ( type, "double:soa" ));
}
inline int type_error (const float * a, const char *type ) {
  (void)a; return (strcmp ( type, "float" ) && strcmp ( type, "float:soa" ));
}
inline int type_error (const int * a, const char *type ) {
  (void)a; return (strcmp ( type, "int" ) && strcmp ( type, "int:soa" ));
}
inline int type_error (const uint * a, const char *type ) {
  (void)a; return (strcmp ( type, "uint" ) && strcmp ( type, "uint:soa" ));
}
inline int type_error (const ll * a, const char *type ) {
  (void)a; return (strcmp ( type, "ll" ) && strcmp ( type, "ll:soa" ));
}
inline int type_error (const ull * a, const char *type ) {
  (void)a; return (strcmp ( type, "ull" ) && strcmp ( type, "ull:soa" ));
}
inline int type_error (const bool * a, const char *type ) {
  (void)a; return (strcmp ( type, "bool" ) && strcmp ( type, "bool:soa" ));
}


extern int OPS_diags;

extern int OPS_block_index, OPS_block_max,
           OPS_dat_index, OPS_dat_max;

extern ops_block * OPS_block_list;
extern Double_linked_list OPS_dat_list; //Head of the double linked list


template < class T >
ops_dat ops_decl_dat ( ops_block block, int data_size,
                      int *block_size, int* offset, T *data,
                      char const * type,
                      char const * name )
{

  if ( type_error ( data, type ) ) {
    printf ( "incorrect type specified for dataset \"%s\" \n", name );
    exit ( 1 );
  }

  return ops_decl_dat_core(block, data_size, block_size, offset, (char *)data, type, name );

}


#endif /* __OP_LIB_CPP_H */
