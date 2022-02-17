cmake_minimum_required(VERSION 3.18)
CreateTempDir()
if(CPU_NUMBER GREATER 4)
    set(OringalCPUNum ${CPU_NUMBER})
    set(CPU_NUMBER 4)
else()
    # Note that undefined variables would be `""` thus false
endif()
if (LIBTRID_PATH)
    BUILD_OPS_C_SAMPLE(compact_3D "NONE" "NONE" "NONE" "YES" "YES")
endif ()
set(CPU_NUMBER ${OringalCPUNum})

