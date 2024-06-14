# Add the Include files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "cuda")
set(SRC ${CORE} ${EXTERN} ${CUDA})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "OpenMP::OpenMP_CXX"
           "CUDA::cudart_static")
setlib(${LibName} "${SRC}" "${Links}")
