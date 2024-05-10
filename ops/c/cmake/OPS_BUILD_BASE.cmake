# Add the Include files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "seq")
set(SRC ${CORE} ${EXTERN} ${SEQ})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "OpenMP::OpenMP_CXX")
message(STATUS "SRC LIST1: ${SRC}")
setlib(${LibName} "${SRC}" "${Links}")
#add_library(${LibName} ${SRC})
#target_link_libraries(${LibName} PRIVATE OpenMP::OpenMP_CXX)
#installtarget(${LibName} ${ConfigPackageLocation})
