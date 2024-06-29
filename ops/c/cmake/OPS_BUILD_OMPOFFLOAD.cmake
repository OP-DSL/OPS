# Add the Include files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "ompoffload")
set(SRC ${CORE} ${EXTERN} ${OMPOFFLOAD})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "OpenMP::OpenMP_CXX")
set(Opts "")
foreach(Flag IN LISTS OPS_CXXFLAGS_OMPOFFLOAD)
  set(Opt "$<$<COMPILE_LANGUAGE:CXX>:${Flag}>")
  message(STATUS "OPS Opt ${Opt}")
  list(APPEND Opts "${Opt}")
endforeach()
setlib(${LibName} "${SRC}" "${Links}" "${Opts}")
