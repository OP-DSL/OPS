# Add the Include files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "seq")
set(SRC ${CORE} ${EXTERN} ${SEQ})
#
set(LibName "${lib_prefix}${TargetName}")
if (USE_OMP)
  set(Links "OpenMP::OpenMP_CXX")
else ()
  set(Links "")
endif()
set(Opts "")
setlib(${LibName} "${SRC}" "${Links}" "${Opts}")
if(MPI_FOUND)
  set(TargetName "mpi")
  set(SRC ${MPICORE} ${MPICommonFiles} ${PUREMPI} ${EXTERN})
  #
  set(LibName "${lib_prefix}${TargetName}")
  list(APPEND Links "MPI::MPI_CXX")
  setlib(${LibName} "${SRC}" "${Links}" "${Opts}")
endif()
