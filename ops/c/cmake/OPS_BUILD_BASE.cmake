# Add the Include files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "seq")
set(SRC ${CORE} ${EXTERN} ${SEQ})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "OpenMP::OpenMP_CXX")
set(Opts "")
setlib(${LibName} "${SRC}" "${Links}" "${Opts}")
if(MPI_FOUND)
  set(TargetName "mpi")
  set(SRC ${MPICORE} ${MPICommonFiles} ${PUREMPI} ${EXTERN})
  #
  set(LibName "${lib_prefix}${TargetName}")
  list(APPEND "MPI::MPI_CXX")
  setlib(${LibName} "${SRC}" "${Links}" "${Opts}")
endif()
