# Build the basic sequential target
set(TargetName "ompoffload")
set(SRC ${CORE} ${EXTERN} ${OMPOFFLOAD})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "OpenMP::OpenMP_CXX")
set(Opts "")
set(Defs "")
set(Deps "")
foreach(Flag IN LISTS OPS_CXXFLAGS_OMPOFFLOAD)
  set(Opt "$<$<COMPILE_LANGUAGE:CXX>:${Flag}>")
  list(APPEND Opts "${Opt}")
endforeach()
setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
if(MPI_FOUND)
  set(TargetName "mpi_ompoffload")
  set(SRC ${MPICORE} ${MPICommonFiles} ${MPIOMPOFFLOAD} ${EXTERN})
  #
  set(LibName "${lib_prefix}${TargetName}")
  list(APPEND Links "MPI::MPI_CXX")
  setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
endif()

