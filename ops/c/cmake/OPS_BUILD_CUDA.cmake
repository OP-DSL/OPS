# Build the basic sequential target
set(TargetName "cuda")
set(SRC ${CORE} ${EXTERN} ${CUDA})
#
set(LibName "${lib_prefix}${TargetName}")
if (USE_OMP)
  set(Links "OpenMP::OpenMP_CXX"
            "CUDA::cudart_static"
            "NVML::nvml")
else ()
  set(Links "CUDA::cudart_static"
            "NVML::nvml")
endif()
set(Opts "")
set(Defs "")
set(Deps "")
setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
if(MPI_FOUND)
  set(TargetName "mpi_cuda")
  set(SRC ${MPICORE} ${EXTERN} ${MPICUDA} ${MPICommonFiles})
  #
  set(LibName "${lib_prefix}${TargetName}")
  list(APPEND Links "MPI::MPI_CXX")
  setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
endif()
