# Build the basic sequential target
set(TargetName "hdf5_seq")
set(SRC ${HDF})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "hdf5::hdf5"
          "hdf5::hdf5_hl"
          "MPI::MPI_CXX")
set(Opts "")
set(Defs "")
set(Deps "")
setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
if(MPI_FOUND)
  set(TargetName "hdf5_mpi")
  set(SRC ${HDF_MPI})
  #
  set(LibName "${lib_prefix}${TargetName}")
  setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
endif()

