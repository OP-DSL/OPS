# Build the basic fortran HDF5
add_library(ops_for_hdf5_common ${F_HDF})
#target_link_libraries(ops_for_hdf5_common PUBLIC ops_for_common)

#set(TargetName "hdf5_seq")
#set(SRC ${HDF})
##
#set(LibName "${lib_prefix}${TargetName}")
#set(Links "ops_for_common"
#	  "ops_for_hdf5_common"
#	  "ops_for_seq"
#	  "hdf5::hdf5"
#          "hdf5::hdf5_hl"
#	  "MPI::MPI_CXX"
#          "MPI::MPI_Fortran")
#set(Opts "")
#set(Defs "OPS_FTN")
#set(Deps "")
#setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
#if(MPI_FOUND)
#  set(TargetName "hdf5_mpi")
#  set(SRC ${HDF_MPI})
#  set(Links "ops_for_common"
#	    "ops_for_hdf5_common"
#	    "ops_for_mpi"
#            "hdf5::hdf5"
#            "hdf5::hdf5_hl"
#            "MPI::MPI_CXX"
#            "MPI::MPI_Fortran")
#  set(Deps "")
#  #
#  set(LibName "${lib_prefix}${TargetName}")
#  setlib(${LibName} "${SRC}" "${Links}" "${Opts}" "${Defs}" "${Deps}")
#endif()

