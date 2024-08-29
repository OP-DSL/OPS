set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OPS_BUILD_APPS)

# Create a folder directory for testing
set(app_dir_c "${app_dir}/c")
file(MAKE_DIRECTORY ${app_dir_c})

# Include include directory for C++
include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")

# Set the OPS Translator 
if(LEGACY_CODEGEN)
  set(OPS_C_TRANSLATOR "${opsc}/ops.py")
else()
  set(OPS_C_TRANSLATOR "${opsc}")
endif()


if(${MPI_FOUND})
  if(DEFINED ENV{I_MPI_ROOT})     #Intel MPI
    set(MPI_INC_LIST -I$ENV{I_MPI_ROOT}/include)
  elseif(DEFINED ENV{MPI_ROOT})       # HPE MPT
    set(MPI_INC_LIST -I$ENV{MPI_ROOT}/include)
  elseif(DEFINED ENV{MPICH_DIR})      # Cray
    set(MPI_INC_LIST -I$ENV{MPICH_DIR}/include)
  else()                          # OpenMPI
    execute_process(COMMAND mpicc --showme:compile
                    OUTPUT_VARIABLE MPI_INC_STRING
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_VARIABLE MPI_INC_FIND)
    if(MPI_INC_FIND)
        message(FATAL_ERROR "Failed to determine MPI include directories: ${MPI_INC_FIND}")
    endif()
    string(REPLACE " " ";" MPI_INC_LIST ${MPI_INC_STRING})
  endif()
endif()

#add_subdirectory(access) #This does not compile
add_subdirectory(CloverLeaf)
#add_subdirectory(complex_numbers) #This does not compile
add_subdirectory(laplace2d_tutorial/step7)
add_subdirectory(ops-lbm/step5)
#add_subdirectory(lowdim_test) # Issue with mpi-cuda with multiple definition
#add_subdirectory(mblock) # Issue with mpi-cuda with multiple definition
#add_subdirectory(mgrid)  # Issue with mpi-cuda with multiple definition
add_subdirectory(multiDim)
add_subdirectory(multiDim3D)
add_subdirectory(poisson)
add_subdirectory(shsgc)
if(${OPS_TEST_FULL})
  add_subdirectory(CloverLeaf_3D)
  #add_subdirectory(CloverLeaf_3D_HDF5)
  add_subdirectory(mb_shsgc)
endif()
