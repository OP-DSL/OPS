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

add_subdirectory(poisson)
add_subdirectory(CloverLeaf)
add_subdirectory(CloverLeaf_3D)


