# Add the Include files
message(STATUS "Build HIP")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "hip")
set(SRC ${CORE} ${EXTERN} ${HIP})
#
set(LibName "${lib_prefix}${TargetName}")
set(Links "OpenMP::OpenMP_CXX"
          "hip::device")

# For ROCm >=3.5, wipe hip-clang specific interface options which are propagated
set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC")
set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")

set(MY_HIPCC_OPTIONS "-fPIC")
set(MY_HCC_OPTIONS "")
set(MY_NVCC_OPTIONS "")
set(MY_CLANG_OPTIONS "")
set(STATIC_OR_SHARED STATIC)
set(Opts "")
set_source_files_properties(${SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

HIP_ADD_LIBRARY(${LibName} ${SRC})
	#	        HIPCC_OPTIONS "${MY_HIPCC_OPTIONS}" 
	#	${STATIC_OR_SHARED})
#HIP_ADD_LIBRARY(${LibName} ${SRC} 
#	        HIPCC_OPTIONS "${MY_HIPCC_OPTIONS}" 
#		HCC_OPTIONS "${MY_HCC_OPTIONS}" 
#		NVCC_OPTIONS "${MY_NVCC_OPTIONS}" 
#		CLANG_OPTIONS "${MY_CLANG_OPTIONS}" 
#		${STATIC_OR_SHARED})
set_target_properties(${LibName} PROPERTIES LINKER_LANGUAGE HIP)
foreach(Link IN LISTS Links)
  target_link_libraries(${LibName} PRIVATE ${Link})
endforeach()
target_link_libraries(${LibName}
  PRIVATE rocrand
  PRIVATE hiprand
  PRIVATE rocsparse
)

# Additional flags only for this target
foreach(Opt IN LISTS Opts)
  target_compile_options(${LibName} PRIVATE ${Opt})
endforeach()
installtarget(${LibName} ${ConfigPackageLocation})


#if(MPI_FOUND)
#  set(TargetName "mpi_hip")
#  set(SRC ${MPICORE} ${EXTERN} ${MPIHIP} ${MPICommonFiles})
#  #
#  set(LibName "${lib_prefix}${TargetName}")
#  list(APPEND Links "MPI::MPI_CXX")
#  HIP_ADD_LIBRARY(${LibName} ${SRC} 
#  	        HIPCC_OPTIONS "${MY_HIPCC_OPTIONS}" 
#  		${STATIC_OR_SHARED})
#  set_target_properties(${LibName} PROPERTIES LINKER_LANGUAGE HIP)
#  foreach(Link IN LISTS Links)
#    target_link_libraries(${LibName} PRIVATE ${Link})
#  endforeach()
#  # Additional flags only for this target
#  foreach(Opt IN LISTS Opts)
#    target_compile_options(${LibName} PRIVATE ${Opt})
#  endforeach()
#  installtarget(${LibName} ${ConfigPackageLocation})
#endif()
