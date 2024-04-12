find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)                                                              
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE}) 
   enable_language(CUDA)
  if(NOT SET_GPU_ARCH)  
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    message(STATUS "CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} ${SET_GPU_ARCH}")
    list(GET CUDA_ARCH_LIST 0 SET_CUDA_ARCH)
    message(STATUS "CUDA Architecture from autodetect ${SET_CUDA_ARCH}")	  
    message(STATUS "If different architecture has to be used set -DSET_GPU_ARCH=XXX!")           
  endif()                                                                         
  set(CMAKE_CUDA_ARCHITECTURES ${SET_GPU_ARCH}                                                                
      CACHE STRING "CUDA architectures")   
  
else()                                                                             
  message(WARNING "CUDA toolkit not found! The CUDA codes won't compile!")      
endif()

