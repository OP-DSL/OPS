cmake_minimum_required(VERSION 3.8)

list(APPEND ADI_OPS_SRC
                adi.cpp
    )

list(APPEND ADI_OTHER_SRC 
    )

BUILD_OPS_C_TRID_SAMPLE(adi ADI_OPS_SRC NO ADI_OTHER_SRC)
