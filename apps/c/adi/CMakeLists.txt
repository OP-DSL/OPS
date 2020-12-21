cmake_minimum_required(VERSION 3.8)

list(APPEND ADI_OPS_SRC
                adi.cpp
    )

list(APPEND ADI_OTHER_SRC
    )

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test-cmake.sh
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

BUILD_OPS_C_TRID_SAMPLE(adi ADI_OPS_SRC YES ADI_OTHER_SRC)
