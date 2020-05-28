# Create the docker image for GNU and Clang
FROM nvidia/cuda
ADD https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.sh /cmake-3.17.2-Linux-x86_64.sh
RUN apt update -y \
    && apt install gfortran libhdf5-openmpi-dev ssh -y \
    && apt autoclean \
    && mkdir /opt/cmake \
    && sh /cmake-3.17.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && cmake --version \
    && gcc --version \
