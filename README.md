OPS
===

OPS is an API with associated libraries and preprocessors to generate 
parallel executables for applications on mulit-block structured grids.


This repository contains the implementation of the run-time library
and the preprocessor, and is structured as follows:

* ops: Implementation of the user and run-time OPS C/C++ APIs

* apps: Application examples in C
  These are examples of user application code and also include
  the target code an OPS preprocessor should produce to correctly
  use the OPS run-time library.
  Currently the main application developed with OPS is a single 
  block structured mesh application - Cloverleaf originally 
  developed at https://github.com/Warwick-PCAV/CloverLeaf

* translator: Pyton OPS preprocessor for C/C++ API

* doc: Documentation
