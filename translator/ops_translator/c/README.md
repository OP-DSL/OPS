###Invoking the Code Generator

Uncomment the parallelization you want to code generate in ops.py. For example for CUDA code generation do:

```
#ops_gen_mpi(str(sys.argv[1]), date, consts, kernels)
#ops_gen_mpi_openmp(str(sys.argv[1]), date, consts, kernels)
ops_gen_mpi_cuda(str(sys.argv[1]), date, consts, kernels)
```

Make it executable

`chmod a+x ./ops.py`

Invoke the code generator by supplying the files that contain ops_* API calls. Thus for example for CLoverleaf do the following.

```
./ops.py clover_leaf.cpp revert.cpp reset_field.cpp ideal_gas.cpp PdV.cpp \
accelerate.cpp advec_cell.cpp accelerate.cpp advec_mom.cpp calc_dt.cpp \
field_summary.cpp flux_calc.cpp viscosity.cpp update_halo.cpp generate.cpp \
initialise_chunk.cpp
```
