cmake_minimum_required(VERSION 3.8)

list(APPEND CL_OPS_SRC
                clover_leaf.cpp initialise_chunk.cpp generate.cpp ideal_gas.cpp
                update_halo.cpp field_summary.cpp viscosity.cpp
                calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp flux_calc.cpp
                advec_cell.cpp advec_mom.cpp reset_field.cpp
    )

list(APPEND CL_OTHER_SRC
                start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp
    )

BUILD_OPS_C_SAMPLE(CloverLeaf CL_OPS_SRC NO CL_OTHER_SRC)
