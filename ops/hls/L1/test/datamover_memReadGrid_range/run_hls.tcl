
source settings.tcl

set DUT_PROJECT "dut.prj"
set SOLUTION "solution1"

if {![info exists CLKP]} {
  set CLKP 3.30
}

open_project -reset $DUT_PROJECT

add_files "top.cpp" -cflags "-I${PROJ_ROOT}/L1/include"
add_files -tb "main.cpp" -cflags "-I${PROJ_ROOT}/L1/include"
set_top dut

open_solution -reset $SOLUTION

set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

exit
