#!/bin/bash


# run test OMP4
# I am inside Cloverleaf folder

rm cloverleaf_openmp4 && rm OpenMP4/*.o
make cloverleaf_openmp4 &> ./Tests/OMP4/outmakecloverleafOMP4
FILE=./Tests/OMP4/outnumberofregistersforkernel
a=$(sed -n "/ptxas info    \: [1-9]* bytes gmem/,/ptxas info    \: Used/p" ./Tests/OMP4/outmakecloverleafOMP4)
echo "$a" >> $FILE

./Tests/scriptparseNumberofregister.pl > ./Tests/OMP4/numberofregister.csv

sort -u ./Tests/OMP4/numberofregister.csv > ./Tests/OMP4/numberofregisterUnique.csv

echo "Finished compilation "

LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/aderango/clang-ykt-install/lib/ nvprof --csv --log-file ./Tests/OMP4/nvprofCloverleafOMP4Classic.csv ./cloverleaf_openmp4

./Tests/scriptparsenvprofClassic.pl > ./Tests/OMP4/nvprofClassicClear.csv

echo "Finished nvprof Classic "


LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/aderango/clang-ykt-install/lib/ nvprof --csv --log-file ./Tests/OMP4/nvprofCloverleafOMP4EventsAndMetrics.csv --events "l1_global_load_hit,l1_global_load_miss,l1_local_load_hit,l1_local_load_miss,l1_local_store_miss,l1_local_store_hit" --metrics 	"l1_cache_global_hit_rate,l1_cache_local_hit_rate,ipc,sm_efficiency_instance,achieved_occupancy,sm_efficiency,warp_execution_efficiency,inst_integer,inst_fp_64,inst_control,local_load_transactions,local_store_transactions" ./cloverleaf_openmp4 

./Tests/scriptparsenvprofEventsAndMetricsChangedlayoutInfo.pl  > ./Tests/OMP4/nvprofEventsAndMetricsChangedlayoutInfo.csv

echo "Finished nvprof Events and Metrics "

# output files
# Number of registers for kernel  - outnumberofregistersforkernel
# Events and Metrics    	  - nvprofCloverleafOMP4
# RunTime 	       		  - nvprofCloverleafOMP4Classic


