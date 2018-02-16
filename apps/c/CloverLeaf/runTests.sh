#!/bin/bash

rm -r Tests/OMP4
rm -r Tests/CUDA

#mkdir -p ./Tests/

mkdir -p ./Tests/OMP4
mkdir -p ./Tests/OMP4/tmp

mkdir -p ./Tests/CUDA
mkdir -p ./Tests/CUDA/tmp


OMPApplication=$1
CUDAApplication=$2

########################## OMP4  ###########################

# run test OMP4
# I am inside Application folder

rm $OMPApplication && rm OpenMP4/*.o
make $OMPApplication &> ./Tests/OMP4/tmp/outmakeOMP4
FILE=./Tests/OMP4/tmp/outnumberofregistersforkernel
a=$(sed -n "/ptxas info    \: [1-9]* bytes gmem/,/ptxas info    \: Used/p" ./Tests/OMP4/tmp/outmakeOMP4)
echo "$a" >> $FILE

./Tests/scriptparseNumberofregister.pl ./Tests/OMP4/tmp/outnumberofregistersforkernel > ./Tests/OMP4/tmp/numberofregister.csv

sort -u ./Tests/OMP4/tmp/numberofregister.csv > ./Tests/OMP4/numberofregisterUnique.csv



echo "Finished compilation OMP4"

LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/aderango/clang-ykt-install/lib/ nvprof --csv --log-file ./Tests/OMP4/tmp/nvprofOMP4Classic.csv ./$OMPApplication

./Tests/scriptparsenvprofClassic.pl ./Tests/OMP4/tmp/nvprofOMP4Classic.csv > ./Tests/OMP4/nvprofClassicClear.csv

echo "Finished nvprof Classic OMP4"


LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/aderango/clang-ykt-install/lib/ nvprof --csv --log-file ./Tests/OMP4/tmp/nvprofOMP4EventsAndMetrics.csv --events "l1_global_load_hit,l1_global_load_miss,l1_local_load_hit,l1_local_load_miss,l1_local_store_miss,l1_local_store_hit" --metrics 	"l1_cache_global_hit_rate,l1_cache_local_hit_rate,ipc,sm_efficiency_instance,achieved_occupancy,sm_efficiency,warp_execution_efficiency,inst_integer,inst_fp_64,inst_control,local_load_transactions,local_store_transactions" ./$OMPApplication 

./Tests/scriptparsenvprofEventsandMetricsChangedlayoutInfo.pl  ./Tests/OMP4/tmp/nvprofOMP4EventsAndMetrics.csv > ./Tests/OMP4/nvprofEventsAndMetricsChangedlayoutInfo.csv

echo "Finished nvprof Events and Metrics OMP4"

########################## OMP4 END ###########################




########################## CUDA  ###########################
rm $CUDAApplication && rm CUDA/*.o
make $CUDAApplication &> ./Tests/CUDA/tmp/outmake
FILE=./Tests/CUDA/tmp/outnumberofregistersforkernel
a=$(sed -n "/ptxas info    \: [1-9]* bytes gmem/,/ptxas info    \: Used/p" ./Tests/CUDA/tmp/outmake)
echo "$a" >> $FILE

./Tests/scriptparseNumberofregister.pl ./Tests/CUDA/tmp/outnumberofregistersforkernel > ./Tests/CUDA/tmp/numberofregister.csv

sort -u ./Tests/CUDA/tmp/numberofregister.csv > ./Tests/CUDA/numberofregisterUnique.csv

echo "Finished compilation CUDA"

LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/aderango/clang-ykt-install/lib/ CUDA_VISIBLE_DEVICES=1 nvprof --csv --log-file ./Tests/CUDA/tmp/nvprofCUDAClassic.csv ./$CUDAApplication

./Tests/scriptparsenvprofClassic.pl ./Tests/CUDA/tmp/nvprofCUDAClassic.csv > ./Tests/CUDA/nvprofClassicClear.csv

perl -i -pe 's/\(.*?\)//g' ./Tests/CUDA/nvprofClassicClear.csv


echo "Finished nvprof Classic CUDA"

LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/aderango/clang-ykt-install/lib/ CUDA_VISIBLE_DEVICES=1 nvprof --csv --log-file ./Tests/CUDA/tmp/nvprofCUDAEventsAndMetrics.csv --events "l1_global_load_hit,l1_global_load_miss,l1_local_load_hit,l1_local_load_miss,l1_local_store_miss,l1_local_store_hit" --metrics 	"l1_cache_global_hit_rate,l1_cache_local_hit_rate,ipc,sm_efficiency_instance,achieved_occupancy,sm_efficiency,warp_execution_efficiency,inst_integer,inst_fp_64,inst_control,local_load_transactions,local_store_transactions" ./$CUDAApplication 

perl -i -pe 's/\(.*?\)//g' ./Tests/CUDA/tmp/nvprofCUDAEventsAndMetrics.csv

./Tests/scriptparsenvprofEventsandMetricsChangedlayoutInfo.pl ./Tests/CUDA/tmp/nvprofCUDAEventsAndMetrics.csv > ./Tests/CUDA/nvprofEventsAndMetricsChangedlayoutInfo.csv

echo "Finished nvprof Events and Metrics CUDA"
########################## CUDA END ###########################


# output files
# Number of registers for kernel  - outnumberofregistersforkernel
# Events and Metrics    	  - nvprofOMP4
# RunTime 	       		  - nvprofOMP4Classic


