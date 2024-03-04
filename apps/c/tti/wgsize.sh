#!/bin/bash -f

# Initialize the lowest time to a very high value
lowest_time=999999
best_i=0
best_j=0
best_k=0

for k in {8,16,32,64,128}
do
  for j in {1,2,4,8,16,32}
  do
    for i in {1,2,4,8,16,32}
    do
      if [[ $((k*j*i)) -gt 1024 ]] || [[ $((k*j*i)) -lt 64 ]]; then
        break
      fi
      echo "wg size test ${k} ${j} ${i}"
      echo "wg size test ${k} ${j} ${i}" >> tti_"$ACCEL"_wgtest3d
      output=$(stdbuf -o0 ./tti_"$ACCEL" 592 592 592 -OPS_DIAGS=2 OPS_SYCL_DEVICE=$SYCL_DEVICE -gpudirect OPS_BLOCK_SIZE_X=${k} OPS_BLOCK_SIZE_Z=${j} OPS_BLOCK_SIZE_Y=${i})
      echo "$output" >> tti_"$ACCEL"_wgtest3d
      # Extract the time from the output and compare with the lowest time
      time=$(echo "$output" | grep "Total Wall time" | awk '{print $4}')
      if (( $(echo "$time < $lowest_time" | bc -l) )); then
        lowest_time=$time
        best_i=$i
        best_j=$j
        best_k=$k
      fi
    done
  done
done

echo "Lowest time: ${lowest_time}"
echo "BS_X=${best_k}"
echo "BS_Y=${best_i}"
echo "BS_Z=${best_j}"
