#!/bin/bash -f

# Initialize the lowest time to a very high value
lowest_time=999999
best_j=0
best_k=0
ACCEL=cuda

for k in {8,16,32,64,128,256,512}
do
  for j in {1,2,4,8,16,32}
  do
      if [[ $((k*j)) -gt 1024 ]] || [[ $((k*j)) -lt 64 ]]; then
        break
      fi
      echo "wg size test ${k} ${j} "
      echo "wg size test ${k} ${j} " >> laplace2d_"$ACCEL"_wgtest
      output=$(stdbuf -o0 ./laplace2d_"$ACCEL" -OPS_DIAGS=2 OPS_BLOCK_SIZE_X=${k} OPS_BLOCK_SIZE_Y=${j})
      echo "$output" >> laplace2d_"$ACCEL"_wgtest
      # Extract the time from the output and compare with the lowest time
      time=$(echo "$output" | grep "Total Wall time" | awk '{print $4}')
      if (( $(echo "$time < $lowest_time" | bc -l) )); then
        lowest_time=$time
        best_j=$j
        best_k=$k
      fi
    done
  done

echo "Lowest time: ${lowest_time}"
echo "BS_X=${best_k}"
echo "BS_Y=${best_j}"
