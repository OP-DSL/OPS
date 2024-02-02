#!/bin/bash
#Uses clang-format to format the back-end library code to conform to the OPS coding guidelines
# ... currently only applies to files within the current directory

for dir in `ls -d */ `; do
  for subdir in `ls -d $dir`; do
    echo  $dir;
    cd $dir
    ls ./*.cu 2> /dev/null
    if [ $? -eq 0 ]
    then
      for file in ./*.cu; do clang-format -i "$file"; done
    fi
    ls ./*.cpp 2> /dev/null
    if [ $? -eq 0 ]
    then
      for file in ./*.cpp; do clang-format -i "$file"; done
    fi
    ls ./*.c 2> /dev/null
    if [ $? -eq 0 ]
    then
      for file in ./*.c; do clang-format -i "$file"; done
    fi
    ls ./*.h 2> /dev/null
    if [ $? -eq 0 ]
    then
      for file in ./*.h; do clang-format -i "$file"; done
    fi
    cd -
  done;
done
