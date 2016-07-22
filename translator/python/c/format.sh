#!/bin/bash
#Uses clang-format to format code to conform to the OPS coding guidelines
# ... currently only applies to files within the current directory
# also only format the files producded by the code generator (i.e. files with kernel in their name)

#for file in ./*.cu ./*.cpp ./*.h ./*.hpp; do clang-format "$file" > "$file"_temp; mv "$file"_temp "$file"; done
for dir in `ls -d */ `; do
  ls ./*_ops.cpp 2> /dev/null
  if [ $? -eq 0 ]
  then
    for file in ./*_ops.cpp; do clang-format -i "$file"; done
  fi
  for subdir in `ls -d $dir`; do
    echo  $dir;
    cd $dir
    ls ./*kernel* 2> /dev/null
    if [ $? -eq 0 ]
    then
      for file in ./*kernel*; do clang-format -i "$file"; done
    fi
    ls ./*_ops.cpp 2> /dev/null
    if [ $? -eq 0 ]
    then
      for file in ./*_ops.cpp; do clang-format -i "$file"; done
    fi
    cd -
  done;
done
