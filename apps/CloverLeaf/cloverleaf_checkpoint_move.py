#!/usr/bin/python
import sys
import os
import os.path
import time


rank = -1
size = -1
if os.environ.get('PMI_SIZE') <> None:
  rank = int(os.environ.get('PMI_RANK'))
  size = int(os.environ.get('PMI_SIZE'))
if os.environ.get('OMPI_COMM_WORLD_SIZE') <> None:
  rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
  size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
if os.environ.get('MV2_COMM_WORLD_SIZE') <> None:
  rank = int(os.environ.get('MV2_COMM_WORLD_RANK'))
  size = int(os.environ.get('MV2_COMM_WORLD_SIZE'))
if rank == -1 or size == -1:
  print 'Error: could not determine rank'
  sys.exit(0)

fname = sys.argv[1]
new_path = sys.argv[2]
dup_stride = int(sys.argv[3])

myfile = fname+'.'+str(rank)
myfile2 = fname+'.'+str((rank+ dup_stride)%size)+'.dup'
print 'rank '+str(rank)+'/'+str(size)+' file: '+myfile+' file2: '+myfile2+'\n'
sys.stdout.flush()

while True:
  if os.path.isfile(myfile+'.done'):
    os.remove(myfile+'.done')
    break
  if os.path.isfile(myfile+'.lock'):
    os.rename(myfile,new_path+myfile)
    os.remove(myfile+'.lock')
  if os.path.isfile(myfile2+'.lock'):
    os.rename(myfile2,new_path+myfile2)
    os.remove(myfile2+'.lock')

  if (not os.path.isfile(myfile+'.lock')) and (not os.path.isfile(myfile2+'.lock')):
    time.sleep(1.0)

