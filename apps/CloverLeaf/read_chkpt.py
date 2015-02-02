#!/usr/bin/python

import sys
import re
import datetime

names = ['revert_kernel',
'reset_field_kernel1',
'reset_field_kernel2',
'ideal_gas_kernel',
'PdV_kernel_predict',
'PdV_kernel_nopredict',
'accelerate_kernel',
'advec_cell_kernel1_xdir',
'advec_cell_kernel2_xdir',
'advec_cell_kernel3_xdir',
'advec_cell_kernel4_xdir',
'advec_cell_kernel1_ydir',
'advec_cell_kernel2_ydir',
'advec_cell_kernel3_ydir',
'advec_cell_kernel4_ydir',
'advec_mom_kernel_x1',
'advec_mom_kernel_y1',
'advec_mom_kernel_x2',
'advec_mom_kernel_y2',
'advec_mom_kernel_mass_flux_x',
'advec_mom_kernel_post_pre_advec_x',
'advec_mom_kernel1_x_nonvector',
'advec_mom_kernel2_x',
'advec_mom_kernel_mass_flux_y',
'advec_mom_kernel_post_pre_advec_y',
'advec_mom_kernel1_y_nonvector',
'advec_mom_kernel2_y',
'calc_dt_kernel',
'calc_dt_kernel_min',
'calc_dt_kernel_get',
'calc_dt_kernel_print',
'field_summary_kernel',
'flux_calc_kernelx',
'flux_calc_kernely',
'viscosity_kernel',
'initialise_chunk_kernel_xx',
'initialise_chunk_kernel_yy',
'initialise_chunk_kernel_x',
'initialise_chunk_kernel_y',
'initialise_chunk_kernel_cellx',
'initialise_chunk_kernel_celly',
'initialise_chunk_kernel_volume',
'generate_chunk_kernel',
'update_halo_kernel1_b2',
'update_halo_kernel1_b1',
'update_halo_kernel1_t2',
'update_halo_kernel1_t1',
'update_halo_kernel1_l2',
'update_halo_kernel1_l1',
'update_halo_kernel1_r2',
'update_halo_kernel1_r1',
'update_halo_kernel2_xvel_plus_4_a',
'update_halo_kernel2_xvel_plus_2_a',
'update_halo_kernel2_xvel_plus_4_b',
'update_halo_kernel2_xvel_plus_2_b',
'update_halo_kernel2_xvel_minus_4_a',
'update_halo_kernel2_xvel_minus_2_a',
'update_halo_kernel2_xvel_minus_4_b',
'update_halo_kernel2_xvel_minus_2_b',
'update_halo_kernel2_yvel_minus_4_a',
'update_halo_kernel2_yvel_minus_2_a',
'update_halo_kernel2_yvel_minus_4_b',
'update_halo_kernel2_yvel_minus_2_b',
'update_halo_kernel2_yvel_plus_4_a',
'update_halo_kernel2_yvel_plus_2_a',
'update_halo_kernel2_yvel_plus_4_b',
'update_halo_kernel2_yvel_plus_2_b',
'update_halo_kernel3_plus_4_a',
'update_halo_kernel3_plus_2_a',
'update_halo_kernel3_plus_4_b',
'update_halo_kernel3_plus_2_b',
'update_halo_kernel3_minus_4_a',
'update_halo_kernel3_minus_2_a',
'update_halo_kernel3_minus_4_b',
'update_halo_kernel3_minus_2_b',
'update_halo_kernel4_minus_4_a',
'update_halo_kernel4_minus_2_a',
'update_halo_kernel4_minus_4_b',
'update_halo_kernel4_minus_2_b',
'update_halo_kernel4_plus_4_a',
'update_halo_kernel4_plus_2_a',
'update_halo_kernel4_plus_4_b',
'update_halo_kernel4_plus_2_b']

OPS_READ = 0
OPS_WRITE = 1
OPS_RW = 2
OPS_INC = 3
OPS_MAX = 4
OPS_MIN = 5

OPS_NOT_SAVED=0
OPS_SAVED=1
OPS_UNDECIDED=2

f=open('checkp_diags.txt','r')
file_text = f.read()
file_lines = file_text.split('\n')

#first read datasets
datasets = []
ndatasets = 0
datasetnames = []

while len(file_lines[ndatasets]) <> 0:
	data = file_lines[ndatasets].split(';')
	temp = {'name': data[0],
			'type': data[3],
			'size': [int(data[1]), int(data[2])]}
	datasets.append(temp)
	datasetnames.append(data[0])
	ndatasets = ndatasets + 1

file_lines = file_lines[ndatasets+1:]

nkernels = 0;
line = 0;
kernels = []
while file_lines[line] <> 'FINISHED':
	if file_lines[line].split(';')[0] == 'reduction':
		line = line + 1
		continue
	else:
		loop_line = file_lines[line].split(';')
		if not 'loop' in loop_line[0]:
			print 'ERROR, unexpected line ' + file_lines[line]
			sys.exit()
		loopidx = int(loop_line[0][5:])
		nargs = int(loop_line[1])
		looprange = [int(loop_line[2]),int(loop_line[3]),int(loop_line[4]),int(loop_line[5])]
		line = line+1
		args = []
		for i in range(0,nargs):
			arg_line = file_lines[line].split(';')
			if arg_line[0] == 'dat':
				temp = {'argtype' : 'dat',
						'idx' : datasetnames.index(arg_line[1]),
						'name' : arg_line[1],
						'stencil' : arg_line[2],
						'type' : arg_line[3],
						'acc' : int(arg_line[4]),
						'opt' : int(arg_line[5])}
			elif arg_line[0] == 'gbl':
				temp = {'argtype' : 'gbl',
						'dim' : int(arg_line[1]),
						'acc' : int(arg_line[2])}
			elif arg_line[0] == 'idx':
				temp = {'argtype':'idx'}
			args.append(temp)
			line = line + 1
		temp = {'idx': loopidx,
				'nargs' : nargs,
				'range' : looprange,
				'args' : args}
		kernels.append(temp)
		nkernels = nkernels + 1

f.close()

totalsize = 0
for i in range(0,ndatasets):
	size = datasets[i]['size'][0]*datasets[i]['size'][1]
	if datasets[i]['type'] == 'double':
		size = size * 8
	elif datasets[i]['type'] == 'int':
		size = size * 4
	totalsize = totalsize + size
print 'Total number of datasets: ' + str(ndatasets) + ' size: ' + str(totalsize) + ' bytes'

#statistics for kernels
minsaved = [totalsize] * len(names)
maxsaved = [0] * len(names)
avgsaved = [0] * len(names)
timescalled = [0] * len(names)
lastcalled = [0] * len(names)
maxcalled = [0] * len(names)
saved_list = []
for i in range(0,len(names)):
	saved_list.append([OPS_NOT_SAVED] * ndatasets)

#statistics for datasets
ever_written = [0]*ndatasets
totsaved = [0] * nkernels
for i in range (0, nkernels):
	dat_status = [OPS_UNDECIDED]*ndatasets
	saved = 0
	for k in range(i,min(nkernels,i+200)):
		kernel = kernels[k]
#		if i == 1246:
#			print names[kernel['idx']]
		for j in range(0,kernel['nargs']):
			if kernel['args'][j]['argtype'] == 'dat' and kernel['args'][j]['acc'] <> OPS_READ:
				ever_written[kernel['args'][j]['idx']] = 1

		for j in range(0,kernel['nargs']):
			if (kernel['args'][j]['argtype'] == 'dat' and
				kernel['args'][j]['opt'] == 1 and
				ever_written[kernel['args'][j]['idx']] == 1 and
				dat_status[kernel['args'][j]['idx']] == OPS_UNDECIDED and
				kernel['args'][j]['acc'] <> OPS_WRITE):
				#saved
				dat_status[kernel['args'][j]['idx']] = OPS_SAVED
				this_saved = datasets[kernel['args'][j]['idx']]['size'][0] * datasets[kernel['args'][j]['idx']]['size'][1]
				if datasets[kernel['args'][j]['idx']]['type'] == 'double':
					this_saved = this_saved * 8
				elif datasets[kernel['args'][j]['idx']]['type'] == 'int':
					this_saved = this_saved * 4
				saved = saved + this_saved
#				if i == 1246:
#					print 'saved '+kernel['args'][j]['name']


			elif (kernel['args'][j]['argtype'] == 'dat' and
				kernel['args'][j]['opt'] == 1 and
				dat_status[kernel['args'][j]['idx']] == OPS_UNDECIDED and
				kernel['args'][j]['acc'] == OPS_WRITE):
				#saved
				dat_status[kernel['args'][j]['idx']] = OPS_NOT_SAVED
				this_size = datasets[kernel['args'][j]['idx']]['size'][0] * datasets[kernel['args'][j]['idx']]['size'][1]
				itersizex = kernel['range'][1] - kernel['range'][0]
				itersizey = kernel['range'][3] - kernel['range'][2]
				if (itersizex > datasets[kernel['args'][j]['idx']]['size'][0] or itersizey > datasets[kernel['args'][j]['idx']]['size'][1]):
					this_saved = this_size
				else:
					this_saved = (datasets[kernel['args'][j]['idx']]['size'][0]-itersizex) * datasets[kernel['args'][j]['idx']]['size'][1]
					this_saved = this_saved + datasets[kernel['args'][j]['idx']]['size'][0] * (datasets[kernel['args'][j]['idx']]['size'][1]-itersizey)
#				if i == 1246:
#					print 'partially saved '+kernel['args'][j]['name'] +' ' + str(min(this_saved,this_size)) + '/' + str(this_size)
				if datasets[kernel['args'][j]['idx']]['type'] == 'double':
					this_saved = min(this_saved,this_size) * 8
				elif datasets[kernel['args'][j]['idx']]['type'] == 'int':
					this_saved = min(this_saved,this_size) * 4
				saved = saved + this_saved

		done = 1
		for j in range(0,ndatasets):
			if ever_written[j] and dat_status[j] == OPS_UNDECIDED:
				done = 0
				break

		if done:
			break
#	if i == 1246:
#		print dat_status
	totsaved[i] = saved

	#register stats
	kernel = kernels[i]
	idx = kernel['idx']
	minsaved[idx] = min(minsaved[idx],saved)
	maxsaved[idx] = max(maxsaved[idx],saved)
	timescalled[idx] = timescalled[idx]+1
	avgsaved[idx] = ((timescalled[idx]-1) * avgsaved[idx] + saved)/timescalled[idx]
	maxcalled[idx] = max(maxcalled[idx],i-lastcalled[idx])
	lastcalled[idx] = i
	for j in range(0,ndatasets):
		if dat_status[j] == OPS_SAVED:
			saved_list[idx][j] = OPS_SAVED


print '############ OPS Checkpointing Statistics ############'
print '\nDatasets never written to and excluded from any checkpoint:'
line = ''
for i in range(0,ndatasets):
	if ever_written[i] == 0:
		line = line + datasets[i]['name']+' '
print line[:-1]+'\n'
avgsaved2 = avgsaved
for j in range(0,len(avgsaved2)):
	if (timescalled[j] == 0):
		avgsaved2[j]=totalsize

print 'Top 5 kernels:'
for j in range(0,5):
	minkernidx = avgsaved2.index(min(avgsaved2))
	if timescalled[minkernidx] < 2:
		avgsaved2[minkernidx] = totalsize
		j = j - 1
		continue
	print 'Kernel: '+str(names[minkernidx])+'\navg. saved: ' + str(avgsaved2[minkernidx]) + ' min. saved: ' +str(minsaved[minkernidx]) +\
		' max saved: '+str(maxsaved[minkernidx])+ ' called: '+str(timescalled[minkernidx]) +\
		' times, on average every ' + str(nkernels/timescalled[minkernidx])+ \
	 ' max period between calls: ' + str(maxcalled[minkernidx])
	avgsaved2[minkernidx] = totalsize
	line = ''
	for k in range(0,ndatasets):
		if saved_list[minkernidx][k] == OPS_SAVED:
			line = line + ' ' + datasets[k]['name']
	print 'List of fully saved datasets: ' + line + '\n'


f2 = open('checkp_out.txt','w')
for i in range(0,len(totsaved)):
	f2.write(str(totsaved[i]) + '\n')
f2.close()
