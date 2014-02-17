

// host stub function
void ops_par_loop_accelerate_kernel_stepbymass(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char *p_a[3];
  int  offs[3][2];

  int  count[dim];
  ops_arg args[3] = { arg0, arg1, arg2};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*3);
  int* end = (int*) xmalloc(sizeof(int)*ndim*3);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if (s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if (e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<3; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n];
      end[i*ndim+n]   = e[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<3;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];

  for (int i = 0; i < 3; i++) {
    if(args[i].argtype == OPS_ARG_DAT)
      ops_exchange_halo(&args[i],2);
  }

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    accelerate_kernel_stepbymass(  (double *)p_a[0], (double *)p_a[1], (double *)p_a[2] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<3; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

  ops_mpi_reduce(&arg0,(double *)p_a[0]);
  ops_mpi_reduce(&arg1,(double *)p_a[1]);
  ops_mpi_reduce(&arg2,(double *)p_a[2]);

 }
