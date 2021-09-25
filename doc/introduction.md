# Introduction

## Overview

[OPS](https://github.com/OP-DSL/OPS)(Oxford Parallel library for Structured mesh solvers) is a high-level embedded domain specific language for writing **multi-block structured mesh** algorithms, and the corresponding software library and code translation tools to enable automatic parallelisation of the high-level code on multi-core and many-core architectures. Multi-block structured grids consists of an unstructured collection of structured meshes/grids. These pages provide detailed documentation on using OPS, including an installation guide, developing and running OPS applications, the OPS API, developer documentation and performance tuning.

## Licencing
OPS is released as an open-source project under the BSD 3-Clause License. See the file called [LICENSE](https://github.com/OP-DSL/OPS/blob/master/LICENSE) for more information.

## Citing
To cite OPS, please reference the following paper:

[I. Z. Reguly, G. R. Mudalige and M. B. Giles, Loop Tiling in Large-Scale Stencil Codes at Run-Time with OPS, in IEEE Transactions on Parallel and Distributed Systems, vol. 29, no. 4, pp. 873-886, 1 April 2018, doi: 10.1109/TPDS.2017.2778161.](https://ieeexplore.ieee.org/abstract/document/8121995)

```
@ARTICLE{Reguly_et_al_2018,
  author={Reguly, István Z. and Mudalige, Gihan R. and Giles, Michael B.},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={Loop Tiling in Large-Scale Stencil Codes at Run-Time with OPS}, 
  year={2018},
  volume={29},
  number={4},
  pages={873-886},
  doi={10.1109/TPDS.2017.2778161}}
```
Full list of publications from the OPS project can be found in the [Publications](https://ops-dsl.readthedocs.io/en/markdowndocdev/pubs.html) section.

## Support
The preferred method of reporting bugs and issues with OPS is to submit an issue via the repository’s issue tracker. Users can also email the authors directly by  contacting the [OP-DSL team](https://op-dsl.github.io/about.html). 

## Funding
The development of OPS was in part supported by the UK Engineering and Physical Sciences Research Council (EPSRC) grants [EP/K038494/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/K038494/1) (“Future-proof massively-parallel execution of multi-block applications”), [EP/J010553/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/J010553/1) (“Software for Emerging Architectures - ASEArch"), The UK Turbulence Consortium grant [EP/T026170/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/T026170/1), The Janos Bolyai Research Scholarship of the Hungarian Academy of Sciences, the Royal Society through their Industry Fellowship Scheme (INF/R1/180012), and the Thematic Research Cooperation Establishing Innovative Informatic and Info-communication Solutions Project, which has been supported by the European Union and co-financed by the European Social Fund under grant number EFOP-3.6.2-16-2017-00013. Research funding support was also provided by the UK AWE under grants CDK0660 ("The Production of Predictive Models for Future Computing Requirements"), CDK0724 ("AWE Technical Outreach Programme"), AWE grant for "High-level Abstractions for Performance, Portability and Continuity of Scientific Software on Future Computing Systems" and the Numerical Algorithms Group [NAG](https://www.nag.com/).

Hardware resources for development and testing provided by the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725, the [ARCHER](http://www.archer.ac.uk) and ARCHER2(https://www.archer2.ac.uk/) UK National Supercomputing Service, [University of Oxford Advanced Research Computing (ARC) facility](http://dx.doi.org/10.5281/zenodo.22558) and through hardware donations and access provided by NVIDIA and Intel.
