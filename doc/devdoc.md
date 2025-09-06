# Developer Guide
Under construction.
<!-- 
## Code-generator
### Frontend API parser
### Target Parallel Templates
### Elemental Kernel Transformations
## Back-end Library
### Sequential and multi-threaded CPU
### MPI and Partitioning
### HDF5
### CUDA
### Cache blocking tiling and comm-avoiding optimizations
-->
## Contributing

To contribute to OPS, please use the following steps:
1. Clone the [OPS](https://github.com/OP-DSL/OPS) repository on your local system.
2. Create a new branch in your cloned repository.
3. Make changes or contributions in your new branch.
4. Submit your changes by creating a pull request to the `develop` branch of the OPS repository.

Contributions in the `develop` branch will be merged into the `master` branch when a new release is created.

<!--
## Git workflow for contribution
To facilitate the concept of "Version" and "Release", we adopt the [Gitflow Workflow model](https://nvie.com/posts/a-successful-git-branching-model/).
### Overall workflow

1. Create a develop branch from main.

2. Create a release branch from develop.

   After creating a release branch, only documentation and bug fixes will be added to this branch.

3. Create feature branches from develop.

4. Merge a feature branch into the develop branch once completed.

5. Merge the release branch into develop and main once completed.

6. Create a hotfix branch from main if an issue is identified.

7. Merge a hotfix branch to both develop and main once fixed.

See also https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow.

### A few issues
Using the Gitflow model tends to produce a few long-lived branches (e.g., feature), which may increase the risk of "conflicts" for integration. To mitigate this, we encourage the following practices:

* Try to create short-lived branches with a few small commits when possible (e.g., a hotfix branch).
* Once a branch is properly merged or a feature is finalized, delete the branch.
* A feature branch tends to be long-lived; try to split a feature into "milestones" and merge into the develop branch when finishing each milestone.

**The Gitflow tool will automatically delete a branch once it is finished.**
### Gitflow tool

See https://github.com/nvie/gitflow
-->
