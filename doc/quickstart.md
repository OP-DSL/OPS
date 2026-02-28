# OPS Quickstart Guide

Welcome to OPS! This guide will help you get a working OPS application running in minutes.

## 1. Clone the Repository

```bash
git clone https://github.com/OP-DSL/OPS.git
cd OPS
```

## 2. Set Up Your Environment

- Load required modules for your system (compiler, MPI, CUDA, etc.). Consult your cluster or system documentation for details.
- If a setup script is provided for your platform (see `source_files/`), source it. Otherwise, set environment variables (e.g., `OPS_INSTALL_PATH`, `CUDA_INSTALL_PATH`, `MPI_INSTALL_PATH`) as needed for your environment.
 You must set up a Python virtual environment, as the OPS code generator (ops_translator) is required to generate the parallelized application code. For more details, see `doc/installation.md`.
 
 From the OPS root directory, run:
 ```bash
 cd ops_translator
 . setup_venv.sh
 source ops_venv/bin/activate
 cd ..
 ```

## 3. Build a Sample Application

Navigate to an example app, e.g. CloverLeaf:
```bash
cd apps/c/CloverLeaf
bash test.sh
```
This will build and run all supported parallel versions for your environment.

## 4. Check Results

- Output and performance logs will be generated in the app directory.
- Check for "PASSED" in the output to confirm success.

## 5. Next Steps

- See `doc/installation.md` for detailed installation and platform-specific setup instructions.
- Explore other applications in `apps/c/` and `apps/fortran/`.
- See `doc/apps.md` for details on each example.
- See `doc/devdoc.md` for developer and contributor information.

For more help, see the full documentation or open an issue on the OPS GitHub.