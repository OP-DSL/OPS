# Developer Guide

This guide provides an overview of OPS internals for developers who wish to contribute to OPS, add new backends, or understand how the code generation and runtime library work.

## Architecture Overview

OPS consists of two main components:

1. **Code Generator** (`ops_translator/`): A Python-based source-to-source translator that parses user applications (using libclang for C++ and fparser2 for Fortran) and generates parallel code for various backends.
2. **Runtime Library** (`ops/c/` and `ops/fortran/`): Backend-specific implementations that handle data management, parallelization, and communication.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Application                            │
│                    (ops_par_loop calls + kernels)                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Code Generator (ops_translator)                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   Parser    │───>│   Scheme    │───>│   Jinja2 Templates      │  │
│  │ (libclang/  │    │  (target    │    │  (loop_host, master_    │  │
│  │  fparser2)  │    │   logic)    │    │   kernel, etc.)         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Generated Parallel Code                       │
│      (CUDA, HIP, SYCL, OpenMP, OpenMP Offload + MPI variants)       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Runtime Library (ops/c/src/)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  core/   │  │  cuda/   │  │  sycl/   │  │   mpi/   │    ...      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Code Generator (ops_translator)

The code generator is located in `ops_translator/ops-translator/` and uses Python with Clang bindings (libclang) for C++ parsing and fparser2 for Fortran parsing.

### Directory Structure

```
ops_translator/
├── ops-translator/          # Main translator code
│   ├── __main__.py          # Entry point & CLI argument handling
│   ├── scheme.py            # Code generation schemes (genLoopHost)
│   ├── target.py            # Target definitions (Cuda, Sycl, Hip, etc.)
│   ├── ops.py               # OPS constructs (Loop, Arg, Dat, etc.)
│   ├── store.py             # Application, Program, ParseError classes
│   ├── util.py              # Utilities, KernelProcess class
│   ├── language.py          # Language definitions (C++, Fortran)
│   ├── jinja_utils.py       # Jinja2 environment setup
│   ├── cpp/                 # C++ specific code
│   │   ├── parser.py        # Clang-based C++ parser
│   │   ├── schemes.py       # C++ target scheme implementations
│   │   └── translator/      # Kernel/program translators
│   └── fortran/             # Fortran specific code
├── resources/               # Code generation resources
│   └── templates/           # Jinja2 templates
│       ├── cpp/             # C++ templates
│       │   ├── loop_host.cpp.j2      # Base loop host template
│       │   ├── master_kernel.cpp.j2  # Master kernel file
│       │   ├── cuda/                 # CUDA-specific templates
│       │   ├── sycl/                 # SYCL-specific templates
│       │   ├── mpi_openmp/           # MPI+OpenMP templates
│       │   └── ...
│       └── fortran/         # Fortran templates
└── ops_venv/                # Python virtual environment
```

### Key Classes

#### Target (`target.py`)

Defines code generation targets and their configurations:

```python
class Target(Findable):
    name: str                    # Target identifier (e.g., "cuda", "sycl")
    kernel_translation: bool     # Whether kernel code needs transformation
    config: Dict[str, Any]       # Target-specific configuration
```

Available targets:
| Target Class     | Name             | Description               |
|------------------|------------------|---------------------------|
| `MPIOpenMP`      | `mpi_openmp`     | CPU sequential/OpenMP     |
| `Cuda`           | `cuda`           | NVIDIA GPUs via CUDA      |
| `Hip`            | `hip`            | AMD GPUs via HIP          |
| `Sycl`           | `sycl`           | Intel/AMD/NVIDIA via SYCL |
| `OpenMPOffload`  | `openmp_offload` | GPU via OpenMP target     |
| `F2CCuda`        | `f2c_cuda`       | Fortran-to-C CUDA         |
| `F2CHip`         | `f2c_hip`        | Fortran-to-C HIP          |
| `F2CSycl`        | `f2c_sycl`       | Fortran-to-C SYCL         |

#### Scheme (`scheme.py`)

Orchestrates code generation for a language/target combination:

```python
class Scheme(Findable):
    lang: Lang                   # Language (C++, Fortran)
    target: Target               # Target backend
    loop_host_template: Path     # Template for loop host code
    
    def genLoopHost(...) -> Tuple[str, str, str]:
        """Generate loop host code from template"""
        # 1. Translate kernel if needed
        # 2. Process kernel text (KernelProcess)
        # 3. Render Jinja2 template
        return (generated_code, extension, kernel_func)
```

#### KernelProcess (`util.py`)

Handles kernel text transformations for different backends:

```python
class KernelProcess:
    def clean_kernel_func_text(kernel_func)     # Remove OPS-specific markers
    def cuda_complex_numbers(kernel_func)       # Handle complex number support
    def sycl_kernel_func_text(kernel_func, consts)  # SYCL-specific transforms
    def get_kernel_body_and_arg_list(kernel_func)   # Extract body and args
```

#### Parser (`cpp/parser.py`)

Uses libclang to parse C++ source files:

```python
def parseLoops(translation_unit, program) -> None:
    """Parse ops_par_loop calls from C++ source"""
    # Find macro instantiations and function calls
    # Extract loop information (kernel, block, range, arguments)
```

### Jinja2 Templates

Templates use Jinja2 syntax with OPS-specific filters and tests. Key template variables:

| Variable           | Description                                     |
|--------------------|-------------------------------------------------|
| `lh`               | Loop host object (kernel name, args, ndim, etc.)|
| `kernel_func`      | Original kernel function text                   |
| `kernel_body`      | Extracted kernel body                           |
| `args_list`        | Argument name list                              |
| `target`           | Current target object                           |
| `consts_in_kernel` | Constants used in kernel                        |

Example template structure (`loop_host.cpp.j2`):
```jinja2
{% block host_prologue %}
    // Setup code: args, dimensions, pointers
{% endblock %}

{% block kernel_call %}
    // Parallel launch code (varies by target)
{% endblock %}

{% block host_epilogue %}
    // Cleanup, timing, diagnostics
{% endblock %}
```

### Adding a New Backend

To add a new backend (e.g., "newgpu"):

1. **Define Target** in `target.py`:
```python
class NewGPU(Target):
    name = "newgpu"
    kernel_translation = True
    config = {"grouped": True, "device": 11}

Target.register(NewGPU)
```

2. **Create Scheme** in `cpp/schemes.py`:
```python
class CppNewGPU(CppScheme):
    target = NewGPU()
    loop_host_template = Path("cpp/newgpu/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/newgpu/master_kernel.cpp.j2")
    loop_kernel_extension = "newgpu.cpp"

Scheme.register(CppNewGPU)
```

3. **Create Templates** in `resources/templates/cpp/newgpu/`:
   - `loop_host.cpp.j2` - Loop host wrapper
   - `master_kernel.cpp.j2` - Master include file

4. **Add Runtime Support** in `ops/c/src/newgpu/` (if needed)

5. **Update Makefiles** in `makefiles/` directory

---

## Runtime Library (ops/c/)

The runtime library provides backend implementations for data management, parallel execution, and communication.

### Directory Structure

```
ops/c/
├── include/                 # Public headers
│   ├── ops_lib_core.h       # Core OPS API
│   ├── ops_seq.h            # Sequential backend header
│   ├── ops_cuda.h           # CUDA backend header
│   ├── ops_hip.h            # HIP backend header
│   ├── ops_sycl.h           # SYCL backend header
│   └── ...
├── src/                     # Implementation
│   ├── core/                # Core library (shared across backends)
│   │   ├── ops_lib_core.cpp # Core API implementation
│   │   ├── ops_lazy.cpp     # Lazy execution & tiling
│   │   └── ops_instance.cpp # OPS instance management
│   ├── sequential/          # Sequential backend
│   ├── cuda/                # CUDA backend
│   ├── hip/                 # HIP backend
│   ├── sycl/                # SYCL backend
│   ├── mpi/                 # MPI support for all backends
│   │   ├── ops_mpi_core.cpp
│   │   ├── ops_mpi_partition.cpp  # Domain decomposition
│   │   ├── ops_mpi_rt_support_cuda.cpp
│   │   ├── ops_mpi_rt_support_sycl.cpp
│   │   └── ...
│   ├── ompoffload/          # OpenMP offload backend
│   └── tridiag/             # Tridiagonal solver support
└── lib/                     # Compiled libraries
```

### Core Components

#### ops_lib_core.cpp
- `ops_init()` / `ops_exit()` - Initialization and cleanup
- `ops_decl_block()` - Block declaration
- `ops_decl_dat()` - Dataset declaration
- `ops_decl_stencil()` - Stencil declaration
- `ops_partition()` - MPI partitioning trigger

#### ops_lazy.cpp
- Lazy execution queue management
- Tiling plan computation
- Communication-avoiding optimizations
- Key structures: `ops_kernel_list`, `tiling_plan`

#### MPI Support (ops/c/src/mpi/)
- Domain decomposition (`ops_mpi_partition.cpp`)
- Halo exchange management
- Backend-specific MPI+GPU support:
  - `ops_mpi_rt_support_cuda.cpp` - CUDA+MPI
  - `ops_mpi_rt_support_sycl.cpp` - SYCL+MPI
  - `ops_mpi_rt_support_hip.cpp` - HIP+MPI

### Adding Runtime Support for a New Backend

1. **Create backend directory**: `ops/c/src/newgpu/`
2. **Implement required functions**:
   - Device memory allocation/deallocation
   - Data transfer (host ↔ device)
   - Kernel launch wrappers
3. **Add MPI support** (if needed): `ops/c/src/mpi/ops_mpi_rt_support_newgpu.cpp`
4. **Update build system**:
   - Add `makefiles/Makefile.newgpu`
   - Update `CMakeLists.txt`

---

## Build System

### Makefile System

The makefile-based build uses modular includes:

```
makefiles/
├── Makefile.common          # Common flags and definitions
├── Makefile.c_app           # Main C application makefile
├── Makefile.cuda            # CUDA-specific flags
├── Makefile.hip             # HIP-specific flags
├── Makefile.sycl            # SYCL flags (via Makefile.icx)
├── Makefile.mpi             # MPI flags
└── Makefile.<compiler>      # Compiler-specific settings
```

### Build Targets

For an application named `APP`, the following targets are generated:

| Target               | Description                         |
|----------------------|-------------------------------------|
| `$(APP)_dev_seq`     | Development sequential (no code-gen)|
| `$(APP)_dev_mpi`     | Development MPI (no code-gen)       |
| `$(APP)_seq`         | Sequential with generated kernels   |
| `$(APP)_openmp`      | OpenMP parallel                     |
| `$(APP)_mpi`         | MPI distributed                     |
| `$(APP)_mpi_openmp`  | MPI + OpenMP hybrid                 |
| `$(APP)_tiled`       | Lazy execution with tiling          |
| `$(APP)_cuda`        | CUDA single GPU                     |
| `$(APP)_mpi_cuda`    | MPI + CUDA                          |
| `$(APP)_sycl`        | SYCL single device                  |
| `$(APP)_mpi_sycl`    | MPI + SYCL                          |
| `$(APP)_hip`         | HIP single GPU                      |
| `$(APP)_mpi_hip`     | MPI + HIP                           |
| `$(APP)_ompoffload`  | OpenMP Offload single GPU           |
| `$(APP)_mpi_ompoffload` | MPI + OpenMP Offload             |

---

## Debugging Tips

### Code Generator Debugging

```bash
# Verbose output
python3 ops-translator -v --file_paths source.cpp

# Dump parsed structure as JSON
python3 ops-translator -d --file_paths source.cpp

# Target specific backend only
python3 ops-translator -t cuda --file_paths source.cpp
```

### Runtime Debugging

```bash
# Enable diagnostics
./app_cuda -OPS_DIAGS=2

# Check block decomposition (MPI)
./app_mpi_cuda -OPS_DIAGS=2

# Timing breakdown
ops_timing_output(stdout);
```

### Common Issues

| Issue                              | Cause                              | Solution                       |
|------------------------------------|------------------------------------| ------------------------------ |
| `GET_MACRO` redefined              | Name collision with Intel headers  | Harmless warning, ignore       |
| `printf` in SYCL kernel            | Variadic functions not allowed     | Guard with `#ifndef OPS_SYCL`  |
| Preprocessor directives stripped   | Code generator limitation          | Use runtime conditionals       |

---

## Contributing

To contribute to OPS, please use the following steps:
1. Clone the [OPS](https://github.com/OP-DSL/OPS) repository on your local system.
2. Create a new branch in your cloned repository.
3. Make changes or contributions in your new branch.
4. Submit your changes by creating a pull request to the `develop` branch of the OPS repository.

Contributions in the `develop` branch will be merged into the `master` branch when a new release is created.

