# OPS Code Generation V2

### Requirements

- Python >= 3.8
- All Python package dependencies listed in `requirements.txt` (installed automatically — see below)

### Setting up the Python environment

#### Makefile build

The virtual environment is managed by the provided `Makefile` and is created under `ops_translator/.python/`.
To build or rebuild it manually:

```bash
# From the ops_translator/ directory:
make python        # creates .python/ venv and installs all dependencies
make clean         # removes the .python/ venv (use before make python to force a clean rebuild)
```

The Makefile uses whatever `python3` is on your `PATH`. On HPC systems this is typically loaded via a module, e.g.:

```bash
module load python/3.9.7
```

> **Note for HPC systems:** Some module-provided Python builds lack SSL support (e.g. compiled against an older `libssl.so.1.1` no longer present on the system). The Makefile automatically detects this and falls back to `/usr/bin/python3` to create the venv, which typically has working SSL.

#### CMake build

When building OPS with CMake, the translator is **not** set up via `make python` and does not use `.python/`. Instead, during the CMake configure step the entire `ops_translator/` tree is copied to `${CMAKE_INSTALL_PREFIX}/translator/ops_translator/` and `setup_venv_cmake.sh` is run automatically to create a venv at:

```
${CMAKE_INSTALL_PREFIX}/translator/ops_translator/ops_venv/
```

No manual step is required; the configure step (`cmake -DCMAKE_INSTALL_PREFIX=<prefix> ..`) handles everything. If the venv already exists at that path it is left untouched.

### How the translator is invoked

In normal use the translator is called automatically by each application's Makefile — no manual step is required. For reference, the application Makefile invokes `translator_setup.sh`, which calls the venv Python directly (no `source activate` needed):

```bash
./translator_setup.sh [ops_translator options] --file_paths <source_files>
```


