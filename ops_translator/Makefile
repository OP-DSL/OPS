CC=gcc
CXX=g++

undefine CFLAGS
undefine CXXFLAGS

PWD != dirname $(realpath $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

PYTHON_DIR := $(PWD)/.python

# Pick a python3 that has SSL support (needed for pip to reach PyPI).
# The module-loaded python3 may lack libssl.so.1.1 on newer OS; fall back to /usr/bin/python3.
PYTHON3 != python3 -c "import ssl" 2>/dev/null && echo python3 || echo /usr/bin/python3

.PHONY: python clean

# Create venv from system python3 and install all required packages.
# Requires a python3 with SSL on PATH or at /usr/bin/python3.
python: $(PYTHON_DIR)/.python-packages

$(PYTHON_DIR)/.python-packages: $(PWD)/requirements.txt $(PWD)/Makefile
	@echo "Using Python: $(shell $(PYTHON3) --version 2>&1)"
	$(PYTHON3) -m venv $(PYTHON_DIR)
	$(PYTHON_DIR)/bin/pip3 install --upgrade pip
	$(PYTHON_DIR)/bin/pip3 install -r $(PWD)/requirements.txt
	$(PYTHON_DIR)/bin/pip3 install --force-reinstall libclang==16.0.6
	touch $@

clean:
	$(RM) -rf $(PYTHON_DIR)
