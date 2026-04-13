CC=gcc
CXX=g++

undefine CFLAGS
undefine CXXFLAGS

PWD != dirname $(realpath $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

PYTHON_VER := cpython-3.11.15+20260325-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz 
PYTHON_DIR := $(PWD)/.python

.PHONY: python packages-download packages clean

# Top-level target
python: $(PYTHON_DIR)/.python-packages

packages-download: $(PYTHON_DIR)/.python-packages-download

packages: $(PYTHON_DIR)/.python-packages

# -------------------------
# Step 1: Setup Python
# -------------------------
$(PYTHON_DIR)/.python: python_source/$(PYTHON_VER)
	mkdir -p $(PYTHON_DIR)
	tar xf $^ --strip-components=1 --directory $(PYTHON_DIR)
	touch $@

# -------------------------
# Step 2: Download wheels
# -------------------------
$(PYTHON_DIR)/.python-packages-download: $(PYTHON_DIR)/.python
	$(PYTHON_DIR)/bin/python3 -m ensurepip
	$(PYTHON_DIR)/bin/pip3 download --destination-directory=python_source/wheels -r requirements.txt
	touch $@

# -------------------------
# Step 3: Install packages
# -------------------------
$(PYTHON_DIR)/.python-packages: $(PYTHON_DIR)/.python-packages-download
	$(PYTHON_DIR)/bin/python3 -m ensurepip
	$(PYTHON_DIR)/bin/pip3 install --no-index --find-links=python_source/wheels -r requirements.txt
	$(PYTHON_DIR)/bin/pip3 install --no-index --find-links=python_source/wheels --force-reinstall libclang==16.0.6
	touch $@

# -------------------------
# Cleanup
# -------------------------
clean:
	$(RM) -rf $(PYTHON_DIR)
