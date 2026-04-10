CC=gcc
CXX=g++

undefine CFLAGS
undefine CXXFLAGS

PWD != dirname $(realpath $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

PYTHON_VER := cpython-3.11.15+20260325-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz 
PYTHON_DIR := $(PWD)/.python

.PHONY: python
python: $(PYTHON_DIR)/.python-packages

$(PYTHON_DIR)/.python: python_source/$(PYTHON_VER)
	mkdir -p $(PYTHON_DIR)
	tar xf $^ --strip-components=1 --directory $(PYTHON_DIR)
	touch $@

$(PYTHON_DIR)/.python-packages: $(PYTHON_DIR)/.python
	$(PYTHON_DIR)/bin/python3 -m ensurepip
	$(PYTHON_DIR)/bin/pip3 install --no-index --find-link=python_source/wheels -r requirements.txt
	$(PYTHON_DIR)/bin/pip3 install --no-index --find-link=python_source/wheels --force-reinstall libclang==16.0.6
	touch $@

.PHONY: python-packages-download
python-packages-download:
	$(PYTHON_DIR)/bin/pip3 download --destination-directory=python_source/wheels -r requirements.txt
	$(PYTHON_DIR)/bin/pip3 install --no-index --find-link=python_source/wheels --force-reinstall libclang==16.0.6

clean:
	$(RM) -rf $(PYTHON_DIR)
