#!/bin/bash

#Do not run below command if virtualenv is already installed
#python3 -m pip install --user virtualenv

mkdir -p $OPS_INSTALL_PATH/../ops_translator/ops_venv

python3 -m venv $OPS_INSTALL_PATH/../ops_translator/ops_venv

source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate

python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt

python3 -m pip install --force-reinstall libclang==16.0.6
