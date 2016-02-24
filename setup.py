#!/usr/bin/env python

from distutils.core import setup

setup(name='ops',
      version='dev',
      description='OPS is an API with associated libraries and preprocessors to generate parallel executables for applications on mulit-block structured meshes.',
      author='Mike Giles, Istvan Reguly, Gihan Mudalige, and others',
      url='http://www.oerc.ox.ac.uk/projects/ops',
      packages=['ops_translator', 'ops_translator.c'],
      package_dir={'ops_translator': 'translator/python'},
      scripts=[],
      classifiers=[
      'Development Status :: 3 - Alpha',
      'Environment :: Console',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 2.7',
      'Topic :: Software Development :: Code Generators',
      ]
      )
