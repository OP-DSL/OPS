#!/usr/bin/env python

from distutils.core import setup

setup(name='ops',
      version='dev',
      description='OPS is an API with associated libraries and preprocessors to generate parallel executables for applications on mulit-block structured meshes.',
      author='Gihan Mudalige, Istvan Reguly, Mike Giles, and others',
      url='https://op-dsl.github.io/',
      packages=['ops_translator', 'ops_translator.c', 'ops_translator.fortran'],
      scripts=[],
      classifiers=[
      'Development Status :: 3 - Alpha',
      'Environment :: Console',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 2.7',
      'Topic :: Software Development :: Code Generators',
      ]
      )
