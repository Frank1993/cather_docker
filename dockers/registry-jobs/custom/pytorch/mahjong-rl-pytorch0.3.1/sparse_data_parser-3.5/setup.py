from distutils.core import setup, Extension
import numpy as np
import platform 
import os
import sys
from distutils.ccompiler import get_default_compiler

compile_args = ['-std=c++11', '-O3']

print("os %s, compile args %s" %(platform.platform(), compile_args))

ext = Extension('_sparse_data_parser',
                include_dirs = [np.get_include(),'.'],
                extra_compile_args=compile_args,
                sources = ['_sparse_data_parser.cpp'])

setup (name = 'sparse data parser',
       version = '0.1',
       description = 'sparse data parser',
       ext_modules = [ext],
       py_modules = ['sparse_data_parser'])