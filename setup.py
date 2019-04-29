from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize('transition_functions_cy.pyx'), include_dirs = [np.get_include()],
      requires=['matplotlib', 'xarray', 'scipy', 'numpy'])
# setup(ext_modules = cythonize('kladd_cy.pyx'),include_dirs = [np.get_include()])

# To compile use: python setup.py build_ext --inplace
# To generate html use: cython -a transition_functions_cy.pyx