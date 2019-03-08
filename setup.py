from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('transition_functions_cy.pyx'))

# To compile use: python setup.py build_ext --inplace
# To generate html use: cython -a transition_functions_cy.pyx