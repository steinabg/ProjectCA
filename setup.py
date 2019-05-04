from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.annotate = True



compiler_directives={
                     'language_level': 3}

setup(
      ext_modules = cythonize('transition_functions_cy.pyx', compiler_directives=compiler_directives),
      include_dirs = [np.get_include()],
      requires=['matplotlib', 'xarray', 'scipy', 'numpy']
)
# setup(ext_modules = cythonize('kladd_cy.pyx'),include_dirs = [np.get_include()])

# To compile use: python setup.py build_ext --inplace
# To generate html use: cython -a transition_functions_cy.pyx