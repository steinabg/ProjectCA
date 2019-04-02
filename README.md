# Project CA
(**Under development!**)\
This repository contains source files of a cellular automaton used for simulating turbidity currents.

## Getting Started

### Single core

1. Run setup.py to compile the c++ code used for the CA transition functions.
    - To compile use: python setup.py build_ext --inplace
    - To generate cython HTML document use: cython -a transition_functions_cy.pyx
2. Initial conditions are modified in the .ini files in the Config folder.
3. Specify which .ini file to use by modifying the arguement in line 20 of GUI.py.
4. Run GUI.py



### Using MPI
1. Run setup.py to compile the c++ code used for the CA transition functions.
    - To compile use: python setup.py build_ext --inplace
    - To generate cython HTML document use: cython -a transition_functions_cy.pyx
2. Initial conditions are modified in the .ini files in the Config folder.
3. Specify which .ini file to use by modifying the argument of CAenv.import_parameters()
in line 442 of mpi_halo_exchange.py
4. Run mpi_halo_exchange.py by calling mpirun, e.g.
```
mpirun -n 4 python mpi_halo_exchange.py
```
Currently, the number of cores should be a square number, as the grid of cells
is divided equally in the x and y direction. 

### General
* Make sure to create the following folder hierachy in your working
 directory, or you will get an error:
```
./Data
./Data/mpi_combined_png
./Data/mpi_combined_txt
```
* For debugging purposes switching from the c++ compiled code to regular python may be 
done in the import section of hexgrid.py by switching

```
import transition_functions_cy as tra # This is the cython version
```
to 
```
import transition_functions as tra # This is the python version
```

### Specifying a bathymetry
The bathymetry used by the simulation can be specified through modifying the 'terrain' parameter 
in the .ini files. Terrain can either be a keyword (string type) or a numpy.ndarray((Ny,Nx)).

Valid keywords are:
* 'river'
* 'river_shallow'
* 'rupert'
* 'sloped_plane'



### Prerequisites

At this point some python packages may be used strictly for debugging
 purposes, and may later be removed. \
 \
In general all one should need is numpy, cython, matplotlib, mpi,
mpi4py and a c++ compiler. The GUI for the single core version 
depends on PyQT.

...
