from mpifunctions import *


ma.ensure_dir('./Bathymetry/')
ma.ensure_dir('./Data/')
ma.ensure_dir('./Config/')
ma.ensure_file('./Config/configs.txt')
configs = CAenv.read_which_configs()
for config in configs:

    mpienv = mpi_environment(config)

    mpienv.run(compare=False)
