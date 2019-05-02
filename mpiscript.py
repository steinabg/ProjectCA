from mpifunctions import *
from timeit import default_timer as timer

ma.ensure_dir('./Bathymetry/')
ma.ensure_dir('./Data/')
ma.ensure_dir('./Config/')
ma.ensure_file('./Config/configs.txt')
configs = CAenv.read_which_configs()
for config in configs:
    save_dt = []
    start = timer()
    mpienv = mpi_environment(config)

    mpienv.run(compare=False)
    if mpienv.my_rank == 0:
        print('{0} is complete. Time elapsed = {1}'.format(config, timer() - start))
