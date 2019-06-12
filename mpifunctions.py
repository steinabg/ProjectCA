from mpi4py import MPI
import numpy as np
import sys
import CAenvironment as CAenv
import scipy.io
import mathfunk as ma
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class mpi_environment:
    def __init__(self, config):
        # Setup MPI
        self.comm = comm = MPI.COMM_WORLD
        self.my_rank = comm.Get_rank()
        self.num_procs = comm.Get_size()
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.NW = 4
        self.NE = 5
        self.SE = 6
        self.SW = 7

        # Load CA parameters from file
        self.config = config
        self.g_params = CAenv.import_parameters(config)

        # N E is "converted" to x y during initialization of result_grid !
        self.result_grid = CAenv.CAenvironment(self.g_params, mpi=True)

        self.IMG_X = self.g_params['nx']
        self.IMG_Y = self.g_params['ny']
        # print("my rank = ", self.my_rank)

        self.nb_procs = [0, 0, 0, 0, 0, 0, 0, 0]

        self.p_y_dims, self.p_x_dims = define_px_py_dims(self.num_procs, self.IMG_Y, self.IMG_X)

        cartesian_communicator = comm.Create_cart((self.p_y_dims, self.p_x_dims), periods=(False, False))

        self.my_mpi_row, self.my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)

        self.nb_procs[self.UP], self.nb_procs[self.DOWN] = cartesian_communicator.Shift(0, 1)
        self.nb_procs[self.LEFT], self.nb_procs[self.RIGHT] = cartesian_communicator.Shift(1, 1)


        self.p_local_grid_x_dim = define_local_hexgrid_size(self.IMG_X, self.p_x_dims, self.my_mpi_col)
        self.p_local_grid_y_dim = define_local_hexgrid_size(self.IMG_Y, self.p_y_dims, self.my_mpi_row)
        self.procs_map = np.zeros((self.p_y_dims, self.p_x_dims), dtype='i', order='C') # Complete map of mpi proc grid
        r = 0
        for row in range(self.p_y_dims):
            for col in range(self.p_x_dims):
                self.procs_map[row, col] = r
                r += 1


        self.define_mpi_diagonals()
        if self.my_rank == 0:
            self.local_dims = []
            r = 0
            for row in range(self.p_y_dims):
                for col in range(self.p_x_dims):
                    self.local_dims.append([])
                    self.local_dims[r].append(define_local_hexgrid_size(self.IMG_Y, self.p_y_dims, row))
                    self.local_dims[r].append(define_local_hexgrid_size(self.IMG_X, self.p_x_dims, col))
                    r += 1

        # Setup stuff for Cellular automaton





        self.ITERATIONS = self.g_params['num_iterations']

        # Setup parameters for local CAenvironment
        self.l_params = self.g_params.copy()
        self.l_params['x'] = None  # if None == no source
        self.l_params['y'] = None  # if None == no source
        self.l_params['nx'] = self.p_local_grid_x_dim + 2  # make room for borders
        self.l_params['ny'] = self.p_local_grid_y_dim + 2  # make room for borders
        self.l_params['rank'] = self.my_rank
        # If N E is provided: global X Y will be added to g_params during initialization of result_grid
        # print("rank {0}, g_params[yx] = {1} ".format(self.my_rank, [self.g_params['y'], self.g_params['x']]))

        self.set_local_grid_source_xy()  # Here local X Y is added to l_params
        # Delete coordinates, so x and y are not overwritten when initilizing local grid
        try:
            del self.l_params['n']
            del self.l_params['e']
        except KeyError:
            pass
        # print("rank {0}, l_params[yx] = {1} ".format(self.my_rank, [self.l_params['y'], self.l_params['x']]))
        self.local_bathy = self.generate_p_local_hex_bathymetry()
        self.l_params['bathymetry'] = self.local_bathy.copy()
        self.p_local_hexgrid = CAenv.CAenvironment(self.l_params, global_grid=False, mpi=True)
        self.set_local_grid_bc()

        self.local_grid_wb = np.zeros((self.p_local_grid_y_dim + 2), dtype=np.double, order='C')
        self.local_grid_eb = np.zeros((self.p_local_grid_y_dim + 2), dtype=np.double, order='C')
        self.local_grid_ev = np.zeros((self.p_local_grid_y_dim + 2), dtype=np.double, order='C')
        self.local_grid_wv = np.zeros((self.p_local_grid_y_dim + 2), dtype=np.double, order='C')

        self.border_row_t = MPI.DOUBLE.Create_vector(self.p_local_grid_x_dim + 2, 1, 1)
        self.border_row_t.Commit()

        self.border_col_t = MPI.DOUBLE.Create_vector(self.p_local_grid_y_dim + 2, 1, self.p_local_grid_x_dim + 2)
        self.border_col_t.Commit()
        self.save_dir = './Data/MPI/' + config + '/'
        self.save_path_txt = self.save_dir + 'binaries/'
        self.save_path_png = self.save_dir
        self.p_local_hexgrid.parameters['save_dir'] = self.save_path_png

        if self.my_rank == 0:
            ma.ensure_dir(self.save_path_txt)
            ma.ensure_dir(self.save_path_png)
            np.save(self.save_path_txt+'X000', self.result_grid.X[:, :, 0])
            np.save(self.save_path_txt+'X001', self.result_grid.X[:, :, 1])
            self.save_dt = [] # Save time steps
            self.mass = [] # Save mass of TC
            self.massBed = [] # mass of sea bed
            self.density = [] # Max density of TC
            self.beddensity = [] # Max density of bed
            self.head_velocity = []
            self.j_values = [] # Save iteration number at samples
        self.bottom_indices = []
        for index in self.result_grid.bot_indices:
            x, y = global_coords_to_local_coords(index[0], index[1], self.my_mpi_row, self.my_mpi_col,
                                              self.p_local_grid_x_dim, self.p_local_grid_y_dim)
            # if self.my_rank == 0: print("index = {0}, in my grid = {1}".format(index, i))
            self.bottom_indices.append(tuple([y, x]))
        self.bottom_indices = [x for x in self.bottom_indices if x != tuple([-1, -1])]

        self.sample_rate = self.l_params['sample_rate']
        self.i_sample_values = []
        self.plot_bool = CAenv.set_which_plots(self.l_params)



        # Mpi fix?
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
        self.p_local_hexgrid.seaBedDiff = self.p_local_hexgrid.calc_bathymetryDiff()

    def define_mpi_diagonals(self):
        """
        This function defines diagonal neighbors in the mpi cartesian topology.
        """
        nb_procs = self.nb_procs
        mpirow = self.my_mpi_row
        mpicol = self.my_mpi_col
        p_y_dims = self.p_y_dims
        p_x_dims = self.p_x_dims
        procs_map = self.procs_map
        nb_index = [[-1, -1], [-1, 1], [1, 1], [1, -1]]  # NE, NW, SE, SW
        for i in range(4):
            nb_row = mpirow + nb_index[i][0]
            nb_col = mpicol + nb_index[i][1]
            if (nb_row >= 0) and (nb_row < p_y_dims) and (nb_col >= 0) and (nb_col < p_x_dims):
                nb_procs[i + 4] = procs_map[nb_row, nb_col]
            else:
                nb_procs[i + 4] = -1

    def set_local_grid_source_xy(self):
        """
        This function will identify if source cells are within the local grid\
        and if so, find their local indices.
        :return:
        """
        p_local_source_tiles = []
        p_local_source_tile_x = []
        p_local_source_tile_y = []
        x_coords_array = self.g_params['x']
        y_coords_array = self.g_params['y']
        my_mpi_row = self.my_mpi_row
        my_mpi_col = self.my_mpi_col
        p_local_grid_x_dim = self.p_local_grid_x_dim
        p_local_grid_y_dim = self.p_local_grid_y_dim
        no_x_coords = np.size(x_coords_array)
        no_y_coords = np.size(y_coords_array)

        for i in range(no_x_coords):
            for j in range(no_y_coords):
                if (no_x_coords == 1):
                    if (no_y_coords == 1):
                        # print("h1")
                        # print(x_coords_array,y_coords_array)
                        local_x, local_y = global_coords_to_local_coords(y_coords_array, x_coords_array, my_mpi_row,
                                                                         my_mpi_col, p_local_grid_x_dim,
                                                                         p_local_grid_y_dim)
                        if (local_x > -1):
                            p_local_source_tiles.append(tuple([local_y, local_x]))
                            p_local_source_tile_x.append(local_x)
                            p_local_source_tile_y.append(local_y)


                    else:
                        # print("h2")
                        # print(x_coords_array,y_coords_array[0][j-1])
                        local_x, local_y = global_coords_to_local_coords(y_coords_array[0][j - 1], x_coords_array,
                                                                         my_mpi_row, my_mpi_col,
                                                                         p_local_grid_x_dim, p_local_grid_y_dim)
                        if (local_x > -1):
                            p_local_source_tiles.append(tuple([local_y, local_x]))
                            p_local_source_tile_x.append(local_x)
                            p_local_source_tile_y.append(local_y)
                else:
                    if (no_y_coords == 1):
                        # print("h3")
                        # print(x_coords_array[0][i-1],y_coords_array)
                        local_x, local_y = global_coords_to_local_coords(y_coords_array, x_coords_array[0][i - 1],
                                                                         my_mpi_row, my_mpi_col,
                                                                         p_local_grid_x_dim, p_local_grid_y_dim)
                        if (local_x > -1):
                            p_local_source_tiles.append(tuple([local_y, local_x]))
                            p_local_source_tile_x.append(local_x)
                            p_local_source_tile_y.append(local_y)
                    else:
                        # print("h4")
                        # print(x_coords_array[0][i-1],y_coords_array[0][j-1])
                        local_x, local_y = global_coords_to_local_coords(y_coords_array[0][j - 1],
                                                                         x_coords_array[0][i - 1],
                                                                         my_mpi_row, my_mpi_col,
                                                                         p_local_grid_x_dim, p_local_grid_y_dim)
                        if (local_x > -1):
                            p_local_source_tiles.append(tuple([local_y, local_x]))
                            p_local_source_tile_x.append(local_x)
                            p_local_source_tile_y.append(local_y)

        # print(p_local_source_tiles)
        p_local_source_tiles = [x for x in p_local_source_tiles if x != tuple([-1, -1])]
        if len(p_local_source_tiles) > 0:
            self.l_params['x'] = np.ix_(
                np.arange(np.min(p_local_source_tile_x), np.max(p_local_source_tile_x) + 1))
            self.l_params['y'] = np.ix_(
                np.arange(np.min(p_local_source_tile_y), np.max(p_local_source_tile_y) + 1))
            # print("p_local_grid_parameters['x'] = {0}, p_local_grid_parameters['y'] = {1}".format(
            #     p_local_grid_parameters['x'], p_local_grid_parameters['y']
            # ))
        # print(p_local_grid_parameters['x'])
        # print(p_local_grid_parameters['y'])

    def generate_p_local_hex_bathymetry(self):
        my_row = self.my_mpi_row
        my_col = self.my_mpi_col
        global_bathy = self.result_grid.bathymetry

        local_bathy = global_bathy[(my_row * self.p_local_grid_y_dim):((my_row + 1) * self.p_local_grid_y_dim),
                      my_col * self.p_local_grid_x_dim:((my_col + 1) * self.p_local_grid_x_dim)]
        return local_bathy

    def set_local_grid_bc(self, type='absorb'):
        """ Sets edge cells in appropriate mpi ranks to boundary values """
        my_mpi_col = self.my_mpi_col
        my_mpi_row = self.my_mpi_row

        if type == 'absorb':

            if my_mpi_col == 0:
                self.p_local_hexgrid.Q_d[:,0:2] = np.inf
                self.p_local_hexgrid.Q_d[1:-1,0:2] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[:,0:2] = np.inf
                self.p_local_hexgrid.Q_a[1:-1,0:2] = self.local_bathy[:,0:2] + self.l_params['q_d[interior]']
                if np.any(self.p_local_hexgrid.Q_th[:, 0] > 1e-5):
                    self.p_local_hexgrid.Q_th[:, 0:3] = 0
                    self.p_local_hexgrid.Q_cj[:, 0:3,:] = 0
                    self.p_local_hexgrid.Q_v[:, 0:3] = 0
                    self.p_local_hexgrid.Q_o[:, 0:3,:] = 0
            if my_mpi_row == 0:
                self.p_local_hexgrid.Q_d[0:2,:] = np.inf
                self.p_local_hexgrid.Q_d[0:2,1:-1] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[0:2,:] = np.inf
                self.p_local_hexgrid.Q_a[0:2,1:-1] = self.local_bathy[0:2, :] + self.l_params['q_d[interior]']
                if np.any( self.p_local_hexgrid.Q_th[0,:] > 1e-5):
                    self.p_local_hexgrid.Q_th[0:3,:] = 0
                    self.p_local_hexgrid.Q_cj[0:3,:,:] = 0
                    self.p_local_hexgrid.Q_v[0:3,:] = 0
                    self.p_local_hexgrid.Q_o[0:3,:,:] = 0
            if my_mpi_row == (self.p_y_dims-1):
                # print("mympirow == p_ydims. myrank = ", my_rank)
                self.p_local_hexgrid.Q_d[-2:,:] = np.inf
                self.p_local_hexgrid.Q_d[-2:,1:-1] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[-2:,:] = np.inf
                self.p_local_hexgrid.Q_a[-2:,1:-1] = self.local_bathy[-2:,:] + self.l_params['q_d[interior]']
                if np.any(self.p_local_hexgrid.Q_th[-1,:] > 1e-5):
                    self.p_local_hexgrid.Q_th[-3:,1:-1] = 0
                    self.p_local_hexgrid.Q_cj[-3:,1:-1,:] = 0
                    self.p_local_hexgrid.Q_v[-3:,1:-1] = 0
                    self.p_local_hexgrid.Q_o[-3:,1:-1,:] = 0
            if my_mpi_col == (self.p_x_dims-1):
                # print("mympicol == p_xdims. myrank = ", my_rank)
                self.p_local_hexgrid.Q_d[:,-2:] = np.inf
                self.p_local_hexgrid.Q_d[1:-1,-2:] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[:,-2:] = np.inf
                self.p_local_hexgrid.Q_a[1:-1,-2:] = self.local_bathy[:,-2:] + self.l_params['q_d[interior]']
                if np.any(self.p_local_hexgrid.Q_th[:,-1] > 1e-5):
                    self.p_local_hexgrid.Q_th[:,-3:] = 0
                    self.p_local_hexgrid.Q_cj[:,-3:,:] = 0
                    self.p_local_hexgrid.Q_v[:,-3:] = 0
                    self.p_local_hexgrid.Q_o[:,-3:,:] = 0
        elif type == 'barrier':
            if my_mpi_col == 0:
                self.p_local_hexgrid.Q_d[:,0] = np.inf
                # self.p_local_hexgrid.Q_d[1:-1,0] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[:,0] = np.inf
                # self.p_local_hexgrid.Q_a[1:-1,0] = self.local_bathy[:,0] + self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_th[:, 0] = 0
                self.p_local_hexgrid.Q_cj[:, 0] = 0
            if my_mpi_row == 0:
                self.p_local_hexgrid.Q_d[0,:] = np.inf
                # self.p_local_hexgrid.Q_d[0,1:-1] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[0,:] = np.inf
                # self.p_local_hexgrid.Q_a[0,1:-1] = self.local_bathy[0, :] + self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_th[0,:] = 0
                self.p_local_hexgrid.Q_cj[0,:] = 0
            if my_mpi_row == (self.p_y_dims-1):
                # print("mympirow == p_ydims. myrank = ", my_rank)
                self.p_local_hexgrid.Q_d[-1,:] = np.inf
                # self.p_local_hexgrid.Q_d[-1,1:-1] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[-1,:] = np.inf
                # self.p_local_hexgrid.Q_a[-1,1:-1] = self.local_bathy[-1,:] + self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_th[-1,1:-1] = 0
                self.p_local_hexgrid.Q_cj[-1,1:-1] = 0
            if my_mpi_col == (self.p_x_dims-1):
                # print("mympicol == p_xdims. myrank = ", my_rank)
                self.p_local_hexgrid.Q_d[:,-1] = np.inf
                # self.p_local_hexgrid.Q_d[1:-1,-1] = self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_a[:,-1] = np.inf
                # self.p_local_hexgrid.Q_a[1:-1,-1] = self.local_bathy[:,-1] + self.l_params['q_d[interior]']
                self.p_local_hexgrid.Q_th[:,-1] = 0
                self.p_local_hexgrid.Q_cj[:,-1] = 0

    def DEBUG_print_mpitopology(self):
        print("my rank = {0}"
              "nb[N] = {1}"
              "nb[S] = {2}"
              "nb[E] = {3}"
              "nb[W] = {4}".format(self.my_rank, self.nb_procs[self.UP], self.nb_procs[self.DOWN],
                                   self.nb_procs[self.RIGHT], self.nb_procs[self.LEFT]))

    def iterateCA(self):

        self.p_local_hexgrid.T_1()
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_th)
        self.set_local_grid_bc()
        self.p_local_hexgrid.T_2()
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_d)
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        self.set_local_grid_bc()
        self.p_local_hexgrid.I_1()
        self.exchange_borders_cube(self.p_local_hexgrid.Q_o, 6)
        self.set_local_grid_bc()
        self.p_local_hexgrid.I_2()
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_th)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        # Try adding outflow calculation for "new cells"
        self.set_local_grid_bc()
        self.p_local_hexgrid.I_1()
        self.exchange_borders_cube(self.p_local_hexgrid.Q_o, 6)
        self.set_local_grid_bc()
        self.p_local_hexgrid.I_3()
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_v)
        self.set_local_grid_bc()
        self.p_local_hexgrid.I_4()
        # if my_rank != 0:
        #     print("0 my rank = ", my_rank, "\nnonzero Qd=\n", np.where(np.logical_and(p_local_hexgrid.Q_d > 1, np.isfinite(p_local_hexgrid.Q_d))))
        self.mpi_toppling_fix()
        # if my_rank != 0:
        #     print("1 my rank = ", my_rank, "\nnonzero Qd=\n",
        #           np.where(np.logical_and(p_local_hexgrid.Q_d > 1, np.isfinite(p_local_hexgrid.Q_d))))
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_d)
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
        self.set_local_grid_bc()

    def exchange_borders_cube(self, local_petri_A, z_dim):
        comm = self.comm
        neighbor_processes = self.nb_procs
        for i in range(z_dim):
            # Send data south and receive from north
            local_grid_send_south = local_petri_A[-2,:,i].copy()
            local_grid_receive_north = np.zeros(self.p_local_grid_x_dim+2, dtype=np.double, order='C')
            comm.Sendrecv(
                [local_grid_send_south, self.p_local_grid_x_dim + 2, MPI.DOUBLE],  # send the second last row
                neighbor_processes[self.DOWN],
                0,
                [local_grid_receive_north, self.p_local_grid_x_dim + 2, MPI.DOUBLE],  # recvbuf = first row
                neighbor_processes[self.UP],
                0
            )
            local_petri_A[0,:,i] = local_grid_receive_north.copy()

            # # Send data north and receive from south
            local_grid_send_north = local_petri_A[1,:,i].copy()
            local_grid_receive_south = np.zeros(self.p_local_grid_x_dim + 2, dtype=np.double, order='C')
            comm.Sendrecv(
                [local_grid_send_north, self.p_local_grid_x_dim + 2, MPI.DOUBLE],  # sendbuf = second row (1.row = border)
                neighbor_processes[self.UP],  # destination
                1,  # sendtag
                [local_grid_receive_south, self.p_local_grid_x_dim +2, MPI.DOUBLE],  # recvbuf = last row
                neighbor_processes[self.DOWN],  # source
                1
            )
            local_petri_A[-1,:,i] = local_grid_receive_south.copy()

            # Send west and receive from east
            local_grid_send_west = local_petri_A[:, 1, i].copy()
            local_grid_receive_east = np.zeros(self.p_local_grid_y_dim + 2, dtype=np.double, order='C')
            comm.Sendrecv(
                [local_grid_send_west, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
                neighbor_processes[self.LEFT],
                2,
                [local_grid_receive_east, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
                neighbor_processes[self.RIGHT],
                2
            )
            local_petri_A[:, -1, i] = local_grid_receive_east.copy()

            # Send east and receive from west
            local_grid_send_east = local_petri_A[:, -2, i].copy()
            local_grid_receive_west = np.zeros(self.p_local_grid_y_dim + 2, dtype=np.double, order='C')
            comm.Sendrecv(
                [local_grid_send_east, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
                neighbor_processes[self.RIGHT],
                0,
                [local_grid_receive_west, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
                neighbor_processes[self.LEFT],
                0
            )
            local_petri_A[:, 0, i] = local_grid_receive_west.copy()

    def exchange_borders_matrix(self, local_petri_A):
        comm = self.comm
        border_row_t = self.border_row_t
        broder_col_t = self.border_col_t
        nb_procs = self.nb_procs


        # Send data south and receive from north
        comm.Sendrecv(
            [local_petri_A[-2, :], 1, border_row_t],  # send the second last row
            nb_procs[self.DOWN],
            0,
            [local_petri_A[0, :], 1, border_row_t],  # recvbuf = first row
            nb_procs[self.UP],
            0
        )

        # # Send data north and receive from south
        comm.Sendrecv(
            [local_petri_A[1, :], 1, border_row_t],  # sendbuf = second row (1.row = border)
            nb_procs[self.UP],  # destination
            1,  # sendtag
            [local_petri_A[-1, :], 1, border_row_t],  # recvbuf = last row
            nb_procs[self.DOWN],  # source
            1
        )
        #
        # Send west and receive from east
        self.local_grid_ev[:] = local_petri_A[:, 1].copy()
        comm.Sendrecv(
            [self.local_grid_ev, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
            nb_procs[self.LEFT],
            2,
            [self.local_grid_wb, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
            nb_procs[self.RIGHT],
            2
        )
        local_petri_A[:, -1] = self.local_grid_wb.copy()

        # # Send east and receive from west
        self.local_grid_wv[:] = local_petri_A[:, -2].copy()
        # print("my value = ", local_grid_wv)
        # print("my destination = ", local_grid_eb)

        comm.Sendrecv(
            [self.local_grid_wv, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
            nb_procs[self.RIGHT],
            0,
            [self.local_grid_eb, self.p_local_grid_y_dim + 2, MPI.DOUBLE],
            nb_procs[self.LEFT],
            0
        )
        local_petri_A[:, 0] = self.local_grid_eb.copy()

    def mpi_toppling_fix(self):
        """ This function gets the influx of sediment due to toppling,\
            and updates the Q_d values of the upper and lower boundaries\
             of the local grids accordingly."""
        my_mpi_row = self.my_mpi_row
        my_mpi_col = self.my_mpi_col
        g = self.p_local_hexgrid
        # My outflux:
        t = g.t
        b = g.b
        l = g.l
        r = g.r
        borders = self.get_mpi_nb_flux(t, b, l, r)  # Get influx
        # print("rank = ", my_rank, " max(right) = ", np.max(borders[3][:,0]))
        nw, ne, se, sw = borders[4:]
        if my_mpi_row > 0:
            # print("rank ", my_rank, " fixing upper")
            # Update upper bound
            top = borders[0]
            for j in range(1, self.p_local_grid_x_dim + 1):
                if top[j, 0]:
                    nQ_d_top = g.Q_d[1, j] + top[j, 0]
                    # print("q_d_top = ", nQ_d_top)
                    for l in range(self.l_params['nj']):
                        old_cut_top = g.Q_cbj[1, j, l] * g.Q_d[1, j]
                        g.Q_cbj[1, j, l] = (old_cut_top + top[j, l + 1]) / nQ_d_top
                        # if g.Q_cbj[1, j, l] != 1:
                        #     print("Q_cbj =", g.Q_cbj[1,j,l])
                    # print("Q_d[0,j] = ", g.Q_d[0,j])
                    g.Q_d[1, j] = nQ_d_top
                    g.Q_a[1, j] += top[j,0]
                    # print("nQ_d[0,j] = ", g.Q_d[0, j])
            # Update nw corner:
            # print("here! ", nw, borders[4])
            if my_mpi_col > 0 and nw[0]:
                nQ_d_nw = g.Q_d[1, 1] + nw[0]
                for l in range(self.l_params['nj']):
                    old_cut_nw = g.Q_cbj[1, 1, l] * g.Q_d[1, 1]
                    g.Q_cbj[1, 1, l] = (old_cut_nw + nw[l+1]) / nQ_d_nw
            #         # if g.Q_cbj[1, j, l] != 1:
                    #     print("Q_cbj =", g.Q_cbj[1,j,l])
                    # print("Q_d[0,j] = ", g.Q_d[0,j])
                g.Q_d[1, 1] = nQ_d_nw
                g.Q_a[1, 1] += nw[0]
            # Update ne corner:
            if my_mpi_col < (self.p_x_dims-1) and ne[0]:
                nQ_d_ne = g.Q_d[1, -2] + ne[0]
                for l in range(self.l_params['nj']):
                    old_cut_ne = g.Q_cbj[1, -2, l] * g.Q_d[1, -2]
                    g.Q_cbj[1, -2, l] = (old_cut_ne + ne[l+1]) / nQ_d_ne
                    # if g.Q_cbj[1, j, l] != 1:
                    #     print("Q_cbj =", g.Q_cbj[1,j,l])
                    # print("Q_d[0,j] = ", g.Q_d[0,j])
                g.Q_d[1, -2] = nQ_d_ne
                g.Q_a[1, -2] += ne[0]

        if my_mpi_row < (self.p_y_dims - 1):
            # print("rank ", my_rank, " fixing lower")
            # Update lower bound
            bot = borders[1]
            for j in range(1, self.p_local_grid_x_dim + 1):
                if bot[j, 0]:
                    nQ_d_bot = g.Q_d[-2, j] + bot[j, 0]
                    for l in range(self.l_params['nj']):
                        old_cut_bot = g.Q_cbj[-2, j, l] * g.Q_d[-2, j]
                        g.Q_cbj[-2, j, l] = (old_cut_bot + bot[j, l + 1]) / nQ_d_bot
                    g.Q_d[-2, j] = nQ_d_bot
                    g.Q_a[-2, j] += bot[j,0]

            # Update sw corner:
            if my_mpi_col > 0 and sw[0]:
                nQ_d_sw = g.Q_d[-2, 1] + sw[0]
                for l in range(self.l_params['nj']):
                    old_cut_sw = g.Q_cbj[-2, 1, l] * g.Q_d[-2, 1]
                    g.Q_cbj[-2, 1, l] = (old_cut_sw + sw[l+1]) / nQ_d_sw
                    # if g.Q_cbj[1, j, l] != 1:
                    #     print("Q_cbj =", g.Q_cbj[1,j,l])
                    # print("Q_d[0,j] = ", g.Q_d[0,j])
                g.Q_d[-2, 1] = nQ_d_sw
                g.Q_a[-2, 1] += sw[0]

            # Update se corner:
            if my_mpi_col > 0 and se[0]:
                nQ_d_se = g.Q_d[-2, -2] + se[0]
                for l in range(self.l_params['nj']):
                    old_cut_se = g.Q_cbj[-2, -2, l] * g.Q_d[-2, -2]
                    g.Q_cbj[-2, -2, l] = (old_cut_se + se[l+1]) / nQ_d_se
                    # if g.Q_cbj[1, j, l] != 1:
                    #     print("Q_cbj =", g.Q_cbj[1,j,l])
                    # print("Q_d[0,j] = ", g.Q_d[0,j])
                g.Q_d[-2, -2] = nQ_d_se
                g.Q_a[-2, -2] += se[0]

        if my_mpi_col > 0:
        #     # print("rank ", my_rank, " fixing left")
        #     # Update left bounds
            left = borders[2]
            for j in range(1, self.p_local_grid_y_dim + 1):
                if left[j, 0]:
                    # print("here! rank = ", my_rank, " left[j,0] = ", left[j,0])
                    nQ_d_left = g.Q_d[j, 1] + left[j, 0]
                    for l in range(self.l_params['nj']):
                        old_cut_left = g.Q_cbj[j, 1, l] * g.Q_d[j, 1]
                        g.Q_cbj[j, 1, l] = (old_cut_left + left[j, l + 1]) / nQ_d_left
                        # if g.Q_cbj[1, j, l] == 1:
                        #     print("left[j,:] = ", left[j,:])
                        #     print("Q_cbj =", g.Q_cbj[1,j,l])
                    g.Q_d[j, 1] = nQ_d_left
                    g.Q_a[j, 1] += left[j,0]
        if my_mpi_col < (self.p_x_dims - 1):
            # print("rank ", self.my_rank, " fixing right", "col  =",
            #       self.my_mpi_col," p_x_dims =",self.p_x_dims )
            # Update right bounds
            right = borders[3]
            for j in range(1, self.p_local_grid_y_dim + 1):
                if right[j, 0]:
                    # print("here! rank = ", my_rank, " right[j,0] = ", right[j,0])
                    nQ_d_right = g.Q_d[j, -2] + right[j, 0]
                    for l in range(self.l_params['nj']):
                        old_cut_right = g.Q_cbj[j, -2, l] * g.Q_d[j, -2]
                        g.Q_cbj[j, -2, l] = (old_cut_right + right[j, l + 1]) / nQ_d_right
                    g.Q_d[j, -2] = nQ_d_right
                    g.Q_a[j, -2] += right[j,0]

    def get_mpi_nb_flux(self, t, b, l, r):
        """ This function sends and receives 'sediment flux', i.e.\
            sediment sendt to another cell due to toppling, in another\
            local grid."""
        comm = self.comm
        p_local_grid_x_dim = self.p_local_grid_x_dim
        p_local_grid_y_dim = self.p_local_grid_y_dim
        neighbor_processes = self.nb_procs
        nj = self.l_params['nj']
        # Receive buffers
        top = np.zeros((p_local_grid_x_dim + 2, nj+1), dtype=np.double, order='C')
        bot = np.zeros((p_local_grid_x_dim + 2, nj+1), dtype=np.double, order='C')
        left = np.zeros((p_local_grid_y_dim + 2, nj+1), dtype=np.double, order='C')
        right = np.zeros((p_local_grid_y_dim + 2, nj+1), dtype=np.double, order='C')
        nw = np.zeros((nj+1), dtype=np.double)  # Corner receive buffers
        ne = np.zeros((nj+1), dtype=np.double)  # Corner receive buffers
        se = np.zeros((nj+1), dtype=np.double)  # Corner receive buffers
        sw = np.zeros((nj+1), dtype=np.double)  # Corner receive buffers

        # Send up, recv from down
        comm.Sendrecv(
            [t, MPI.DOUBLE],
            neighbor_processes[self.UP],
            0,
            [bot, MPI.DOUBLE],
            neighbor_processes[self.DOWN],
            0
        )
        # Send down, recv from up
        comm.Sendrecv(
            [b, MPI.DOUBLE],
            neighbor_processes[self.DOWN],
            0,
            [top, MPI.DOUBLE],
            neighbor_processes[self.UP],
            0
        )
        # Send left, recv from right
        comm.Sendrecv(
            [l, MPI.DOUBLE],
            neighbor_processes[self.LEFT],
            0,
            [right, MPI.DOUBLE],
            neighbor_processes[self.RIGHT],
            0
        )
        # Send right, recv from left
        comm.Sendrecv(
            [r, MPI.DOUBLE],
            neighbor_processes[self.RIGHT],
            0,
            [left, MPI.DOUBLE],
            neighbor_processes[self.LEFT],
            0
        )
        send_nw = t[0]
        send_ne = t[-1]
        send_se = b[-1]
        send_sw = b[0]

        # Send NW, recv from SE
        comm.Sendrecv(
            [send_nw, nj+1, MPI.DOUBLE],
            neighbor_processes[self.NW],
            0,
            [se, nj+1, MPI.DOUBLE],
            neighbor_processes[self.SE],
            0
        )
        # Send SE, recv from NW
        comm.Sendrecv(
            [send_se, nj+1, MPI.DOUBLE],
            neighbor_processes[self.SE],
            0,
            [nw, nj+1, MPI.DOUBLE],
            neighbor_processes[self.NW],
            0
        )
        # Send NE, recv from SW
        comm.Sendrecv(
            [send_ne, nj+1, MPI.DOUBLE],
            neighbor_processes[self.NE],
            0,
            [sw, nj+1, MPI.DOUBLE],
            neighbor_processes[self.SW],
            0
        )
        # Send SW, recv from NE
        comm.Sendrecv(
            [send_sw, nj+1, MPI.DOUBLE],
            neighbor_processes[self.SW],
            0,
            [ne, nj+1, MPI.DOUBLE],
            neighbor_processes[self.NE],
            0
        )
        result = [top, bot, left, right, nw, ne, se, sw]
        return result

    def gather_grid(self, local_petri_A):
        comm = self.comm
        send = np.zeros((self.p_local_grid_y_dim, self.p_local_grid_x_dim), dtype=np.double)
        TEMP = np.zeros((self.IMG_Y, self.IMG_X), dtype=np.double)

        send[:, :] = local_petri_A[1:-1, 1:-1].copy()

        if self.my_rank != 0:
            # print("rank = {0}\n"
            #       "send = \n"
            #       "{1}".format(rank, send))
            comm.Send([send, MPI.DOUBLE], dest=0, tag=self.my_rank)
        else:
            i = 0  # Receive from rank i
            x_start = 0
            y_start = 0
            for row in range(self.p_y_dims):
                x_start = 0
                for col in range(self.p_x_dims):
                    if i > 0:
                        dest = np.zeros((self.local_dims[i][0], self.local_dims[i][1]), dtype=np.double)
                        # print("i = {0}, dest.shape = {1}".format(i,dest.shape))
                        # dest = TEMP[(row * self.local_dims[i][0]):((row + 1) * self.local_dims[i][0]),
                        #        col * self.local_dims[i][1]:((col + 1) * self.local_dims[i][1])]
                        comm.Recv([dest, MPI.DOUBLE], source=i, tag=i)
                        # print("recved = \n"
                        #       "{0}\n"
                        #       "from rank {1}\n"
                        #       "put in TEMP[{2}:{3},{4}:{5}]\n".format(dest, i,y_start, y_start + self.local_dims[i][0],
                        #                                              x_start, x_start + self.local_dims[i][1]))
                        # print("x_start = {0}, y_start = {1}".format(x_start, y_start))
                        TEMP[(y_start):(y_start) + self.local_dims[i][0],
                        x_start:(x_start + self.local_dims[i][1])] = dest.copy()
                    i += 1
                    x_start += self.local_dims[i - 1][1]
                y_start += self.local_dims[i - 1][0]

                # Insert own local grid
                TEMP[0:self.local_dims[0][0], 0:self.local_dims[0][1]] = send.copy()
        comm.barrier()
        return TEMP

    def gather_cube(self, local_petri_A, z_dim):
        ans = np.zeros((self.IMG_Y, self.IMG_X, z_dim),
                       dtype=np.double, order='C')
        # temp = np.zeros((p_local_grid_y_dim * p_y_dims, p_local_grid_x_dim * p_x_dims),
        #                 dtype=np.double, order='C')
        for i in range(z_dim):
            ans[:,:,i] = self.gather_grid(local_petri_A[:,:,i])
        return ans

    def run(self, compare = False):
        """ Called by user in separate script. Starts simulation. """
        comm = self.comm
        start = timer()

        # Adding source before calculating dt has no effect

        # Exchange borders
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_th)
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_v)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_d)
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_o, 6)
        self.set_local_grid_bc()
        #
        # if self.my_rank==3:
        #     self.p_local_hexgrid.Q_d[:,:] = 10
        #     self.p_local_hexgrid.Q_cbj[:,:,0] = 1

        # Run toppling rule until convergence, before starting simulation
        try:
            if self.l_params['converged_toppling'][0]:
                tol = self.l_params['converged_toppling'][1]
                if self.my_rank == 0: print("Running toppling rule until converged with tol = {0}. This may take a while!".format(tol))
                jj = 0
                b_value = np.zeros((1),dtype='i')
                while True:
                    IMAGE_Qd_old = self.gather_grid(self.p_local_hexgrid.Q_d)


                    self.p_local_hexgrid.I_4()
                    # if my_rank != 0:
                    #     print("0 my rank = ", my_rank, "\nnonzero Qd=\n", np.where(np.logical_and(p_local_hexgrid.Q_d > 1, np.isfinite(p_local_hexgrid.Q_d))))
                    self.mpi_toppling_fix()
                    # if my_rank != 0:
                    #     print("1 my rank = ", my_rank, "\nnonzero Qd=\n",
                    #           np.where(np.logical_and(p_local_hexgrid.Q_d > 1, np.isfinite(p_local_hexgrid.Q_d))))
                    self.exchange_borders_matrix(self.p_local_hexgrid.Q_d)
                    self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
                    self.exchange_borders_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
                    self.set_local_grid_bc()
                    IMAGE_Qd = self.gather_grid(self.p_local_hexgrid.Q_d)
                    self.comm.barrier()

                    jj += 1
                    if self.my_rank == 0:
                        if ma.two_norm(IMAGE_Qd_old, IMAGE_Qd) <= tol:
                            b_value[0] = 1
                            print("Convergence achieved after {0} iterations. Continuing simulation.".format(jj))
                        if jj > 1000:
                            b_value[0] = 1
                            print("Convergence could not be achieved after {0} iterations. Continuing simulation.".format(jj))
                    self.comm.barrier()
                    self.comm.Bcast(b_value,0)
                    if b_value:
                        break
        except KeyError:
            print("Toppling rule keyword not found! Skipping toppling before simulation.")
        # print("rank = {0}, xy = {1}".format(self.my_rank, [self.l_params['x'],self.l_params['y']]))
        # if self.p_local_hexgrid.Q_th.all() == 0:
        #     raise Exception("rank {0}".format(self.my_rank))
        if not compare:
            del self.result_grid # Free memory during simulation
        self.comm.barrier()
        sourcelim = 10000
        for num_iterations in range(self.ITERATIONS):





            # Calculate time step and set common dt in all local grids
            p_global_dt = np.zeros((1), dtype=np.double, order='C')
            p_local_dt = np.array(self.p_local_hexgrid.calc_dt(global_grid=False),
                                  dtype=np.double, order='C')

            comm.barrier()
            comm.Allreduce(p_local_dt, p_global_dt, op=MPI.MIN)
            p_global_dt = p_global_dt[0]
            if p_global_dt >= 9999999:  # If dt = default MPI value set to 0.01 (same as single core)
                p_global_dt = 0.01
            # print("rank {0}, local_dt = {1}, global dt ={2}".format(self.my_rank, p_local_dt, p_global_dt))
            self.p_local_hexgrid.dt = p_global_dt  # Set dt
            if self.my_rank == 0:
                self.save_dt.append(p_global_dt)
            # Add source
            if num_iterations < sourcelim:
                self.p_local_hexgrid.addSource()
            # print("rank", self.my_rank, "before qth[y,x] = ", self.p_local_hexgrid.Q_th.sum())
            self.set_local_grid_bc()
            # print("after", self.my_rank, " qth[y,x] = ", self.p_local_hexgrid.Q_th.sum())

            # if self.p_local_hexgrid.Q_th.all() == 0:
            #     print("rank {0} has zero qth".format(self.my_rank))

            # Iterate CA
            # self.print_subgridQ('Qa')
            # self.print_subgridQ('Qth')
            # if num_iterations == 100 or num_iterations == 150:
            #     g = self.p_local_hexgrid
            #     print("rank = {0}, sum = {1}".format(self.my_rank,g.Q_d[1:-1, 1:-1].sum()))
                # if self.my_rank==0 or self.my_rank==2:
            #
            #         np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)
            #
            #         print("rank = {0}, g.Q_d=\n{1}".format(self.my_rank, g.Q_d))
            #         print("outflux = ", g.t.sum(), " ", g.b.sum())
            #         print("Q_a[ydim, 5] = ", g.Q_a[self.p_local_grid_y_dim+1,5])
            #         print("sbdiff[...,3] = ", g.seaBedDiff[self.p_local_grid_y_dim-1,5,3])
            #         g.I_4(debug=1)

            if compare:
                self.compare_steps(num_iterations, p_global_dt)
            else:
                self.iterateCA()
                # pass
            if ((num_iterations + 1) % self.sample_rate == 0) and num_iterations > 0:
                self.sample(num_iterations)
                if self.my_rank == 0: self.j_values.append(num_iterations + 1)
        self.comm.barrier()  # Ensure that no rank tries to load while writing npy files
        # if self.plot_bool[1]:
        del self.p_local_hexgrid
        self.result_grid = CAenv.CAenvironment(self.g_params, mpi=True)
        self.result_grid.parameters['save_dir'] = self.save_path_png
        self.print_figures()
        wtime = timer() - start
        if self.my_rank == 0:
            q_th0 = self.result_grid.sourcevalues['Q_th']
            q_cj0: list = self.result_grid.sourcevalues['Q_cj']
            qcjsum = np.sum(q_cj0)
            discharged = 2650*qcjsum*q_th0*np.sum(self.save_dt[:sourcelim])
            print("active source = {0:.2f} s".format(np.sum(self.save_dt[:sourcelim]) ))
            stime = sum(self.save_dt)
            self.result_grid.print_log("Wall time used = {0}, simulated time = {1}\n"
                                       "n_cores = {2}\n"
                                       "{3} kg discharged".format(wtime,stime, self.num_procs,
                                                                  discharged))
            print('{0} is complete. Wall time elapsed = {1:.2f}\n'
                  '{2:.2f} seconds simulation time. {3:.2f} kg discharged'.format(self.config, wtime, stime,
                                                                          discharged))

    def print_subgridQ(self, Q):
        g = self.p_local_hexgrid
        if Q == 'Qa':
            print("rank {0}: g.Q_a =\n{1}".format(self.my_rank, g.Q_a))
        elif Q == 'Qd':
            print("rank {0}: g.Q_d =\n{1}".format(self.my_rank, g.Q_d))
        elif Q == 'Qth':
            print("rank {0}: g.Q_th =\n{1}".format(self.my_rank, g.Q_th))

    def compare_steps(self, j, p_global_dt):
        """ This function compares the mpi result with the\
         single core result after each component of the transition fuction."""
        self.result_grid.addSource()
        self.result_grid.dt = self.result_grid.calc_dt()
        if self.my_rank == 0 and np.abs(p_global_dt - self.result_grid.dt) > 1e-15:
            print("rank = {3} image_dt = {0}, result_grid.dt = {1}. delta = {2}".format(p_global_dt, self.result_grid.dt,
                                                                         np.abs(p_global_dt - self.result_grid.dt), self.my_rank))
        self.p_local_hexgrid.T_1()
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_th)
        self.set_local_grid_bc()
        self.compare_mpi_singlecore(j, self.result_grid.T_1)
        self.p_local_hexgrid.T_2()
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_d)
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        self.set_local_grid_bc()
        self.compare_mpi_singlecore(j, self.result_grid.T_2)
        self.p_local_hexgrid.I_1()
        self.exchange_borders_cube(self.p_local_hexgrid.Q_o, 6)
        self.set_local_grid_bc()
        self.compare_mpi_singlecore(j, self.result_grid.I_1)
        self.p_local_hexgrid.I_2()
        self.exchange_borders_matrix(self.p_local_hexgrid.Q_th)
        self.exchange_borders_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        self.set_local_grid_bc()
        self.compare_mpi_singlecore(j, self.result_grid.I_2)
        # self.p_local_hexgrid.I_3()
        # self.exchange_borders_matrix(self.p_local_hexgrid.Q_v)
        # self.set_local_grid_bc()
        # self.compare_mpi_singlecore(j, self.result_grid.I_3)
        # self.p_local_hexgrid.I_4()
        # # if my_rank != 0:
        # #     print("0 my rank = ", my_rank, "\nnonzero Qd=\n", np.where(np.logical_and(p_local_hexgrid.Q_d > 1, np.isfinite(p_local_hexgrid.Q_d))))
        # self.mpi_toppling_fix()
        # # if my_rank != 0:
        # #     print("1 my rank = ", my_rank, "\nnonzero Qd=\n",
        # #           np.where(np.logical_and(p_local_hexgrid.Q_d > 1, np.isfinite(p_local_hexgrid.Q_d))))
        # self.exchange_borders_matrix(self.p_local_hexgrid.Q_d)
        # self.exchange_borders_matrix(self.p_local_hexgrid.Q_a)
        # self.exchange_borders_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
        # self.set_local_grid_bc()
        # self.compare_mpi_singlecore(j, self.result_grid.I_4)



    def compare_mpi_singlecore(self, j, function):
        """ This function compares an mpi calculated substate with an SC\
            calculated substate. """

        image_Q_th = self.gather_grid(self.p_local_hexgrid.Q_th)
        if self.my_rank == 0:
            # Check if the mpi and nonmpi grids are identical

            function()
            # print("image = \n", image_Q_th)
            # print("\nresult_grid = \n",  self.result_grid.Q_th)
            # result_grid.grid.Q_th[1,1] += 1 #induce intentional error

            if np.any(np.abs(image_Q_th -self.result_grid.Q_th)>1e-15):
                ii, jj = np.where(np.abs(image_Q_th -self.result_grid.Q_th)>1e-15)
                s = "num_iterations = {0}\n".format(j)
                s = s + ''.join("image ={0}\n".format(image_Q_th))
                s = s + ''.join("\nresult_grid = {0}\n".format(self.result_grid.Q_th))
                # s = s + ''.join("local_source_tiles = {0}, {1}\n. result_grid_source = {2},{3}.\n".format(
                #     p_local_grid_parameters['x'], p_local_grid_parameters['y'], parameters['x'], parameters['y']
                # ))
                s = s + ''.join("image_q_th[{0},{1}] = {2} != result_grid.grid.Q_th[{3},{4}] = {5}, delta = {6}\n".format(
                    ii[x], jj[x], image_Q_th[ii[x], jj[x]], ii[x], jj[x], self.result_grid.Q_th[ii[x], jj[x]],
                    np.abs(image_Q_th[ii[x], jj[x]] - self.result_grid.Q_th[ii[x], jj[x]])) for x in
                                range(len(ii)))
                raise Exception("not equal: " + s)
            # print("all clear!")

    def sample(self, num_iterations):
        """ This function samples/saves substates to a binaries\
            directory. These are used at the end of the simulation\
            for generating plots. """
        self.i_sample_values.append(num_iterations)
        bottom_indices = self.bottom_indices

        if self.plot_bool[4]:  # save assembled grids
            self.p_local_hexgrid.print_npy(num_iterations, self.num_procs)
        # print("sample")
        # Gather grids
        IMAGE_Q_th = self.gather_grid(self.p_local_hexgrid.Q_th)
        # print("rank = {0}, zero local qth = {1}".format(self.my_rank, np.all(self.p_local_hexgrid.Q_th == 0)))
        # if self.my_rank ==0: print("gathered qth zero? ", np.all(IMAGE_Q_th == 0))
        IMAGE_Q_cbj = self.gather_cube(self.p_local_hexgrid.Q_cbj, self.l_params['nj'])
        IMAGE_Q_cj = self.gather_cube(self.p_local_hexgrid.Q_cj, self.l_params['nj'])
        IMAGE_Q_d = self.gather_grid(self.p_local_hexgrid.Q_d)
        IMAGE_Q_v = self.gather_grid(self.p_local_hexgrid.Q_v)
        IMAGE_Q_o = self.gather_cube(self.p_local_hexgrid.Q_o, 6)

        if self.my_rank == 0:
            # if self.plot_bool[0]: # 1d plot
            #     ch_bot_thickness = [IMAGE_Q_th[bottom_indices[i]] for i in
            #                         range(len(bottom_indices))]
            #     ch_bot_speed = [IMAGE_Q_v[bottom_indices[i]] for i in range(len(bottom_indices))]
            #     ch_bot_thickness_cons = []
            #     ch_bot_sediment_cons = []
            #     for jj in range(self.l_params['nj']):
            #         ch_bot_thickness_cons.append(
            #             [IMAGE_Q_cj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
            #         ch_bot_sediment_cons.append(
            #             [IMAGE_Q_cbj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
            #     ch_bot_sediment = [IMAGE_Q_d[bottom_indices[i]] for i in range(len(bottom_indices))]
            #     ch_bot_outflow = [sum(IMAGE_Q_o[bottom_indices[i]]) for i in
            #                       range(len(bottom_indices))]
            #
            #     np.save(self.save_path_txt + 'ch_bot_outflow_{0}'.format(num_iterations + 1), ch_bot_outflow)
            #     np.save(self.save_path_txt + 'ch_bot_speed_{0}'.format(num_iterations + 1), ch_bot_speed)
            #     np.save(self.save_path_txt + 'ch_bot_thickness_{0}'.format(num_iterations + 1), ch_bot_thickness)
            #     np.save(self.save_path_txt + 'ch_bot_sediment_{0}'.format(num_iterations + 1), ch_bot_sediment)
            #     for jj in range(self.l_params['nj']):
            #         np.save(self.save_path_txt + 'ch_bot_sediment_cons{0}_{1}'.format(jj, num_iterations + 1),
            #                 ch_bot_sediment_cons[jj])
            #         np.save(self.save_path_txt + 'ch_bot_thickness_cons{0}_{1}'.format(jj, num_iterations + 1),
            #                 ch_bot_thickness_cons[jj])
            # if self.plot_bool[1]:  # 2d plot
            np.save(self.save_path_txt + 'Q_th_{0}'.format(num_iterations + 1), IMAGE_Q_th)
            np.save(self.save_path_txt + 'Q_cbj_{0}'.format(num_iterations + 1), IMAGE_Q_cbj)
            np.save(self.save_path_txt + 'Q_cj_{0}'.format(num_iterations + 1), IMAGE_Q_cj)
            np.save(self.save_path_txt + 'Q_d_{0}'.format(num_iterations + 1), IMAGE_Q_d)
            np.save(self.save_path_txt + 'Q_v_{0}'.format(num_iterations + 1), IMAGE_Q_v)
            if self.plot_bool[3]:  # stability curves
                self.mass.append(IMAGE_Q_th[:, :, None] * IMAGE_Q_cj)
                self.massBed.append(np.sum(IMAGE_Q_d[1:-1, 1:-1].flatten(), axis=None))
                self.density.append(np.amax(IMAGE_Q_cj[:, :, 0].flatten()))
                self.beddensity.append(np.amax(IMAGE_Q_cbj[:, :, 0]))


    def print_figures(self):
        # Stability curves
        if self.my_rank == 0 and self.plot_bool[3]:  # stability curves:
            g = self.result_grid
            g.density = self.density
            g.time = [self.save_dt[x-1] for x in self.j_values]
            np.save(self.save_path_txt+'timesteps', self.save_dt)
            g.mass = self.mass
            g.plotStabilityCurves(self.j_values)
        if self.my_rank == 0 and self.plot_bool[5]:
            self.result_grid.plot_bathy()


        num_figs = self.ITERATIONS // self.sample_rate
        figs_per_proc = num_figs // self.num_procs
        X0 = np.load(self.save_path_txt + 'X000.npy')
        X1 = np.load(self.save_path_txt + 'X001.npy')

        upper_lim = (self.my_rank + 1) * figs_per_proc
        if self.my_rank == (self.num_procs - 1) and (upper_lim - 1) < num_figs:
            # print("rank ={0}, upperlim = {1}, numprocs = {2}".format(my_rank,upper_lim, num_procs))
            upper_lim = num_figs
            # print(upper_lim)
        g = self.result_grid
        for i in range(self.my_rank * figs_per_proc, upper_lim):
            # print("i = ", i, " len(i_sample_) = ", len(i_sample_values))
            r = self.load_txt_files(self.i_sample_values[i], self.l_params)
            IMAGE_Q_th, IMAGE_Q_cbj, IMAGE_Q_cj, IMAGE_Q_d, IMAGE_Q_v = r[0:5]
            g.Q_th = IMAGE_Q_th
            g.Q_cbj = IMAGE_Q_cbj
            g.Q_cj = IMAGE_Q_cj
            g.Q_d = IMAGE_Q_d
            g.Q_v = IMAGE_Q_v
            # print("here")
            # print("in mpi zero Q_th = ", np.all(IMAGE_Q_th == 0))
            # print("sum Q_d = ", (IMAGE_Q_d).sum())
            if self.plot_bool[1]: g.plot2d(self.i_sample_values[i])
            if self.plot_bool[0]: # plot ch_bot
                print("printing")
                g.sampleValues()  # Creates ch_bot_ variables using the fresh substates
                # ch_bot_outflow, ch_bot_thickness, ch_bot_speed, \
                # ch_bot_sediment, ch_bot_sediment_cons, ch_bot_thickness_cons = r[5:]
                # g.ch_bot_outflow = ch_bot_outflow
                # g.ch_bot_thickness = IMAGE_Q_th[g.bot_indices]
                # g.ch_bot_speed = ch_bot_speed
                # g.ch_bot_sediment = [g.Q_d[g.bot_indices[i]] for i in range(len(g.bot_indices))]
                # g.ch_bot_sediment_cons = ch_bot_sediment_cons
                # g.ch_bot_thickness_cons = ch_bot_thickness_cons
                g.plot1d(self.i_sample_values[i])


            # print_substate(self.l_params['ny'], self.l_params['nx'], self.i_sample_values[i],
            #                IMAGE_Q_th, IMAGE_Q_cj, IMAGE_Q_cbj, IMAGE_Q_d,
            #                X0, X1, self.l_params['terrain'], ch_bot_thickness, ch_bot_speed, ch_bot_outflow,
            #                ch_bot_sediment, ch_bot_sediment_cons, ch_bot_thickness_cons)

    def load_txt_files(self, num_iterations, parameters):
        """ This function loads and returns the sampled substates. """
        save_path_txt = self.save_path_txt
        IMAGE_Q_th = np.load((save_path_txt+ 'Q_th_{0}.npy'.format(num_iterations + 1)))
        IMAGE_Q_cbj = np.load((save_path_txt+ 'Q_cbj_{0}.npy'.format(num_iterations + 1)))
        IMAGE_Q_cj = np.load((save_path_txt+ 'Q_cj_{0}.npy'.format(num_iterations + 1)))
        IMAGE_Q_d = np.load((save_path_txt+ 'Q_d_{0}.npy'.format(num_iterations + 1)))
        IMAGE_Q_v = np.load((save_path_txt+ 'Q_v_{0}.npy'.format(num_iterations + 1)))
        r = [IMAGE_Q_th, IMAGE_Q_cbj, IMAGE_Q_cj, IMAGE_Q_d, IMAGE_Q_v]
        if self.plot_bool[0]:
            ch_bot_thickness = np.load((save_path_txt+ 'ch_bot_thickness_{0}.npy'.format(num_iterations + 1)))
            ch_bot_outflow = np.load((save_path_txt+ 'ch_bot_outflow_{0}.npy'.format(num_iterations + 1)))
            ch_bot_speed = np.load((save_path_txt+ 'ch_bot_speed_{0}.npy'.format(num_iterations + 1)))
            ch_bot_sediment = np.load((save_path_txt+ 'ch_bot_sediment_{0}.npy'.format(num_iterations + 1)))
            ch_bot_sediment_cons = []
            ch_bot_thickness_cons = []
            for jj in range(parameters['nj']):
                ch_bot_sediment_cons.append(np.load(
                    (save_path_txt+ 'ch_bot_sediment_cons{0}_{1}.npy'.format(jj, num_iterations + 1))))
                ch_bot_thickness_cons.append(np.load(
                    (save_path_txt+ 'ch_bot_thickness_cons{0}_{1}.npy'.format(jj, num_iterations + 1))))
            r.extend([ch_bot_outflow, ch_bot_thickness, ch_bot_speed, ch_bot_sediment, ch_bot_sediment_cons, ch_bot_thickness_cons])
            # ch_bot_sediment_cons0     = np.load((save_path_txt,     'ch_bot_sediment_cons0_{0}.npy'.format(num_iterations + 1)))
            # ch_bot_sediment_cons1     = np.load((save_path_txt,     'ch_bot_sediment_cons1_{0}.npy'.format(num_iterations + 1)))
            # ch_bot_thickness_cons0     = np.load((save_path_txt,     'ch_bot_thickness_cons0_{0}.npy'.format(num_iterations + 1)))
            # ch_bot_thickness_cons1     = np.load((save_path_txt,     'ch_bot_thickness_cons1_{0}.npy'.format(num_iterations + 1)))
        return r


def global_coords_to_local_coords(y, x, my_mpi_row, my_mpi_col, p_local_grid_x_dim, p_local_grid_y_dim):
    ''' Convert between global and local indices. Return (-1,-1) if outside local grid.'''
    # This ensures no error even if y = np.ix_(np.arange(5,6)) = array([5]) or y = 5.
    try:
        x = int(x[0])
    except:
        pass
    try:
        y = int(y[0])
    except:
        pass

    if (x >= (my_mpi_col * p_local_grid_x_dim)) & (x < ((my_mpi_col + 1) * p_local_grid_x_dim)):
        local_x = x - (my_mpi_col * p_local_grid_x_dim)
    else:
        return tuple([-1, -1])
    # print("{0}, {1}".format(type(y),type((my_mpi_row * p_local_grid_y_dim))))
    if (y >= (my_mpi_row * p_local_grid_y_dim)) & (y < ((my_mpi_row + 1) * p_local_grid_y_dim)):
        local_y = y - (my_mpi_row * p_local_grid_y_dim)
    else:
        return tuple([-1, -1])
    return tuple([local_x + 1, local_y + 1]) # Add 1 because of the borders

def define_px_py_dims(num_procs, ny, nx):
    '''

    :param num_procs: total number of processors
    :param ny: grid size in y direction
    :param nx: grid size in x direction
    :return: Number of processors in x and y direction
    '''
    assert num_procs > 0
    if ma.is_square(num_procs): # If square number
        p_y_dims = np.sqrt(num_procs)
        p_x_dims = np.sqrt(num_procs)
    # elif num_procs % 2 == 0:
    else:
        factors = ma.largest_factor(num_procs) # factors[0]>factors[1]
        if ny >= nx:
            p_x_dims = factors[1]
            p_y_dims = factors[0]
        else:
            p_x_dims = factors[0]
            p_y_dims = factors[1]
    # else:
    #     raise Exception("Please use an even or square number of processors!")
    return int(p_y_dims), int(p_x_dims)

def define_local_hexgrid_size(IMG_dim, p_xy_dim, my_dim_coord):
    '''

    :param IMG_dim: Size of combined grid in x or y direction
    :param p_xy_dim: Number of procs in x or y direction
    :param my_dim_coord: my row/col in the cartesian grid \
    if defining y direction size -> give my_row
    if defining x direction size -> give my_col
    :return: Size of the local grid in either x or y direction
    '''
    if int(IMG_dim / p_xy_dim) == IMG_dim / p_xy_dim:
        p_local_grid_dim = IMG_dim / p_xy_dim
    else:
        if my_dim_coord == 0:
            remainder = IMG_dim - np.floor(IMG_dim / p_xy_dim) * p_xy_dim
            p_local_grid_dim = np.floor(IMG_dim / p_xy_dim) + remainder
        else:
            p_local_grid_dim = np.floor(IMG_dim / p_xy_dim)
    return int(p_local_grid_dim)

