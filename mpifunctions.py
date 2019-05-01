from mpi4py import MPI
import numpy as np
import sys
import CAenvironment as CAenv
import scipy.io
import mathfunk as ma
import matplotlib.pyplot as plt


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
        print("my rank = {0}"
              "nb[N] = {1}"
              "nb[S] = {2}"
              "nb[E] = {3}"
              "nb[W] = {4}".format(self.my_rank, self.nb_procs[self.UP], self.nb_procs[self.DOWN],
                                   self.nb_procs[self.RIGHT], self.nb_procs[self.LEFT]))

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
        # Load CA parameters from file
        self.g_params = CAenv.import_parameters(config)

        if self.my_rank is 0:
            result_grid = CAenv.CAenvironment(self.g_params)

        self.IMG_X = self.g_params['nx']
        self.IMG_Y = self.g_params['ny']


        self.ITERATIONS = self.g_params['num_iterations']

        # Setup parameters for local CAenvironment
        self.l_params = self.g_params.copy()
        self.l_params['x'] = None  # if None == no source
        self.l_params['y'] = None  # if None == no source
        self.l_params['nx'] = self.p_local_grid_x_dim + 2  # make room for borders
        self.l_params['ny'] = self.p_local_grid_y_dim + 2  # make room for borders
        # TODO convert from N, E to x,y
        self.set_local_grid_source_xy()

        # self.border_row_t = MPI.DOUBLE.Create_vector(p_local_grid_x_dim + 2,
        #                                         1,
        #                                         1)
        # self.border_row_t.Commit()
        #
        # self.border_col_t = MPI.DOUBLE.Create_vector(p_local_grid_y_dim + 2,
        #                                         1,
        #                                         p_local_grid_x_dim + 2)
        # self.border_col_t.Commit()

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

def exchange_borders_cube(local_petri_A, z_dim):

    for i in range(z_dim):
        # Send data south and receive from north
        local_grid_send_south = local_petri_A[-2,:,i].copy()
        local_grid_receive_north = np.zeros(p_local_grid_x_dim+2, dtype=np.double, order='C')
        comm.Sendrecv(
            [local_grid_send_south, p_local_grid_x_dim + 2, MPI.DOUBLE],  # send the second last row
            neighbor_processes[DOWN],
            0,
            [local_grid_receive_north, p_local_grid_x_dim + 2, MPI.DOUBLE],  # recvbuf = first row
            neighbor_processes[UP],
            0
        )
        local_petri_A[0,:,i] = local_grid_receive_north.copy()

        # # Send data north and receive from south
        local_grid_send_north = local_petri_A[1,:,i].copy()
        local_grid_receive_south = np.zeros(p_local_grid_x_dim + 2, dtype=np.double, order='C')
        comm.Sendrecv(
            [local_grid_send_north, p_local_grid_x_dim + 2, MPI.DOUBLE],  # sendbuf = second row (1.row = border)
            neighbor_processes[UP],  # destination
            1,  # sendtag
            [local_grid_receive_south, p_local_grid_x_dim +2, MPI.DOUBLE],  # recvbuf = last row
            neighbor_processes[DOWN],  # source
            1
        )
        local_petri_A[-1,:,i] = local_grid_receive_south.copy()

        # Send west and receive from east
        local_grid_send_west = local_petri_A[:, 1, i].copy()
        local_grid_receive_east = np.zeros(p_local_grid_y_dim + 2, dtype=np.double, order='C')
        comm.Sendrecv(
            [local_grid_send_west, p_local_grid_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            2,
            [local_grid_receive_east, p_local_grid_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            2
        )
        local_petri_A[:, -1, i] = local_grid_receive_east.copy()

        # Send east and receive from west
        local_grid_send_east = local_petri_A[:, -2, i].copy()
        local_grid_receive_west = np.zeros(p_local_grid_y_dim + 2, dtype=np.double, order='C')
        comm.Sendrecv(
            [local_grid_send_east, p_local_grid_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            0,
            [local_grid_receive_west, p_local_grid_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            0
        )
        local_petri_A[:, 0, i] = local_grid_receive_west.copy()


def exchange_borders_matrix(comm, local_petri_A, nb_procs, ):
    p_local_grid_y_dim = parameters['']
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    local_grid_wb = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_eb = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_ev = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_wv = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')


    # Send data south and receive from north
    comm.Sendrecv(
        [local_petri_A[-2, :], 1, border_row_t],  # send the second last row
        nb_procs[DOWN],
        0,
        [local_petri_A[0, :], 1, border_row_t],  # recvbuf = first row
        nb_procs[UP],
        0
    )

    # # Send data north and receive from south
    comm.Sendrecv(
        [local_petri_A[1, :], 1, border_row_t],  # sendbuf = second row (1.row = border)
        nb_procs[UP],  # destination
        1,  # sendtag
        [local_petri_A[-1, :], 1, border_row_t],  # recvbuf = last row
        nb_procs[DOWN],  # source
        1
    )
    #
    # Send west and receive from east
    local_grid_ev[:] = local_petri_A[:, 1].copy()
    comm.Sendrecv(
        [local_grid_ev, p_local_grid_y_dim + 2, MPI.DOUBLE],
        nb_procs[LEFT],
        2,
        [local_grid_wb, p_local_grid_y_dim + 2, MPI.DOUBLE],
        nb_procs[RIGHT],
        2
    )
    local_petri_A[:, -1] = local_grid_wb.copy()

    # # Send east and receive from west
    local_grid_wv[:] = local_petri_A[:, -2].copy()
    # print("my value = ", local_grid_wv)
    # print("my destination = ", local_grid_eb)

    comm.Sendrecv(
        [local_grid_wv, p_local_grid_y_dim + 2, MPI.DOUBLE],
        nb_procs[RIGHT],
        0,
        [local_grid_eb, p_local_grid_y_dim + 2, MPI.DOUBLE],
        nb_procs[LEFT],
        0
    )
    local_petri_A[:, 0] = local_grid_eb.copy()

def get_mpi_nb_flux(t, b, l, r):
    nj = parameters['nj']
    # Receive buffers
    top = np.zeros((p_local_grid_x_dim + 2, nj+1), dtype=np.double, order='C')
    bot = np.zeros((p_local_grid_x_dim + 2, nj+1), dtype=np.double, order='C')
    left = np.zeros((p_local_grid_y_dim + 2, nj+1), dtype=np.double, order='C')
    right = np.zeros((p_local_grid_y_dim + 2, nj+1), dtype=np.double, order='C')
    nw = ne = se = sw = np.zeros((1), dtype=np.double) # Corner receive buffers

    # Send up, recv from down
    comm.Sendrecv(
        [t, MPI.DOUBLE],
        neighbor_processes[UP],
        0,
        [bot, MPI.DOUBLE],
        neighbor_processes[DOWN],
        0
    )
    # Send down, recv from up
    comm.Sendrecv(
        [b, MPI.DOUBLE],
        neighbor_processes[DOWN],
        0,
        [top, MPI.DOUBLE],
        neighbor_processes[UP],
        0
    )
    # Send left, recv from right
    comm.Sendrecv(
        [l, MPI.DOUBLE],
        neighbor_processes[LEFT],
        0,
        [right, MPI.DOUBLE],
        neighbor_processes[RIGHT],
        0
    )
    # Send right, recv from left
    comm.Sendrecv(
        [r, MPI.DOUBLE],
        neighbor_processes[RIGHT],
        0,
        [left, MPI.DOUBLE],
        neighbor_processes[LEFT],
        0
    )
    send_nw = t[0]
    send_ne = t[-1]
    send_se = b[-1]
    send_sw = b[0]

    # Send NW, recv from SE
    comm.Sendrecv(
        [send_nw, 1, MPI.DOUBLE],
        neighbor_processes[NW],
        0,
        [se, 1, MPI.DOUBLE],
        neighbor_processes[SE],
        0
    )
    # Send SE, recv from NW
    comm.Sendrecv(
        [send_se, 1, MPI.DOUBLE],
        neighbor_processes[SE],
        0,
        [nw, 1, MPI.DOUBLE],
        neighbor_processes[NW],
        0
    )
    # Send NE, recv from SW
    comm.Sendrecv(
        [send_ne, 1, MPI.DOUBLE],
        neighbor_processes[NE],
        0,
        [sw, 1, MPI.DOUBLE],
        neighbor_processes[SW],
        0
    )
    # Send SW, recv from NE
    comm.Sendrecv(
        [send_sw, 1, MPI.DOUBLE],
        neighbor_processes[SW],
        0,
        [ne, 1, MPI.DOUBLE],
        neighbor_processes[NE],
        0
    )
    result = [top, bot, left, right, nw, ne, se, sw]
    return result

def mpi_toppling_fix():
    g = p_local_hexgrid.grid
    t = g.t
    b = g.b
    l = g.l
    r = g.r
    borders = get_mpi_nb_flux(t,b,l,r)
    # print("rank = ", my_rank, " max(right) = ", np.max(borders[3][:,0]))

    if my_mpi_row > 0:
        # print("rank ", my_rank, " fixing upper")
        # Update upper bound
        top = borders[0]
        for j in range(1,p_local_grid_x_dim+1):
            if top[j, 0]:
                nQ_d_top = g.Q_d[1, j] + top[j, 0]
                # print("q_d_top = ", nQ_d_top)
                for l in range(parameters['nj']):
                    old_cut_top = g.Q_cbj[1,j,l] * g.Q_d[1, j]
                    g.Q_cbj[1, j, l] = (old_cut_top + top[j, l+1]) / nQ_d_top
                    # if g.Q_cbj[1, j, l] != 1:
                    #     print("Q_cbj =", g.Q_cbj[1,j,l])
                # print("Q_d[0,j] = ", g.Q_d[0,j])
                g.Q_d[1, j] = nQ_d_top
                # print("nQ_d[0,j] = ", g.Q_d[0, j])
    if my_mpi_row < (p_y_dims):
        # print("rank ", my_rank, " fixing lower")
        # Update lower bound
        bot = borders[1]
        for j in range(1,p_local_grid_x_dim+1):
            if bot[j, 0]:
                nQ_d_bot = g.Q_d[-2, j] + bot[j, 0]
                for l in range(parameters['nj']):
                    old_cut_bot = g.Q_cbj[-2,j,l] * g.Q_d[-2, j]
                    g.Q_cbj[-2, j, l] = (old_cut_bot + bot[j, l+1]) / nQ_d_bot
                g.Q_d[-2, j] = nQ_d_bot
    if my_mpi_col > 0:
        # print("rank ", my_rank, " fixing left")
        # Update left bounds
        left = borders[2]
        for j in range(1,p_local_grid_y_dim +1):
            if left[j, 0]:
                # print("here! rank = ", my_rank, " left[j,0] = ", left[j,0])
                nQ_d_left = g.Q_d[j, 1] + left[j, 0]
                for l in range(parameters['nj']):
                    old_cut_left = g.Q_cbj[j,1,l] * g.Q_d[j,1]
                    g.Q_cbj[j, 1, l] = (old_cut_left + left[j, l+1]) / nQ_d_left
                    # if g.Q_cbj[1, j, l] == 1:
                    #     print("left[j,:] = ", left[j,:])
                    #     print("Q_cbj =", g.Q_cbj[1,j,l])
                g.Q_d[j, 1] = nQ_d_left
    if my_mpi_col < (p_x_dims):
        # print("rank ", my_rank, " fixing right")
        # Update right bounds
        right = borders[3]
        for j in range(1, p_local_grid_y_dim + 1):
            if right[j, 0]:
                # print("here! rank = ", my_rank, " right[j,0] = ", right[j,0])
                nQ_d_right = g.Q_d[j, -2] + right[j, 0]
                for l in range(parameters['nj']):
                    old_cut_right = g.Q_cbj[j, -2, l] * g.Q_d[j, -2]
                    g.Q_cbj[j, -2, l] = (old_cut_right + right[j, l+1]) / nQ_d_right
                g.Q_d[j, -2] = nQ_d_right

def iterateCA():
    p_local_hexgrid.grid.T_1()
    exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
    exchange_borders_matrix(p_local_hexgrid.grid.Q_th)
    set_bc(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy,
           parameters)
    p_local_hexgrid.grid.T_2()
    exchange_borders_matrix(p_local_hexgrid.grid.Q_d)
    exchange_borders_matrix(p_local_hexgrid.grid.Q_a)
    exchange_borders_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
    exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
    set_bc(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy,
           parameters)
    p_local_hexgrid.grid.I_1()
    exchange_borders_cube(p_local_hexgrid.grid.Q_o, 6)
    set_bc(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy,
           parameters)
    p_local_hexgrid.grid.I_2()
    exchange_borders_matrix(p_local_hexgrid.grid.Q_th)
    exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
    set_bc(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy,
           parameters)
    p_local_hexgrid.grid.I_3()
    exchange_borders_matrix(p_local_hexgrid.grid.Q_v)
    set_bc(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy,
           parameters)
    p_local_hexgrid.grid.I_4()
    # if my_rank != 0:
    #     print("0 my rank = ", my_rank, "\nnonzero Qd=\n", np.where(np.logical_and(p_local_hexgrid.grid.Q_d > 1, np.isfinite(p_local_hexgrid.grid.Q_d))))
    mpi_toppling_fix()
    # if my_rank != 0:
    #     print("1 my rank = ", my_rank, "\nnonzero Qd=\n",
    #           np.where(np.logical_and(p_local_hexgrid.grid.Q_d > 1, np.isfinite(p_local_hexgrid.grid.Q_d))))
    exchange_borders_matrix(p_local_hexgrid.grid.Q_d)
    exchange_borders_matrix(p_local_hexgrid.grid.Q_a)
    exchange_borders_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
    set_bc(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy,
           parameters)

def gather_cube(local_petri_A, z_dim):
    ans = np.zeros((IMG_Y, IMG_X, z_dim),
                   dtype=np.double, order='C')
    # temp = np.zeros((p_local_grid_y_dim * p_y_dims, p_local_grid_x_dim * p_x_dims),
    #                 dtype=np.double, order='C')
    for i in range(z_dim):
        ans[:,:,i] = gather_grid(local_petri_A[:,:,i])
    return ans

def gather_grid(local_petri_A):
    send = np.zeros((p_local_grid_y_dim, p_local_grid_x_dim), dtype=np.double)
    TEMP = np.zeros((IMG_Y, IMG_X), dtype=np.double)

    send[:, :] = local_petri_A[1:-1, 1:-1].copy()

    if my_rank != 0:
        # print("rank = {0}\n"
        #       "send = \n"
        #       "{1}".format(rank, send))
        comm.Send([send, MPI.DOUBLE], dest=0, tag=my_rank)
    else:
        i = 0 # Receive from rank i
        x_start = 0
        y_start = 0
        for row in range(p_y_dims):
            x_start = 0
            for col in range(p_x_dims):
                if i > 0:
                    dest = np.zeros((local_dims[i][0],local_dims[i][1]), dtype=np.double)
                    # print("i = {0}, dest.shape = {1}".format(i,dest.shape))
                    # dest = TEMP[(row * local_dims[i][0]):((row + 1) * local_dims[i][0]),
                    #        col * local_dims[i][1]:((col + 1) * local_dims[i][1])]
                    comm.Recv([dest, MPI.DOUBLE], source=i, tag=i)
                    # print("recved = \n"
                    #       "{0}\n"
                    #       "from rank {1}\n"
                    #       "put in TEMP[{2}:{3},{4}:{5}]\n".format(dest, i,y_start, y_start + local_dims[i][0],
                    #                                              x_start, x_start + local_dims[i][1]))
                    # print("x_start = {0}, y_start = {1}".format(x_start, y_start))
                    TEMP[(y_start):(y_start) + local_dims[i][0],
                            x_start:(x_start + local_dims[i][1])] = dest.copy()
                i += 1
                x_start += local_dims[i-1][1]
            y_start += local_dims[i-1][0]

            # Insert own local grid
            TEMP[0:local_dims[0][0], 0:local_dims[0][1]] = send.copy()
    comm.barrier()
    return TEMP

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



def load_txt_files(num_iterations, parameters):
    IMAGE_Q_th  = np.load(os.path.join(save_path_txt,  'Q_th_{0}.npy'.format(num_iterations + 1)))
    IMAGE_Q_cbj = np.load(os.path.join(save_path_txt, 'Q_cbj_{0}.npy'.format(num_iterations + 1)))
    IMAGE_Q_cj  = np.load(os.path.join(save_path_txt,  'Q_cj_{0}.npy'.format(num_iterations + 1)))
    IMAGE_Q_d   = np.load(os.path.join(save_path_txt,   'Q_d_{0}.npy'.format(num_iterations + 1)))

    ch_bot_thickness = np.load(os.path.join(save_path_txt, 'ch_bot_thickness_{0}.npy'.format(num_iterations + 1)))
    ch_bot_outflow   = np.load(os.path.join(save_path_txt,   'ch_bot_outflow_{0}.npy'.format(num_iterations + 1)))
    ch_bot_speed     = np.load(os.path.join(save_path_txt,     'ch_bot_speed_{0}.npy'.format(num_iterations + 1)))
    ch_bot_sediment     = np.load(os.path.join(save_path_txt,     'ch_bot_sediment_{0}.npy'.format(num_iterations + 1)))
    ch_bot_sediment_cons = []
    ch_bot_thickness_cons = []
    for jj in range(parameters['nj']):
        ch_bot_sediment_cons.append(np.load(
            os.path.join(save_path_txt, 'ch_bot_sediment_cons{0}_{1}.npy'.format(jj,num_iterations + 1))))
        ch_bot_thickness_cons.append(np.load(
            os.path.join(save_path_txt, 'ch_bot_thickness_cons{0}_{1}.npy'.format(jj,num_iterations + 1))))
    # ch_bot_sediment_cons0     = np.load(os.path.join(save_path_txt,     'ch_bot_sediment_cons0_{0}.npy'.format(num_iterations + 1)))
    # ch_bot_sediment_cons1     = np.load(os.path.join(save_path_txt,     'ch_bot_sediment_cons1_{0}.npy'.format(num_iterations + 1)))
    # ch_bot_thickness_cons0     = np.load(os.path.join(save_path_txt,     'ch_bot_thickness_cons0_{0}.npy'.format(num_iterations + 1)))
    # ch_bot_thickness_cons1     = np.load(os.path.join(save_path_txt,     'ch_bot_thickness_cons1_{0}.npy'.format(num_iterations + 1)))
    return IMAGE_Q_th, IMAGE_Q_cbj, IMAGE_Q_cj, IMAGE_Q_d, ch_bot_outflow, ch_bot_thickness,\
           ch_bot_speed, ch_bot_sediment, ch_bot_sediment_cons, ch_bot_thickness_cons

def print_substate(Ny, Nx, i, Q_th, Q_cj, Q_cbj, Q_d, X0, X1, terrain,
                   ch_bot_thickness, ch_bot_speed, ch_bot_outflow, ch_bot_sediment,
                   ch_bot_sediment_cons, ch_bot_thickness_cons):
    fig = plt.figure(figsize=(10, 6))
    ax = [fig.add_subplot(3, 2, i, aspect='equal') for i in range(1, 7)]
    ind = np.unravel_index(np.argmax(Q_th, axis=None), Q_th.shape)

    points = ax[0].scatter(X0.flatten(), X1.flatten(), marker='h',
                           c=Q_cj[:, :,0].flatten())

    plt.colorbar(points, shrink=0.6, ax=ax[0])
    ax[0].set_title('Q_cj[:,:,0]. n = ' + str(i + 1))

    points = ax[1].scatter(X0.flatten(), X1.flatten(), marker='h',
                           c=Q_th.flatten())
    # ax[1].scatter(X[ind[0],ind[1],0], X[ind[0],ind[1],1], c='r')  # Targeting
    plt.colorbar(points, shrink=0.6, ax=ax[1])
    ax[1].set_title('Q_th')

    points = ax[2].scatter(X0[1:-1, 1:-1].flatten(), X1[1:-1, 1:-1].flatten(), marker='h',
                           c=Q_cbj[1:-1, 1:-1,0].flatten())
    plt.colorbar(points, shrink=0.6, ax=ax[2])
    ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

    points = ax[3].scatter(X0[1:-1, 1:-1].flatten(), X1[1:-1, 1:-1].flatten(), marker='h',
                           c=Q_d[1:-1, 1:-1].flatten())
    plt.colorbar(points, shrink=0.6, ax=ax[3])
    ax[3].set_title('Q_d[1:-1,1:-1]')

    try:
        points = ax[4].scatter(X0[1:-1, 1:-1].flatten(), X1[1:-1, 1:-1].flatten(), marker='h',
                               c=Q_cbj[1:-1, 1:-1,1].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[4])
        ax[4].set_title('Q_cbj[1:-1,1:-1,1]')

        points = ax[5].scatter(X0[1:-1, 1:-1].flatten(), X1[1:-1, 1:-1].flatten(), marker='h',
                               c=Q_cj[1:-1, 1:-1, 1].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[5])
        ax[5].set_title('Q_cj[1:-1,1:-1,1]')
    except:
        pass
    plt.tight_layout()
    s1 = str(terrain) if terrain is None else terrain
    plt.savefig(os.path.join(save_path_png,'full_%03ix%03i_%s_%03i_thetar%0.0f.png' % (Nx, Ny, s1, i + 1, parameters['theta_r'])),
                bbox_inches='tight', pad_inches=0, dpi=240)
    plt.close('all')

    # Plot the 1D substates along the bottom of the channel
    fig = plt.figure(figsize=(10, 6))
    ax = [fig.add_subplot(2, 2, i, aspect='auto') for i in range(1, 5)]
    lnns1 = ax[0].plot(np.arange(len(ch_bot_thickness)), ch_bot_thickness, label='Q_th')
    ax[0].plot((5, 5), (0, 3), 'k-')
    ax[0].set_title('1D Q_th, time step = %03i' % (i + 1))
    ax[0].set_ylim([0, 3])
    ax[0].set_ylabel('Q_{th}')
    ax3 = ax[0].twinx()
    lnns2 = []
    for ii in range(Q_cj.shape[2]):
        lnns2.append(ax3.plot(np.arange(len(ch_bot_speed)), ch_bot_thickness_cons[ii], linestyle='--', label='Q_cbj[y,x,{0}]'.format(ii)))
    # lnns2 = ax3.plot(np.arange(len(ch_bot_speed)), ch_bot_thickness_cons0, color='tab:red', linestyle='--', label='Q_cbj[y,x,0]')
    # lnns3 = ax3.plot(np.arange(len(ch_bot_speed)), ch_bot_thickness_cons1, color='tab:cyan', linestyle='-.', label='Q_cbj[y,x,1]')
    lnns = lnns1
    for z in range(len(lnns2)):
        lnns += lnns2[z]
    labbs = [l.get_label() for l in lnns]
    ax[0].legend(lnns, labbs, loc=0)
    maxconc = np.max([np.amax(ch_bot_thickness_cons)])
    upper = (0.1 * (maxconc//0.1) + 0.1)
    ax3.set_ylim([0, upper])
    ax3.set_ylabel('Concentration')
    for xx in range(0, 3):
        ax[xx].set_xlabel('y: Channel axis')


    # plt.figure(figsize=(10, 6))
    ax[1].plot(np.arange(len(ch_bot_speed)), ch_bot_speed)
    ax[1].plot((5, 5), (0, 0.2), 'k-')
    ax[1].set_title('1D speed, time step = %03i' % (i + 1))
    ax[1].set_ylim([0, 0.2])
    ax[1].set_ylabel('Q_{v}')
    # plt.savefig('ch_bot_speed_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)

    # ax[2].figure(figsize=(10, 6))
    ax[2].plot(np.arange(len(ch_bot_speed)), ch_bot_outflow)
    ax[2].plot((5, 5), (0, 2), 'k-')
    ax[2].set_title('Sum 1D outflow, time step = %03i' % (i + 1))
    ax[2].set_ylim([0, 2])
    ax[2].set_ylabel('sum(Q_{o}[y,x])')

    lns1 = ax[3].plot(np.arange(len(ch_bot_speed)), ch_bot_sediment, label='Q_{d}')
    # ax[3].plot((5, 5), (0, 2), 'k-')
    ax[3].set_title('Sediment, time step = %03i' % (i + 1))
    # ax[3].set_ylim([0, 2])
    ax[3].set_ylabel('Q_{d}[y,x])')
    ax2 = ax[3].twinx()
    lns2 = []
    for ii in range(Q_cj.shape[2]):
        lns2.append(ax2.plot(np.arange(len(ch_bot_speed)), ch_bot_sediment_cons[ii], label='Q_cbj[y,x,{0}]'.format(ii)))
    # lns2 = ax2.plot(np.arange(len(ch_bot_speed)), ch_bot_sediment_cons0, color='tab:red', label='Q_cbj[y,x,0]')
    # lns3 = ax2.plot(np.arange(len(ch_bot_speed)), ch_bot_sediment_cons1, color='tab:cyan', label='Q_cbj[y,x,1]')
    lns = lns1
    for z in range(len(lns2)):
        lns += lns2[z]
    labs = [l.get_label() for l in lns]
    ax[3].legend(lns, labs, loc=0)
    ax2.set_ylim([0,1])
    ax2.set_ylabel('Concentration')
    plt.tight_layout()
    plt.savefig('./Data/mpi_combined_png/ch_bot_%03i.png' % (i + 1), bbox_inches='tight', pad_inches=0, dpi=240)

    plt.close('all')

def find_channel_bot(Q_a : np.ndarray):
    bot_indices = []
    Ny = Q_a.shape[0]
    for i in range(Ny):
        bot_indices.append((i,np.min(np.where(np.min(Q_a[i,:])==Q_a[i,:]))))
    return bot_indices

def gather_and_print_Qa_Qd(savename):
    # Check how Q_a and Q_d looks
    Image_Q_a = gather_grid(p_local_hexgrid.grid.Q_a)
    Image_Q_d = gather_grid(p_local_hexgrid.grid.Q_d)
    if my_rank == 0:
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(1, 2, i, aspect='equal') for i in range(1, 3)]
        # ind = np.unravel_index(np.argmax(Image_Q_a, axis=None), Image_Q_a.shape)

        points = ax[0].scatter(result_grid.grid.X[:, :, 0].flatten(), result_grid.grid.X[:, :, 1].flatten(),
                               marker='h',
                               c=Image_Q_a[:, :].flatten())

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_a[:,:]. ')
        points = ax[1].scatter(result_grid.grid.X[:, :, 0].flatten(), result_grid.grid.X[:, :, 1].flatten(),
                               marker='h',
                               c=Image_Q_d[:, :].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[1])

        plt.savefig(os.path.join('./Data/mpi_combined_png/', savename), bbox_inches='tight', pad_inches=0)

def generate_p_local_hex_bathymetry(parameters):
    my_mpi_row = parameters['row']
    my_mpi_col = parameters['col']
    terrain = parameters['terrain']
    p_local_grid_y_dim = parameters['']
    if terrain == 'rupert':
        global_bathy, junk = ma.generate_rupert_inlet_bathymetry(parameters['theta_r'],
                                                                 parameters['dx'],
                                                                 Ny=parameters['ny'],
                                                                 Nx=parameters['nx'])
        global_bathy = np.transpose(global_bathy)
    if terrain is None:
        global_bathy = np.zeros((parameters['ny'],parameters['nx']))

    local_bathy = global_bathy[(my_mpi_row*p_local_grid_y_dim):((my_mpi_row+1)*p_local_grid_y_dim),
                                 my_mpi_col*p_local_grid_x_dim:((my_mpi_col+1)*p_local_grid_x_dim)]
    return local_bathy

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
    elif num_procs % 2 == 0:
        if ny >= nx:
            p_x_dims = 2
            p_y_dims = num_procs/2.0
        else:
            p_x_dims = num_procs / 2.0
            p_y_dims = 2
    else:
        raise Exception("Please use an even or square number of processors!")
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



def set_p_local_hex_bathymetry(p_local_hexgrid, local_bathy):
    temp = p_local_hexgrid.Q_d[1:-1,1:-1] + local_bathy
    p_local_hexgrid.Q_a[1:-1,1:-1] = temp

def set_bc(p_local_hexgrid, local_bathy, parameters):
    my_mpi_col = parameters['col']
    my_mpi_row = parameters['row']
    p_y_dims = parameters['p_y_dims']
    p_x_dims = parameters['p_x_dims']
    if my_mpi_col == 0:
        p_local_hexgrid.Q_d[:,0] = np.inf
        p_local_hexgrid.Q_a[:,0] = np.inf
    if my_mpi_row == 0:
        p_local_hexgrid.Q_d[0,:] = np.inf
        p_local_hexgrid.Q_a[0,:] = np.inf
    if my_mpi_row == (p_y_dims-1):
        # print("mympirow == p_ydims. myrank = ", my_rank)
        p_local_hexgrid.Q_d[-1,:] = parameters['q_d[interior]']
        p_local_hexgrid.Q_a[-1,1:-1] = local_bathy[-1,:] + parameters['q_d[interior]']
    if my_mpi_col == (p_x_dims-1):
        # print("mympicol == p_xdims. myrank = ", my_rank)
        p_local_hexgrid.Q_d[:,-1] = np.inf
        p_local_hexgrid.Q_a[:,-1] = np.inf