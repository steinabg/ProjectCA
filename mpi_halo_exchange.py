from mpi4py import MPI
import numpy as np
import sys
import CAenvironment as CAenv
import scipy.io
import mathfunk as ma
import matplotlib.pyplot as plt

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


def exchange_borders_matrix(local_petri_A):


    # Send data south and receive from north
    comm.Sendrecv(
        [local_petri_A[-2, :], 1, border_row_t],  # send the second last row
        neighbor_processes[DOWN],
        0,
        [local_petri_A[0, :], 1, border_row_t],  # recvbuf = first row
        neighbor_processes[UP],
        0
    )

    # # Send data north and receive from south
    comm.Sendrecv(
        [local_petri_A[1, :], 1, border_row_t],  # sendbuf = second row (1.row = border)
        neighbor_processes[UP],  # destination
        1,  # sendtag
        [local_petri_A[-1, :], 1, border_row_t],  # recvbuf = last row
        neighbor_processes[DOWN],  # source
        1
    )
    #
    # Send west and receive from east
    local_grid_ev[:] = local_petri_A[:, 1].copy()
    comm.Sendrecv(
        [local_grid_ev, p_local_grid_y_dim + 2, MPI.DOUBLE],
        neighbor_processes[LEFT],
        2,
        [local_grid_wb, p_local_grid_y_dim + 2, MPI.DOUBLE],
        neighbor_processes[RIGHT],
        2
    )
    local_petri_A[:, -1] = local_grid_wb.copy()

    # # Send east and receive from west
    local_grid_wv[:] = local_petri_A[:, -2].copy()
    # print("my value = ", local_grid_wv)
    # print("my destination = ", local_grid_eb)

    comm.Sendrecv(
        [local_grid_wv, p_local_grid_y_dim + 2, MPI.DOUBLE],
        neighbor_processes[RIGHT],
        0,
        [local_grid_eb, p_local_grid_y_dim + 2, MPI.DOUBLE],
        neighbor_processes[LEFT],
        0
    )
    local_petri_A[:, 0] = local_grid_eb.copy()



def iterateCA():
    pass
    # p_local_hexgrid.grid.time_step(global_grid=False)

def gather_cube(local_petri_A, z_dim):
    ans = np.zeros((p_local_grid_y_dim * p_y_dims, p_local_grid_x_dim * p_x_dims, z_dim),
                   dtype=np.double, order='C')
    # temp = np.zeros((p_local_grid_y_dim * p_y_dims, p_local_grid_x_dim * p_x_dims),
    #                 dtype=np.double, order='C')
    for i in range(z_dim):
        ans[:,:,i] = gather_grid(local_petri_A[:,:,i])
    return ans

def gather_grid(local_petri_A):
    send = np.zeros((p_local_grid_y_dim, p_local_grid_x_dim), dtype=np.double)
    TEMP = np.zeros((p_local_grid_y_dim * p_y_dims * p_local_grid_x_dim * p_x_dims), dtype=np.double)
    IMAGE = np.zeros((p_local_grid_y_dim * p_y_dims * p_local_grid_x_dim * p_x_dims), dtype=np.double)


    send[:, :] = local_petri_A[1:-1, 1:-1]

    comm.Gather(send, TEMP, 0)
    # if rank == 0: print('TEMP = \n', TEMP.reshape(p_local_grid_y_dim*p_y_dims,p_local_grid_x_dim*p_x_dims))

    if my_rank == 0:

        Tempindex = 0
        imageXcounter = 1
        imageYcounter = 1
        for i in range(p_local_grid_y_dim * p_y_dims * p_local_grid_x_dim * p_x_dims):

            if ((i + 1) % (p_local_grid_x_dim) == 0):
                IMAGE[i] = TEMP[Tempindex]

                if (imageXcounter == (p_local_grid_x_dim * p_x_dims)):
                    if (imageYcounter == p_local_grid_y_dim):
                        Tempindex += 1
                        imageYcounter = 0
                    else:
                        Tempindex = Tempindex - ((p_x_dims - 1) * p_local_grid_x_dim * p_local_grid_y_dim) + 1
                    imageXcounter = 0
                    imageYcounter += 1
                else:
                    Tempindex += (p_local_grid_x_dim * p_local_grid_y_dim) - p_local_grid_x_dim + 1
            else:
                IMAGE[i] = TEMP[Tempindex]
                Tempindex += 1

            imageXcounter += 1

    IMAGE = IMAGE.reshape(p_local_grid_y_dim * p_y_dims, p_local_grid_x_dim * p_x_dims)
    return IMAGE

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
    return tuple([local_x, local_y])

def set_local_grid_source_xy():
    p_local_source_tiles = []
    p_local_source_tile_x = []
    p_local_source_tile_y = []
    x_coords_array = parameters['x']
    y_coords_array = parameters['y']
    no_x_coords = np.size(x_coords_array)
    no_y_coords = np.size(y_coords_array)


    for i in range(no_x_coords):
        for j in range(no_y_coords):
            if (no_x_coords == 1):
                if(no_y_coords == 1):
                    # print("h1")
                    # print(x_coords_array,y_coords_array)
                    local_x, local_y = global_coords_to_local_coords(y_coords_array,x_coords_array,my_mpi_row,
                                                                     my_mpi_col,p_local_grid_x_dim,
                                                                     p_local_grid_y_dim)
                    if(local_x >-1):
                        p_local_source_tiles.append(tuple([local_y,local_x]))
                        p_local_source_tile_x.append(local_x)
                        p_local_source_tile_y.append(local_y)


                else:
                    # print("h2")
                    # print(x_coords_array,y_coords_array[0][j-1])
                    local_x, local_y = global_coords_to_local_coords(y_coords_array[0][j-1], x_coords_array,
                                                                     my_mpi_row, my_mpi_col,
                                                                     p_local_grid_x_dim, p_local_grid_y_dim)
                    if(local_x >-1):
                        p_local_source_tiles.append(tuple([local_y,local_x]))
                        p_local_source_tile_x.append(local_x)
                        p_local_source_tile_y.append(local_y)
            else:
                if(no_y_coords == 1):
                    # print("h3")
                    # print(x_coords_array[0][i-1],y_coords_array)
                    local_x, local_y = global_coords_to_local_coords(y_coords_array, x_coords_array[0][i-1],
                                                                     my_mpi_row, my_mpi_col,
                                                                     p_local_grid_x_dim, p_local_grid_y_dim)
                    if(local_x >-1):
                        p_local_source_tiles.append(tuple([local_y,local_x]))
                        p_local_source_tile_x.append(local_x)
                        p_local_source_tile_y.append(local_y)
                else:
                    # print("h4")
                    # print(x_coords_array[0][i-1],y_coords_array[0][j-1])
                    local_x, local_y = global_coords_to_local_coords(y_coords_array[0][j-1], x_coords_array[0][i-1],
                                                                     my_mpi_row, my_mpi_col,
                                                                     p_local_grid_x_dim, p_local_grid_y_dim)
                    if(local_x >-1):
                        p_local_source_tiles.append(tuple([local_y,local_x]))
                        p_local_source_tile_x.append(local_x)
                        p_local_source_tile_y.append(local_y)

    # print(p_local_source_tiles)
    p_local_source_tiles = [x for x in p_local_source_tiles if x != tuple([-1, -1])]
    if len(p_local_source_tiles) > 0:
        p_local_grid_parameters['x'] = np.ix_(np.arange(np.min(p_local_source_tile_x),np.max(p_local_source_tile_x)+1))
        p_local_grid_parameters['y'] = np.ix_(np.arange(np.min(p_local_source_tile_y),np.max(p_local_source_tile_y)+1))
        print("p_local_grid_parameters['x'] = {0}, p_local_grid_parameters['y'] = {1}".format(
            p_local_grid_parameters['x'], p_local_grid_parameters['y']
        ))
    # print(p_local_grid_parameters['x'])
    # print(p_local_grid_parameters['y'])

def load_txt_files(num_iterations):
    IMAGE_Q_th = np.loadtxt(os.path.join(save_path_txt, 'Q_th_{0}.txt'.format(num_iterations + 1)))
    IMAGE_Q_cbj = np.loadtxt(os.path.join(save_path_txt, 'Q_cbj_{0}.txt'.format(num_iterations + 1)))
    IMAGE_Q_cj = np.loadtxt(os.path.join(save_path_txt, 'Q_cj_{0}.txt'.format(num_iterations + 1)))
    IMAGE_Q_d = np.loadtxt(os.path.join(save_path_txt, 'Q_d_{0}.txt'.format(num_iterations + 1)))
    return IMAGE_Q_th, IMAGE_Q_cbj, IMAGE_Q_cj, IMAGE_Q_d

def print_substate(Ny, Nx, i, Q_th, Q_cj, Q_cbj, Q_d, X0, X1, terrain):
    fig = plt.figure(figsize=(10, 6))
    ax = [fig.add_subplot(2, 2, i, aspect='equal') for i in range(1, 5)]
    ind = np.unravel_index(np.argmax(Q_th, axis=None), Q_th.shape)

    points = ax[0].scatter(X0.flatten(), X1.flatten(), marker='h',
                           c=Q_cj[:, :].flatten())

    plt.colorbar(points, shrink=0.6, ax=ax[0])
    ax[0].set_title('Q_cj[:,:,0]. n = ' + str(i + 1))

    points = ax[1].scatter(X0.flatten(), X1.flatten(), marker='h',
                           c=Q_th.flatten())
    # ax[1].scatter(X[ind[0],ind[1],0], X[ind[0],ind[1],1], c='r')  # Targeting
    plt.colorbar(points, shrink=0.6, ax=ax[1])
    ax[1].set_title('Q_th')

    points = ax[2].scatter(X0[1:-1, 1:-1].flatten(), X1[1:-1, 1:-1].flatten(), marker='h',
                           c=Q_cbj[1:-1, 1:-1].flatten())
    plt.colorbar(points, shrink=0.6, ax=ax[2])
    ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

    points = ax[3].scatter(X0[1:-1, 1:-1].flatten(), X1[1:-1, 1:-1].flatten(), marker='h',
                           c=Q_d[1:-1, 1:-1].flatten())
    plt.colorbar(points, shrink=0.6, ax=ax[3])
    ax[3].set_title('Q_d[1:-1,1:-1]')
    plt.tight_layout()
    s1 = str(terrain) if terrain is None else terrain
    plt.savefig(os.path.join(save_path_png,'full_%03ix%03i_%s_%03i_thetar%0.0f.png' % (Nx, Ny, s1, i + 1, parameters['theta_r'])),
                bbox_inches='tight', pad_inches=0, dpi=240)
    plt.close('all')

if __name__ == "__main__":
    # Setup MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    num_procs = comm.Get_size()

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    neighbor_processes = [0, 0, 0, 0]



    # Load CA parameters from file
    parameters = CAenv.import_parameters()

    if my_rank is 0:
        result_grid = CAenv.CAenvironment(parameters)
        # print("size = ", sys.getsizeof(result_grid))

    IMG_X = parameters['nx']
    IMG_Y = parameters['ny']





    ITERATIONS = parameters['num_iterations']


    p_y_dims = int(np.sqrt(num_procs))
    p_x_dims = int(np.sqrt(num_procs))

    cartesian_communicator = comm.Create_cart((p_y_dims, p_x_dims), periods=(False, False))

    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)

    neighbor_processes[UP], neighbor_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbor_processes[LEFT], neighbor_processes[RIGHT] = cartesian_communicator.Shift(1, 1)



    p_local_grid_x_dim = int(IMG_X / p_x_dims)  # Må være delelig
    p_local_grid_y_dim = int(IMG_Y / p_y_dims)

    p_local_grid_parameters = parameters.copy()
    p_local_grid_parameters['x'] = None  # if None == no source
    p_local_grid_parameters['y'] = None  # if None == no source
    p_local_grid_parameters['nx'] = p_local_grid_x_dim + 2  # make room for borders
    p_local_grid_parameters['ny'] = p_local_grid_y_dim + 2  # make room for borders

    set_local_grid_source_xy()

    def generate_p_local_hex_bathymetry(terrain):
        if terrain == 'rupert':
            global_bathy, junk = ma.generate_rupert_inlet_bathymetry(parameters['theta_r'],
                                                                     Ny=parameters['ny'],
                                                                     Nx=parameters['nx'])
            global_bathy = np.transpose(global_bathy)
            # global_bathy = np.ones((parameters['ny'],parameters['nx'])) # TODO: Remove after testing
            local_bathy = global_bathy[(my_mpi_row*p_local_grid_y_dim):((my_mpi_row+1)*p_local_grid_y_dim),
                                         my_mpi_col*p_local_grid_x_dim:((my_mpi_col+1)*p_local_grid_x_dim)]
            return local_bathy
    local_bathy = generate_p_local_hex_bathymetry(parameters['terrain'])

    p_local_hexgrid = CAenv.CAenvironment(p_local_grid_parameters, global_grid=False)
    # print("rank= ", my_rank, " np.where(p_local_hexgrid.grid.Q_v>0): ", np.where(p_local_hexgrid.grid.Q_v>0))
    def set_p_local_hex_bathymetry(p_local_hexgrid, local_bathy):
        temp = p_local_hexgrid.grid.Q_d[1:-1,1:-1] + local_bathy
        p_local_hexgrid.grid.Q_a[1:-1,1:-1] = temp

    set_p_local_hex_bathymetry(p_local_hexgrid, local_bathy)
    # print("my rank = {0}, my_mpi_col = {1}, my_mpi_row = {2}".format(my_rank, my_mpi_col,my_mpi_row))
    # if my_rank == 0: print("p_xdims = {0}, p_y_dims = {1}".format(p_x_dims,p_y_dims))

    def set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims):
        if my_mpi_col == 0:
            p_local_hexgrid.grid.Q_d[:,0] = np.inf
            p_local_hexgrid.grid.Q_a[:,0] = np.inf
        if my_mpi_row == 0:
            p_local_hexgrid.grid.Q_d[0,:] = np.inf
            p_local_hexgrid.grid.Q_a[0,:] = np.inf
        if my_mpi_row == (p_y_dims-1):
            # print("mympirow == p_ydims. myrank = ", my_rank)
            p_local_hexgrid.grid.Q_d[-1,:] = np.inf
            p_local_hexgrid.grid.Q_a[-1,:] = np.inf
        if my_mpi_col == (p_x_dims-1):
            # print("mympicol == p_xdims. myrank = ", my_rank)
            p_local_hexgrid.grid.Q_d[:,-1] = np.inf
            p_local_hexgrid.grid.Q_a[:,-1] = np.inf
    set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims)
    p_local_hexgrid.grid.my_rank = my_rank

    # print("my rank = {0}\n my local Q_d = \n{1}".format(my_rank,p_local_hexgrid.grid.Q_d))
    # print("my rank = {0}\nmy local Q_d =\n{1}".format(my_rank, p_local_hexgrid.grid.Q_d))

    # local_petri_A = np.zeros((p_local_grid_x_dim + 2, p_local_grid_y_dim + 2))

    local_grid_wb = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_eb = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_ev = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_wv = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')

    # p_local_hexgrid.grid.Q_o[1,1,1] = my_rank
    # p_local_hexgrid.grid.Q_o[1,p_local_grid_x_dim,1] = my_rank
    # p_local_hexgrid.grid.Q_o[p_local_grid_y_dim,1,1] = my_rank
    # p_local_hexgrid.grid.Q_o[p_local_grid_y_dim,p_local_grid_x_dim,1] = my_rank
    # local_petri_A[:] = comm.Get_rank()
    # print("process = ", comm.rank,"\n",
    #       local_grid_A)

    border_row_t = MPI.DOUBLE.Create_vector(p_local_grid_x_dim + 2,
                                            1,
                                            1)
    border_row_t.Commit()

    border_col_t = MPI.DOUBLE.Create_vector(p_local_grid_y_dim + 2,
                                            1,
                                            p_local_grid_x_dim + 2)
    border_col_t.Commit()
    sample_rate = parameters['sample_rate']
    i_sample_values = []

    import os.path
    save_path_txt = './Data/mpi_combined_txt'
    save_path_png = './Data/mpi_combined_png'
    if my_rank == 0:
        np.savetxt(os.path.join(save_path_txt, 'X000.txt'), result_grid.grid.X[:, :, 0])
        np.savetxt(os.path.join(save_path_txt, 'X001.txt'), result_grid.grid.X[:, :, 1])

    q_th0 = parameters['q_th[y,x]']
    q_cj0 = parameters['q_cj[y,x,0]']
    q_v0 = parameters['q_v[y,x]']

    # Check how Q_a and Q_d looks
    Image_Q_a = gather_grid(p_local_hexgrid.grid.Q_a)
    Image_Q_d = gather_grid(p_local_hexgrid.grid.Q_d)
    if my_rank == 0:
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(1, 2, i, aspect='equal') for i in range(1, 3)]
        # ind = np.unravel_index(np.argmax(Image_Q_a, axis=None), Image_Q_a.shape)

        points = ax[0].scatter(result_grid.grid.X[:, :, 0].flatten(), result_grid.grid.X[:, :, 1].flatten(), marker='h',
                               c=Image_Q_a[:, :].flatten())

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_a[:,:]. ')
        points = ax[1].scatter(result_grid.grid.X[:, :, 0].flatten(), result_grid.grid.X[:, :, 1].flatten(), marker='h',
                               c=Image_Q_d[:, :].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[1])

        plt.savefig(os.path.join(save_path_png,'before.png'),bbox_inches='tight', pad_inches=0)
    if my_rank== 0:
        from timeit import default_timer as timer
        start = timer()
    # print("my_rank = {0}, parameters[x] = {1}, parameters[y] = {2}".format(my_rank,p_local_grid_parameters['x'],p_local_grid_parameters['y']))
    for num_iterations in range(ITERATIONS):
        # Add source
        p_local_hexgrid.addSource(q_th0,q_v0, q_cj0)

        # Exchange borders
        exchange_borders_matrix(p_local_hexgrid.grid.Q_th)
        exchange_borders_matrix(p_local_hexgrid.grid.Q_v)
        exchange_borders_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
        exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
        exchange_borders_matrix(p_local_hexgrid.grid.Q_d)
        exchange_borders_matrix(p_local_hexgrid.grid.Q_a)
        exchange_borders_cube(p_local_hexgrid.grid.Q_o, 6)
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims)


        # Calculate time step and set common dt in all local grids
        p_global_dt = np.zeros((1),dtype=np.double, order='C')
        p_local_dt = np.array(p_local_hexgrid.grid.calc_dt(global_grid=False),
                              dtype=np.double, order='C')
        comm.barrier()
        comm.Allreduce(p_local_dt, p_global_dt, op=MPI.MIN)
        p_global_dt = p_global_dt[0]
        # print("my rank = {0} and dt = {1}. I received {2}".format(my_rank, p_local_dt, p_global_dt))
        p_local_hexgrid.grid.dt = p_global_dt # Set dt



        # Iterate CA
        p_local_hexgrid.grid.time_step(global_grid=False)

        # Debugging
        if (num_iterations == 0):
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

                plt.savefig(os.path.join(save_path_png,'after.png'), bbox_inches='tight', pad_inches=0)

        if ((num_iterations + 1) % sample_rate == 0) and num_iterations > 0:
            i_sample_values.append(num_iterations)
            # print("sample")
            # Gather grids
            IMAGE_Q_th = gather_grid(p_local_hexgrid.grid.Q_th)
            IMAGE_Q_cbj = gather_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
            IMAGE_Q_cj = gather_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
            IMAGE_Q_d = gather_grid(p_local_hexgrid.grid.Q_d)

            if my_rank== 0:
                np.savetxt(os.path.join(save_path_txt, 'Q_th_{0}.txt'.format(num_iterations + 1)), IMAGE_Q_th)
                np.savetxt(os.path.join(save_path_txt, 'Q_cbj_{0}.txt'.format(num_iterations + 1)), IMAGE_Q_cbj[:, :, 0])
                np.savetxt(os.path.join(save_path_txt, 'Q_cj_{0}.txt'.format(num_iterations + 1)), IMAGE_Q_cj[:, :, 0])
                np.savetxt(os.path.join(save_path_txt, 'Q_d_{0}.txt'.format(num_iterations + 1)), IMAGE_Q_d)

        #
        #
        #
        #
        #     CAenv.sampleValues()
        #     CAenv.printSubstates(i)
    # print("AFTER\nmy rank = {0}\nmy local Q_d =\n{1}".format(my_rank,p_local_hexgrid.grid.Q_d))

    comm.barrier()

    # Gather the results
    IMAGE_Q_th = gather_grid(p_local_hexgrid.grid.Q_th)
    IMAGE_Q_v = gather_grid(p_local_hexgrid.grid.Q_v)
    IMAGE_Q_cbj = gather_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
    IMAGE_Q_cj = gather_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
    IMAGE_Q_d = gather_grid(p_local_hexgrid.grid.Q_d)
    IMAGE_Q_a = gather_grid(p_local_hexgrid.grid.Q_a)
    IMAGE_Q_o = gather_cube(p_local_hexgrid.grid.Q_o, 6)


    # Print figures
    num_figs = ITERATIONS//sample_rate
    figs_per_proc = num_figs//num_procs
    X0 = np.loadtxt(os.path.join(save_path_txt, 'X000.txt'))
    X1 = np.loadtxt(os.path.join(save_path_txt, 'X001.txt'))

    for i in range(my_rank*figs_per_proc,(my_rank+1)*figs_per_proc):
        IMAGE_Q_th, IMAGE_Q_cbj, IMAGE_Q_cj, IMAGE_Q_d = load_txt_files(i_sample_values[i])
        print_substate(parameters['ny'],parameters['nx'],i_sample_values[i],
                       IMAGE_Q_th, IMAGE_Q_cj, IMAGE_Q_cbj, IMAGE_Q_d,
                       X0, X1, parameters['terrain'])


if my_rank == 0:
    print("time = ", timer() - start)


    # if my_rank == 0: print(IMAGE_Q_o[:,:,0])



