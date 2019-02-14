from mpi4py import MPI
import numpy as np
import sys
import CAenvironment as CAenv

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
    return tuple([local_y, local_x])

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
    # print(p_local_grid_parameters['x'])
    # print(p_local_grid_parameters['y'])


if __name__ == "__main__":
    # Setup MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    num_procs = comm.Get_size()

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NORTH_EAST = 4
    SOUTH_WEST = 5

    neighbor_processes = [0, 0, 0, 0, 0, 0]



    # Load CA parameters from file
    parameters = CAenv.import_parameters()

    if my_rank is 0:
        result_grid = CAenv.CAenvironment(parameters)

    IMG_X = parameters['nx']
    IMG_Y = parameters['ny']





    ITERATIONS = parameters['num_iterations']


    p_y_dims = int(np.sqrt(num_procs))
    p_x_dims = int(np.sqrt(num_procs))

    cartesian_communicator = comm.Create_cart((p_y_dims, p_x_dims), periods=(False, False))

    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)

    neighbor_processes[UP], neighbor_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbor_processes[LEFT], neighbor_processes[RIGHT] = cartesian_communicator.Shift(1, 1)
    neighbor_processes[NORTH_EAST], neighbor_processes[SOUTH_WEST] = cartesian_communicator.Shift(direction=(1,1), disp=1)



    p_local_grid_x_dim = int(IMG_X / p_x_dims)  # Må være delelig
    p_local_grid_y_dim = int(IMG_Y / p_y_dims)

    p_local_grid_parameters = parameters.copy()
    p_local_grid_parameters['x'] = None  # if None == no source
    p_local_grid_parameters['y'] = None  # if None == no source
    p_local_grid_parameters['nx'] = p_local_grid_x_dim + 2  # make room for borders
    p_local_grid_parameters['ny'] = p_local_grid_y_dim + 2  # make room for borders







    # if my_rank is 0:
    set_local_grid_source_xy()

    # if my_rank is 0:
    p_local_hexgrid = CAenv.CAenvironment(p_local_grid_parameters, global_grid=False)
    print("rank= ", my_rank, " np.where(p_local_hexgrid.grid.Q_th>0): ", np.where(p_local_hexgrid.grid.Q_th>0))




    # local_petri_A = np.zeros((p_local_grid_x_dim + 2, p_local_grid_y_dim + 2))

    local_grid_wb = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_eb = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_ev = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')
    local_grid_wv = np.zeros((p_local_grid_y_dim + 2), dtype=np.double, order='C')

    p_local_hexgrid.grid.Q_o[:,:,1] = my_rank
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

    for num_iterations in range(ITERATIONS):
        exchange_borders_matrix(p_local_hexgrid.grid.Q_th)
        exchange_borders_matrix(p_local_hexgrid.grid.Q_v)
        for j in range(parameters['nj']):
            exchange_borders_matrix(p_local_hexgrid.grid.Q_cj[:, :, j])
            exchange_borders_matrix(p_local_hexgrid.grid.Q_cbj[:, :, j])
        exchange_borders_matrix(p_local_hexgrid.grid.Q_d)
        exchange_borders_matrix(p_local_hexgrid.grid.Q_a)
        exchange_borders_cube(p_local_hexgrid.grid.Q_o, 6)
        iterateCA()

    comm.barrier()
    IMAGE_Q_th = gather_grid(p_local_hexgrid.grid.Q_th)
    test = p_local_hexgrid.grid.Q_o[:,:,1].copy()
    IMAGE_Q_o = gather_grid(test)
    if my_rank == 0: print (IMAGE_Q_o)

    print("after \nprocess = ", my_rank,"\n",
          p_local_hexgrid.grid.Q_o[:,:,1])
