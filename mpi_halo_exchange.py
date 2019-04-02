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
    return tuple([local_x + 1, local_y + 1]) # Add 1 because of the borders

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
        # print("p_local_grid_parameters['x'] = {0}, p_local_grid_parameters['y'] = {1}".format(
        #     p_local_grid_parameters['x'], p_local_grid_parameters['y']
        # ))
    # print(p_local_grid_parameters['x'])
    # print(p_local_grid_parameters['y'])

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


    # TODO: Fix toppling rule behaviour for MPI
    # Load CA parameters from file
    parameters = CAenv.import_parameters()

    if my_rank is 0:
        # import pandas as pd
        #
        # desired_width = 320
        # pd.set_option('display.width', desired_width)
        # np.set_printoptions(linewidth=desired_width)
        result_grid = CAenv.CAenvironment(parameters)
        # print("size = ", sys.getsizeof(result_grid))

    IMG_X = parameters['nx']
    IMG_Y = parameters['ny']





    ITERATIONS = parameters['num_iterations']


    p_y_dims = int(np.sqrt(num_procs)) # num_procs must be a square number
    p_x_dims = int(np.sqrt(num_procs))

    cartesian_communicator = comm.Create_cart((p_y_dims, p_x_dims), periods=(False, False))

    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)

    neighbor_processes[UP], neighbor_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbor_processes[LEFT], neighbor_processes[RIGHT] = cartesian_communicator.Shift(1, 1)



    p_local_grid_x_dim = int(IMG_X / p_x_dims)  # IMG_X must be divisible by p_x_dims
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
                                                                     parameters['dx'],
                                                                     Ny=parameters['ny'],
                                                                     Nx=parameters['nx'])
            global_bathy = np.transpose(global_bathy)
            local_bathy = global_bathy[(my_mpi_row*p_local_grid_y_dim):((my_mpi_row+1)*p_local_grid_y_dim),
                                         my_mpi_col*p_local_grid_x_dim:((my_mpi_col+1)*p_local_grid_x_dim)]
            return local_bathy
    local_bathy = generate_p_local_hex_bathymetry(parameters['terrain'])

    p_local_hexgrid = CAenv.CAenvironment(p_local_grid_parameters, global_grid=False)
    # print("rank= ", my_rank, " np.where(p_local_hexgrid.grid.Q_th>0): ", np.where(p_local_hexgrid.grid.Q_th>0))
    def set_p_local_hex_bathymetry(p_local_hexgrid, local_bathy):
        temp = p_local_hexgrid.grid.Q_d[1:-1,1:-1] + local_bathy
        p_local_hexgrid.grid.Q_a[1:-1,1:-1] = temp

    set_p_local_hex_bathymetry(p_local_hexgrid, local_bathy)
    # print("my rank = {0}, my_mpi_col = {1}, my_mpi_row = {2}".format(my_rank, my_mpi_col,my_mpi_row))
    # if my_rank == 0: print("p_xdims = {0}, p_y_dims = {1}".format(p_x_dims,p_y_dims))

    def set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters):
        if my_mpi_col == 0:
            p_local_hexgrid.grid.Q_d[:,0] = np.inf
            p_local_hexgrid.grid.Q_a[:,0] = np.inf
        if my_mpi_row == 0:
            p_local_hexgrid.grid.Q_d[0,:] = np.inf
            p_local_hexgrid.grid.Q_a[0,:] = np.inf
        if my_mpi_row == (p_y_dims-1):
            # print("mympirow == p_ydims. myrank = ", my_rank)
            p_local_hexgrid.grid.Q_d[-1,:] = parameters['q_d[interior]']
            p_local_hexgrid.grid.Q_a[-1,1:-1] = local_bathy[-1,:] + parameters['q_d[interior]']
        if my_mpi_col == (p_x_dims-1):
            # print("mympicol == p_xdims. myrank = ", my_rank)
            p_local_hexgrid.grid.Q_d[:,-1] = np.inf
            p_local_hexgrid.grid.Q_a[:,-1] = np.inf
    set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)
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
        np.save(os.path.join(save_path_txt, 'X000'), result_grid.grid.X[:, :, 0])
        np.save(os.path.join(save_path_txt, 'X001'), result_grid.grid.X[:, :, 1])

    q_th0 = parameters['q_th[y,x]']
    q_cj0 = parameters['q_cj[y,x,0]']
    q_v0 = parameters['q_v[y,x]']

    # Check how Q_a and Q_d looks
    # gather_and_print_Qa_Qd('before.png')


    image_q_a = gather_grid(p_local_hexgrid.grid.Q_a)
    if my_rank== 0:
        bottom_indices = find_channel_bot(image_q_a)

        from timeit import default_timer as timer

        save_dt = []
        start = timer()

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
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)


        # Calculate time step and set common dt in all local grids
        p_global_dt = np.zeros((1),dtype=np.double, order='C')
        p_local_dt = np.array(p_local_hexgrid.grid.calc_dt(global_grid=False),
                              dtype=np.double, order='C')
        comm.barrier()
        comm.Allreduce(p_local_dt, p_global_dt, op=MPI.MIN)
        p_global_dt = p_global_dt[0]
        p_local_hexgrid.grid.dt = p_global_dt # Set dt
        if my_rank == 0:
            save_dt.append(p_global_dt)

        # Iterate CA
        # Can't use p_local_hexgrid.grid.time_step(global_grid=False).
        # Must exchange borders between each iteraction/transform
        p_local_hexgrid.grid.T_1()
        p_local_hexgrid.grid.sanityCheck()
        exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
        exchange_borders_matrix(p_local_hexgrid.grid.Q_th)
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)
        p_local_hexgrid.grid.T_2()
        p_local_hexgrid.grid.sanityCheck()
        exchange_borders_matrix(p_local_hexgrid.grid.Q_d)
        exchange_borders_matrix(p_local_hexgrid.grid.Q_a)
        exchange_borders_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
        exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)
        p_local_hexgrid.grid.I_1()
        p_local_hexgrid.grid.sanityCheck()
        exchange_borders_cube(p_local_hexgrid.grid.Q_o, 6)
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)
        p_local_hexgrid.grid.I_2()
        p_local_hexgrid.grid.sanityCheck()
        exchange_borders_matrix(p_local_hexgrid.grid.Q_th)
        exchange_borders_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)
        p_local_hexgrid.grid.I_3()
        p_local_hexgrid.grid.sanityCheck()
        exchange_borders_matrix(p_local_hexgrid.grid.Q_v)
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)
        p_local_hexgrid.grid.I_4()
        p_local_hexgrid.grid.sanityCheck()
        exchange_borders_matrix(p_local_hexgrid.grid.Q_d)
        exchange_borders_matrix(p_local_hexgrid.grid.Q_a)
        exchange_borders_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
        set_p_local_hex_boundary_conditions(p_local_hexgrid, my_mpi_col, my_mpi_row, p_y_dims, p_x_dims, local_bathy, parameters)

        # Debugging
        # image_Q_th = gather_grid(p_local_hexgrid.grid.Q_th)
        # if my_rank == 0:
        #     # Check if the mpi and nonmpi grids are identical
        #     result_grid.addSource(q_th0, q_v0, q_cj0)
        #     result_grid.CAtimeStep()
        #     print("image = \n", image_Q_th)
        #     print("\nresult_grid = \n", result_grid.grid.Q_th)
        #     print("image_dt = {0}, result_grid.dt = {1}. equal = {2}".format(p_global_dt,result_grid.grid.dt, p_global_dt == result_grid.grid.dt))
        #     # result_grid.grid.Q_th[1,1] += 1 #induce intentional error
        #
        #     if np.any(image_Q_th.flatten() != result_grid.grid.Q_th.flatten()):
        #         ii, jj = np.where(image_Q_th != result_grid.grid.Q_th)
        #         s = "num_iterations = 0\n"
        #         # s = s + ''.join("local_source_tiles = {0}, {1}\n. result_grid_source = {2},{3}.\n".format(
        #         #     p_local_grid_parameters['x'], p_local_grid_parameters['y'], parameters['x'], parameters['y']
        #         # ))
        #         s = s + ''.join("image_q_th[{0},{1}] = {2} != result_grid.grid.Q_th[{3},{4}] = {5}\n".format(
        #             ii[x], jj[x], image_Q_th[ii[x], jj[x]], ii[x], jj[x], result_grid.grid.Q_th[ii[x], jj[x]]) for x in
        #                         range(len(ii)))
        #         raise Exception("not equal: " + s)
        #     print("all clear!")


        # if (num_iterations == 0):
        #     gather_and_print_Qa_Qd('after_func.png')

        # Sample and print sub states to .txt
        if ((num_iterations + 1) % sample_rate == 0) and num_iterations > 0:
            i_sample_values.append(num_iterations)
            # print("sample")
            # Gather grids
            IMAGE_Q_th = gather_grid(p_local_hexgrid.grid.Q_th)
            IMAGE_Q_cbj = gather_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
            IMAGE_Q_cj = gather_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
            IMAGE_Q_d = gather_grid(p_local_hexgrid.grid.Q_d)
            IMAGE_Q_v = gather_grid(p_local_hexgrid.grid.Q_v)
            IMAGE_Q_o = gather_cube(p_local_hexgrid.grid.Q_o,6)

            if my_rank== 0:
                ch_bot_thickness = [IMAGE_Q_th[bottom_indices[i]] for i in
                                    range(len(bottom_indices))]
                ch_bot_speed = [IMAGE_Q_v[bottom_indices[i]] for i in range(len(bottom_indices))]
                ch_bot_thickness_cons = []
                ch_bot_sediment_cons = []
                for jj in range(parameters['nj']):
                    ch_bot_thickness_cons.append([IMAGE_Q_cj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
                    ch_bot_sediment_cons.append([IMAGE_Q_cbj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
                # ch_bot_thickness_cons0 = [IMAGE_Q_cj[bottom_indices[i] + (0,)] for i in range(len(bottom_indices))]
                # ch_bot_thickness_cons1 = [IMAGE_Q_cj[bottom_indices[i] + (1,)] for i in range(len(bottom_indices))]
                ch_bot_sediment = [IMAGE_Q_d[bottom_indices[i]] for i in range(len(bottom_indices))]
                # ch_bot_sediment_cons0 = [IMAGE_Q_cbj[bottom_indices[i] + (0,)] for i in range(len(bottom_indices))]
                # ch_bot_sediment_cons1 = [IMAGE_Q_cbj[bottom_indices[i] + (1,)] for i in range(len(bottom_indices))]
                ch_bot_outflow = [sum(IMAGE_Q_o[bottom_indices[i]]) for i in
                                  range(len(bottom_indices))]

                np.save(os.path.join(save_path_txt, 'Q_th_{0}'.format(num_iterations + 1)), IMAGE_Q_th)
                np.save(os.path.join(save_path_txt, 'Q_cbj_{0}'.format(num_iterations + 1)), IMAGE_Q_cbj)
                np.save(os.path.join(save_path_txt, 'Q_cj_{0}'.format(num_iterations + 1)), IMAGE_Q_cj)
                np.save(os.path.join(save_path_txt, 'Q_d_{0}'.format(num_iterations + 1)), IMAGE_Q_d)
                np.save(os.path.join(save_path_txt, 'ch_bot_outflow_{0}'.format(num_iterations + 1)), ch_bot_outflow)
                np.save(os.path.join(save_path_txt, 'ch_bot_speed_{0}'.format(num_iterations + 1)), ch_bot_speed)
                np.save(os.path.join(save_path_txt, 'ch_bot_thickness_{0}'.format(num_iterations + 1)), ch_bot_thickness)
                np.save(os.path.join(save_path_txt, 'ch_bot_sediment_{0}'.format(num_iterations + 1)), ch_bot_sediment)
                for jj in range(parameters['nj']):
                    np.save(os.path.join(save_path_txt, 'ch_bot_sediment_cons{0}_{1}'.format(jj,num_iterations + 1)),
                            ch_bot_sediment_cons[jj])
                    np.save(os.path.join(save_path_txt, 'ch_bot_thickness_cons{0}_{1}'.format(jj,num_iterations + 1)),
                            ch_bot_thickness_cons[jj])
                # np.save(os.path.join(save_path_txt, 'ch_bot_sediment_cons0_{0}'.format(num_iterations + 1)), ch_bot_sediment_cons0)
                # np.save(os.path.join(save_path_txt, 'ch_bot_sediment_cons1_{0}'.format(num_iterations + 1)), ch_bot_sediment_cons1)
                # np.save(os.path.join(save_path_txt, 'ch_bot_thickness_cons0_{0}'.format(num_iterations + 1)), ch_bot_thickness_cons0)
                # np.save(os.path.join(save_path_txt, 'ch_bot_thickness_cons1_{0}'.format(num_iterations + 1)), ch_bot_thickness_cons1)




    # Gather the results
    # IMAGE_Q_th = gather_grid(p_local_hexgrid.grid.Q_th)
    # IMAGE_Q_v = gather_grid(p_local_hexgrid.grid.Q_v)
    # IMAGE_Q_cbj = gather_cube(p_local_hexgrid.grid.Q_cbj, parameters['nj'])
    # IMAGE_Q_cj = gather_cube(p_local_hexgrid.grid.Q_cj, parameters['nj'])
    # IMAGE_Q_d = gather_grid(p_local_hexgrid.grid.Q_d)
    # IMAGE_Q_a = gather_grid(p_local_hexgrid.grid.Q_a)
    # IMAGE_Q_o = gather_cube(p_local_hexgrid.grid.Q_o, 6)


    # Print figures
    num_figs = ITERATIONS//sample_rate
    figs_per_proc = num_figs//num_procs
    X0 = np.load(os.path.join(save_path_txt, 'X000.npy'))
    X1 = np.load(os.path.join(save_path_txt, 'X001.npy'))

    upper_lim = (my_rank+1)*figs_per_proc
    if my_rank == (num_procs-1) and (upper_lim-1) < num_figs:
        # print("rank ={0}, upperlim = {1}, numprocs = {2}".format(my_rank,upper_lim, num_procs))
        upper_lim = num_figs
        # print(upper_lim)
    for i in range(my_rank*figs_per_proc,upper_lim):
        IMAGE_Q_th, IMAGE_Q_cbj, IMAGE_Q_cj, IMAGE_Q_d,\
        ch_bot_outflow, ch_bot_thickness, ch_bot_speed,\
        ch_bot_sediment, ch_bot_sediment_cons, ch_bot_thickness_cons =\
            load_txt_files(i_sample_values[i], parameters)
        print_substate(parameters['ny'],parameters['nx'],i_sample_values[i],
                       IMAGE_Q_th, IMAGE_Q_cj, IMAGE_Q_cbj, IMAGE_Q_d,
                       X0, X1, parameters['terrain'], ch_bot_thickness, ch_bot_speed, ch_bot_outflow,
                       ch_bot_sediment, ch_bot_sediment_cons, ch_bot_thickness_cons)

comm.barrier()
if my_rank == 0:
    print("time = ", timer() - start)
    plt.figure()
    plt.plot(np.arange(ITERATIONS),save_dt)
    plt.xlabel('iterations')
    plt.ylabel('dt')
    plt.savefig('./Data/mpi_combined_png/dt_vs_n.png',bbox_inches='tight', pad_inches=0)


    # if my_rank == 0: print(IMAGE_Q_o[:,:,0])



