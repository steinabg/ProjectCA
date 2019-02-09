from mpi4py import MPI
import numpy as np
import sys
from sympy.ntheory import factorint  # To find processor grid size

Nx = Ny = 10
IMG_X = Nx
IMG_Y = Ny

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

neighbor_processes = [0, 0, 0, 0]

local_petri_A = np.zeros(())
local_petri_B = np.zeros(())

ITERATIONS = 1


def exchange_borders():
    if (num_iterations + 1) % 2:

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
        local_petri_ev[:] = local_petri_A[:, 1].copy()
        comm.Sendrecv(
            [local_petri_ev, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            2,
            [local_petri_wb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            2
        )
        local_petri_A[:, -1] = local_petri_wb.copy()

        # # Send east and receive from west
        local_petri_wv[:] = local_petri_A[:, -2].copy()
        # print("my value = ", local_petri_wv)
        # print("my destination = ", local_petri_eb)

        comm.Sendrecv(
            [local_petri_wv, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            0,
            [local_petri_eb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            0
        )
        local_petri_A[:, 0] = local_petri_eb.copy()
    else:

        # Send data south and receive from north
        comm.Sendrecv(
            [local_petri_B[-2, :], 1, border_row_t],  # send the second last row
            neighbor_processes[DOWN],
            0,
            [local_petri_B[0, :], 1, border_row_t],  # recvbuf = first row
            neighbor_processes[UP],
            0
        )

        # # Send data north and receive from south
        comm.Sendrecv(
            [local_petri_B[1, :], 1, border_row_t],  # sendbuf = second row (1.row = border)
            neighbor_processes[UP],  # destination
            1,  # sendtag
            [local_petri_B[-1, :], 1, border_row_t],  # recvbuf = last row
            neighbor_processes[DOWN],  # source
            1
        )
        #
        # Send west and receive from east
        local_petri_ev[:] = local_petri_B[:, 1].copy()
        comm.Sendrecv(
            [local_petri_ev, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            2,
            [local_petri_wb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            2
        )
        local_petri_B[:, -1] = local_petri_wb.copy()

        # # Send east and receive from west
        local_petri_wv[:] = local_petri_B[:, -2].copy()
        # print("my value = ", local_petri_wv)
        # print("my destination = ", local_petri_eb)

        comm.Sendrecv(
            [local_petri_wv, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            0,
            [local_petri_eb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            0
        )
        local_petri_B[:, 0] = local_petri_eb.copy()


def iterateCA():
    pass


def gather_petri():
    send = np.zeros((p_local_petri_y_dim, p_local_petri_x_dim), dtype=np.double)
    TEMP = np.zeros((p_local_petri_y_dim * p_y_dims * p_local_petri_x_dim * p_x_dims), dtype=np.double)
    IMAGE = np.zeros((p_local_petri_y_dim * p_y_dims * p_local_petri_x_dim * p_x_dims), dtype=np.double)

    if ((num_iterations + 1) % 2 == 0):
        send[:, :] = local_petri_B[1:-1, 1:-1]
    else:
        send[:, :] = local_petri_A[1:-1, 1:-1]

    comm.Gather(send, TEMP, 0)
    # if rank == 0: print('TEMP = \n', TEMP.reshape(p_local_petri_y_dim*p_y_dims,p_local_petri_x_dim*p_x_dims))

    if rank == 0:

        Tempindex = 0
        imageXcounter = 1
        imageYcounter = 1
        for i in range(p_local_petri_y_dim * p_y_dims * p_local_petri_x_dim * p_x_dims):

            if ((i + 1) % (p_local_petri_x_dim) == 0):
                IMAGE[i] = TEMP[Tempindex]

                if (imageXcounter == (p_local_petri_x_dim * p_x_dims)):
                    if (imageYcounter == p_local_petri_y_dim):
                        Tempindex += 1
                        imageYcounter = 0
                    else:
                        Tempindex = Tempindex - ((p_x_dims - 1) * p_local_petri_x_dim * p_local_petri_y_dim) + 1
                    imageXcounter = 0;
                    imageYcounter += 1;
                else:
                    Tempindex += (p_local_petri_x_dim * p_local_petri_y_dim) - p_local_petri_x_dim + 1
            else:
                IMAGE[i] = TEMP[Tempindex]
                Tempindex += 1

            imageXcounter += 1

        IMAGE = IMAGE.reshape(p_local_petri_y_dim * p_y_dims, p_local_petri_x_dim * p_x_dims)
    return IMAGE

if __name__ == "__main__":

    # print(sys.argv[0])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # p_y_dims = 2 # num procs y dir
    # p_x_dims = 2 # num procs x dir
    p_y_dims = int(np.sqrt(size))
    p_x_dims = int(np.sqrt(size))

    cartesian_communicator = comm.Create_cart((p_y_dims, p_x_dims), periods=(False, False))

    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)

    neighbor_processes[UP], neighbor_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbor_processes[LEFT], neighbor_processes[RIGHT] = cartesian_communicator.Shift(1, 1)

    # print("Process = %s\n"
    #       "my_mpi_row = %s\n"
    #       "my_mpi_column = %s --->\n"
    #       "neighbour_processes[UP] = %s\n"
    #       "neighbour_processes[DOWN] = %s\n"
    #       "neighbour_processes[LEFT] = %s\n"
    #       "neighbour_processes[RIGHT] = %s\n" % (rank, my_mpi_row, my_mpi_col,
    #                                            neighbor_processes[UP], neighbor_processes[DOWN],
    #                                            neighbor_processes[LEFT], neighbor_processes[RIGHT]))

    p_local_petri_x_dim = int(IMG_X / p_x_dims)  # Må være delelig
    p_local_petri_y_dim = int(IMG_Y / p_y_dims)

    local_petri_A = np.zeros((p_local_petri_x_dim + 2, p_local_petri_y_dim + 2))
    local_petri_B = np.zeros((p_local_petri_x_dim + 2, p_local_petri_y_dim + 2))
    local_petri_wb = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    local_petri_eb = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    local_petri_ev = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    local_petri_wv = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    TEMP = np.empty((1), dtype=np.double, order='C')

    local_petri_A[:] = comm.Get_rank()
    # print("process = ", comm.rank,"\n",
    #       local_petri_A)

    border_row_t = MPI.DOUBLE.Create_vector(p_local_petri_x_dim + 2,
                                            1,
                                            1)
    border_row_t.Commit()

    border_col_t = MPI.DOUBLE.Create_vector(p_local_petri_y_dim + 2,
                                            1,
                                            p_local_petri_x_dim + 2)
    border_col_t.Commit()

    for num_iterations in range(ITERATIONS):
        exchange_borders()
        iterateCA()

    comm.barrier()
    IMAGE = gather_petri()
    if rank == 0: print (IMAGE)

    # print("after \nprocess = ", comm.rank,"\n",
    #       local_petri_A)
