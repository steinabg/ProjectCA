# distutils: language = c++
import numpy as np
cimport numpy as cnp
import T1functions as T1
import T2functions as T2
import mathfunk as ma
cimport cython
from libc.stdlib cimport calloc
from libc.math cimport sqrt as csqrt
from libc.math cimport atan as carctan
from libc.math cimport tan as ctan
from libc.math cimport isnan as cisnan
from libc.math cimport pow as cpow
from libc.math cimport log2 as clog2
from libcpp.vector cimport vector as cvector

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def T_1(int Ny,int Nx,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,double[:,:] Q_th,double[:,:] Q_v,double dt,double g,
        ):  # Water entrainment. IN: Q_a,Q_th,Q_cj,Q_v. OUT: Q_vj,Q_th
    '''
    This function calculates the water entrainment.\
    Entrainment is the transport of fluid across an interface\
    between two bodies of fluid by a shear induced turbulent flux.
    I.e. the 'mixing' of two fluids across their interface. \

    '''
    cdef int ii,jj,zz
    cdef double g_prime, Ri_number, dimless_entrainment_rate, entrainment_rate
    nQ_th = np.zeros((Ny,Nx),dtype=np.double, order='C')
    nQ_cj = np.zeros((Ny,Nx,Nj),dtype=np.double, order='C')
    cdef double[:,:] nQ_th_view = nQ_th
    cdef double[:,:,:] nQ_cj_view = nQ_cj
    for ii in range(Ny):
        for jj in range(Nx):
            if (Q_th[ii, jj] > 0) and (Q_v[ii,jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                g_prime = 0
                for zz in range(Nj):
                    g_prime += g * (Q_cj[ii, jj, zz] * (rho_j[zz] - rho_a) / rho_a)
                Ri_number = g_prime * Q_th[ii, jj] / (Q_v[ii, jj] * Q_v[ii, jj])
                dimless_entrainment_rate = 0.075 / csqrt(1 + 718 * (Ri_number) ** (2.4))
                entrainment_rate = Q_v[ii, jj] * dimless_entrainment_rate

                nQ_th_view[ii, jj] = Q_th[ii, jj] + entrainment_rate * dt
                if cisnan(nQ_th_view[ii,jj]):
                    raise ValueError
                for zz in range(Nj):
                    nQ_cj_view[ii, jj, zz] = Q_cj[ii, jj, zz] * Q_th[ii, jj] / nQ_th_view[ii, jj]

            else:
                nQ_th_view[ii,jj] = Q_th[ii,jj]
                for zz in range(Nj):
                    nQ_cj_view[ii, jj, zz] = Q_cj[ii, jj, zz]

    return nQ_cj, nQ_th

@cython.cdivision(True)
cdef double cstd(double *arr, int length):
    cdef double sum = 0.0, mean, standardDeviation = 0.0
    cdef int i

    for i in range(length):
        sum += arr[i]

    mean = sum/length

    for i in range(length):
        standardDeviation += cpow(arr[i] - mean, 2)
    return csqrt(standardDeviation/length)

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def T_2(int Ny,int Nx,int Nj,int[:] rho_j,int rho_a,double[:] D_sj,double nu,double g,double c_D,double[:,:] Q_v,
        double[:] v_sj,double[:,:,:] Q_cj,double[:,:,:] Q_cbj,double[:,:] Q_th,double[:,:] Q_d,
        double dt,double porosity,double[:,:] Q_a,int[:,:] Q_v_is_zero_two_timesteps):
    '''
    This function updates Q_a,Q_d,Q_cj and Q_cbj. According to erosion and deposition rules.\
    IN: Q_a,Q_th,Q_cj,Q_cbj,Q_v. OUT:
    '''

    nQ_a = np.zeros((Ny, Nx), dtype=np.double, order='C')
    nQ_d = np.zeros((Ny, Nx), dtype=np.double, order='C')
    nQ_cj = np.zeros((Ny, Nx, Nj), dtype=np.double, order='C')
    nQ_cbj = np.zeros((Ny, Nx, Nj), dtype=np.double, order='C')
    nQ_th = np.zeros((Ny, Nx), dtype=np.double, order='C')
    cdef double[:,:] nQ_a_view = nQ_a
    cdef double[:,:] nQ_d_view = nQ_d
    cdef double[:,:] nQ_th_view = nQ_th
    cdef double[:,:,:] nQ_cj_view = nQ_cj
    cdef double[:,:,:] nQ_cbj_view = nQ_cbj
    cdef int num_cells = 0, ii, jj, kk, ll, invalid
    cdef int num_cells_invalid = 0
    cdef double *log_2_D_sj, sum_q_cj, sediment_mean_size, kappa, *f_sj, f_sj_sum, q_cj_sum
    cdef double fall_velocity_dimless, near_bed_c, particle_reynolds, Z_mj, erosion_rate
    for ii in range(Ny):
        for jj in range(Nx):
            nQ_th_view[ii,jj] = Q_th[ii,jj]
            invalid = 0
            if (Q_th[ii, jj] > 0) and (Q_v[ii, jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                num_cells += 1
                # Deposition initialization:
                sediment_mean_size = 1
                sum_q_cj = 0

                # Erosion initialization:
                # log_2_D_sj = np.zeros((Nj), dtype=np.double, order='C')
                log_2_D_sj = <double*> calloc(Nj,sizeof(double))

                for kk in range(Nj):
                    # Deposition part:
                    sum_q_cj += Q_cj[ii, jj, kk]
                    sediment_mean_size *= Q_cj[ii, jj, kk] * D_sj[kk]

                    # Erosion part:
                    log_2_D_sj[kk] = clog2(D_sj[kk])

                kappa = 1 - 0.288 * cstd(log_2_D_sj, Nj)
                sediment_mean_size = sediment_mean_size ** (1.0 / Nj) / sum_q_cj
                # f_sj = np.zeros((Nj), dtype=np.double, order='C')
                f_sj = <double*> calloc(Nj,sizeof(double))
                f_sj_sum = 0

                for kk in range(Nj):
                    # Deposition part:
                    fall_velocity_dimless = v_sj[kk] ** (3) * rho_a / ((rho_j[kk] - rho_a) * g * nu)
                    near_bed_c = Q_cj[ii, jj, kk] * (0.40 * (D_sj[kk] / sediment_mean_size) ** (1.64) + 1.64)
                    deposition_rate = fall_velocity_dimless * near_bed_c

                    # Erosion part:
                    particle_reynolds = csqrt(g * (rho_j[kk] - rho_a) * D_sj[kk] / rho_a) * D_sj[kk] / nu
                    # print("g = {}, rho_j[kk] = {}, rho_a = {}, Dsj = {}, nu = {}".format(g,rho_j[kk],rho_a,D_sj[kk],nu))
                    # print(particle_reynolds)
                    if (particle_reynolds >= 3.5):
                        function_reynolds = particle_reynolds ** (0.6)
                    elif (particle_reynolds > 1) and (particle_reynolds < 3.5):
                        function_reynolds = 0.586 * particle_reynolds ** (1.23)
                    else:
                        raise Exception('Eq. (40) (Salles) not defined for R_pj = {0}'.format(particle_reynolds))
                    Z_mj = kappa * csqrt(c_D * Q_v[ii, jj]) * function_reynolds / fall_velocity_dimless
                    erosion_rate = (1.3 * 10 ** (-7) * Z_mj ** (5)) / (1 + 4.3 * 10 ** (-7) * Z_mj ** (5))

                    # Exner equation:
                    f_sj[kk] = deposition_rate - erosion_rate * Q_cbj[ii, jj, kk]
                    f_sj_sum += f_sj[kk]

                nQ_a_view[ii, jj] = Q_a[ii, jj] + dt / (1 - porosity) * f_sj_sum
                nQ_d_view[ii, jj] = Q_d[ii, jj] + dt / (1 - porosity) * f_sj_sum

                for kk in range(Nj):
                    nQ_cj_view[ii, jj, kk] = Q_cj[ii, jj,kk] - dt / ((1 - porosity) * Q_th[ii, jj]) * f_sj[kk]
                    nQ_cbj_view[ii, jj, kk] = Q_cbj[ii, jj, kk] + dt / ((1 - porosity) * Q_d[ii, jj]) * \
                                         (f_sj[kk] - Q_cbj[ii, jj, kk] * f_sj_sum)
                    # If this operation leads to unphysical state, undo the rule:
                    if (nQ_cj_view[ii,jj,kk] < 0) or (cisnan(nQ_cj_view[ii,jj,kk])):
                        num_cells_invalid += 1
                        invalid = 1
                        break
                if(invalid == 1):
                    nQ_a_view[ii, jj] = Q_a[ii, jj]
                    nQ_d_view[ii, jj] = Q_d[ii, jj]
                    for ll in range(Nj):
                        nQ_cj_view[ii, jj, ll] = Q_cj[ii, jj, ll]
                        nQ_cbj_view[ii, jj, ll] = Q_cbj[ii, jj, ll]


            # If (interior cell) && (velocity has been zero for two time steps): sediment everything; remove t current
            elif (Q_v_is_zero_two_timesteps[ii, jj] == 1) and (Q_v[ii, jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                q_cj_sum = 0
                for kk in range(Nj):
                    nQ_cbj_view[ii, jj, kk] = Q_cbj[ii, jj, kk] + Q_cj[ii, jj, kk] * Q_th[ii, jj]
                    nQ_cj_view[ii, jj, kk] = 0
                    q_cj_sum += Q_cj[ii,jj, kk]
                nQ_th_view[ii, jj] = 0
                nQ_a_view[ii, jj] = Q_a[ii, jj] + q_cj_sum * Q_th[ii, jj]
                nQ_d_view[ii, jj] = Q_d[ii, jj] + q_cj_sum * Q_th[ii, jj]



            else:
                nQ_a_view[ii, jj] = Q_a[ii, jj]
                nQ_d_view[ii, jj] = Q_d[ii, jj]
                for kk in range(Nj):
                    nQ_cj_view[ii, jj, kk] = Q_cj[ii, jj, kk]
                    nQ_cbj_view[ii, jj, kk] = Q_cbj[ii, jj, kk]
    # if num_cells == num_cells_invalid:
    #     raise Exception("No cell changed")

    return nQ_a, nQ_d, nQ_cj, nQ_cbj, nQ_th


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def I_1(double[:,:] Q_th,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,double[:,:] Q_v,double[:,:] Q_a,
        int Ny,int Nx,double dx,double p_f, double p_adh,double dt, double g):
    '''
    This function calculates the turbidity current outflows.\
    IN: Q_a,Q_th,Q_v,Q_cj. OUT: Q_o
    p_f = np.deg2rad(1) # Height threshold friction angle

    '''
    nQ_o = np.zeros((Ny,Nx,6), dtype=np.double, order='C') # Reset outflow
    cdef double [:,:,:] nQ_o_view = nQ_o
    cdef int ii,jj,dir, nb_i, nb_j, zz, index
    cdef list
    cdef double Average, r, h_k, g_prime, height_center, *nb_h, *f, factor_n, factor_r, sum_nb_h_in_A
    # nb_index = [[-1,0],[-1,1],[0,1],[1,0],[1,-1],[0,-1]]
    cdef int nb_index[6][2]
    nb_index[0][:] = [-1, 0]
    nb_index[1][:] = [-1, 1]
    nb_index[2][:] = [0, 1]
    nb_index[3][:] = [1, 0]
    nb_index[4][:] = [1, -1]
    nb_index[5][:] = [0, -1]
    for ii in range(1, Ny - 1):
        for jj in range(1, Nx - 1):
            if (Q_th[ii,jj] > 0.0): # If cell has flow perform algorithm
                g_prime = 0
                for kk in range(Nj):
                    g_prime += g * (Q_cj[ii, jj, kk] * (rho_j[kk] - rho_a) / rho_a)
                h_k = 0.5 * Q_v[ii,jj] * Q_v[ii,jj] / g_prime
                r = Q_th[ii,jj] + h_k
                height_center = Q_a[ii,jj] + r

                A = [0,1,2,3,4,5]

                # nb_h = []
                nb_h = <double*> calloc(6,sizeof(double))

                for dir in range(6):
                    nb_i = ii + nb_index[dir][0]
                    nb_j = jj + nb_index[dir][1]

                    # nb_h.append(Q_a[nb_i,nb_j] + Q_th[nb_i,nb_j])
                    nb_h[dir] = Q_a[nb_i,nb_j] + Q_th[nb_i,nb_j]
                    if (carctan((height_center - nb_h[dir])/dx) < p_f):
                        del A[dir - (6-len(A))]

                eliminated = True
                Average = 0.0
                while eliminated and len(A) > 0:
                    eliminated = False
                    sum_nb_h_in_A = 0
                    for zz in range(len(A)):
                        dir = A[zz]
                        sum_nb_h_in_A += nb_h[dir]
                    Average = ( (r-p_adh) + sum_nb_h_in_A) / len(A)

                    A_copy = A.copy()
                    for index in range(len(A)):
                        dir = A[index]
                        if(nb_h[dir] >= Average):
                            del A_copy[index - (len(A) - len(A_copy))]
                            eliminated = True
                    A = A_copy.copy()

                # f = np.zeros((6),dtype=np.double, order='C')
                f = <double*> calloc(6,sizeof(double))
                factor_n = Q_th[ii,jj]/r
                factor_r = csqrt(2*r*g_prime)*dt/(dx/2)

                for zz in range(len(A)):
                    dir = A[zz]
                    f[dir] = Average - nb_h[dir]
                    nQ_o_view[ii,jj,dir] = f[dir] * factor_n * factor_r

    return nQ_o

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def I_2(int Ny,int Nx,int Nj,double[:,:,:] Q_o,double[:,:] Q_th,double[:,:,:] Q_cj):
    '''Update thickness and concentration. IN: Q_th,Q_cj,Q_o. OUT: Q_th,Q_cj'''
    nQ_th = np.zeros((Ny, Nx), dtype=np.double, order='C')
    nQ_cj = np.zeros((Ny, Nx, Nj), dtype=np.double, order='C')
    cdef double[:,:] nQ_th_view = nQ_th
    cdef double[:,:,:] nQ_cj_view = nQ_cj
    # nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    cdef int nb_index[6][2]
    nb_index[0][:] = [-1, 0]
    nb_index[1][:] = [-1, 1]
    nb_index[2][:] = [0, 1]
    nb_index[3][:] = [1, 0]
    nb_index[4][:] = [1, -1]
    nb_index[5][:] = [0, -1]
    cdef int nb_flow_dir[6]
    nb_flow_dir[:] = [3, 4, 5, 0, 1, 2]
    cdef int ii, jj, kk, ll
    cdef double Q_o_from_center_sum, *Q_o_Q_cj_neighbors
    # Only update interior cells:
    for ii in range(Ny):
        for jj in range(Nx):
            if (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                Q_o_from_center_sum = 0
                # Q_o_Q_cj_neighbors = np.zeros((Nj), dtype=np.double, order='C')
                Q_o_Q_cj_neighbors = <double*> calloc(Nj, sizeof(double))
                nQ_th_view[ii,jj] += Q_th[ii,jj]
                for kk in range(6):
                    # For thickness:
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    nQ_th_view[ii, jj] += Q_o[nb_ii, nb_jj, nb_flow_dir[kk]] - Q_o[ii, jj, kk]

                    # For concentration:
                    Q_o_from_center_sum += Q_o[ii, jj, kk]
                    for ll in range(Nj):
                        Q_o_Q_cj_neighbors[ll] += Q_o[nb_ii, nb_jj, nb_flow_dir[kk]] * Q_cj[nb_ii, nb_jj, ll]
                if (nQ_th_view[ii, jj] < 0) or cisnan(nQ_th_view[ii,jj]):
                    raise Exception("I_2: Negative sediment due to excessive outflow!")

                if (nQ_th_view[ii, jj] > 0):
                    for kk in range(Nj):
                        nQ_cj_view[ii, jj, kk] = 1 / nQ_th_view[ii, jj] * ((Q_th[ii, jj] - Q_o_from_center_sum) * Q_cj[ii, jj, kk] +
                                                                 Q_o_Q_cj_neighbors[kk])
                        if (cisnan(nQ_cj_view[ii,jj,kk])) or (nQ_cj_view[ii,jj,kk] < 0):
                            raise ValueError
            else:
                nQ_th_view[ii,jj] = Q_th[ii,jj]
                for kk in range(Nj):
                    nQ_cj_view[ii,jj,kk] = Q_cj[ii,jj,kk]

    return nQ_th, nQ_cj

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def I_3(double g,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,int Ny,int Nx,double[:,:] Q_a,
    double[:,:] Q_th,double[:,:,:] Q_o,double f,double a, Q_v):  # Should be done
    '''
    Update of turbidity flow velocity (speed!). IN: Q_a,Q_th,Q_o,Q_cj. OUT: Q_v.
    '''
    # nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    cdef int nb_index[6][2]
    nb_index[0][:] = [-1, 0]
    nb_index[1][:] = [-1, 1]
    nb_index[2][:] = [0, 1]
    nb_index[3][:] = [1, 0]
    nb_index[4][:] = [1, -1]
    nb_index[5][:] = [0, -1]
    nQ_v = np.zeros((Ny, Nx), dtype=np.double, order='C')
    Q_v_is_zero_two_timesteps = np.zeros((Ny, Nx), dtype=np.intc, order='C')
    cdef int [:,:] Q_v_is_zero_two_timesteps_view = Q_v_is_zero_two_timesteps
    cdef double[:,:] nQ_v_view = nQ_v
    cdef int ii, jj, nb_ii, nb_jj, num_removed
    cdef double U, Q_cj_sum, g_prime, slope
    for ii in range(1, Ny-1):
        for jj in range(1, Nx - 1):
            if (Q_th[ii,jj] > 0):
                U = 0
                Q_cj_sum = 0
                g_prime = 0
                num_removed = 0
                for kk in range(Nj):
                    g_prime += g * (Q_cj[ii, jj, kk] * (rho_j[kk] - rho_a) / rho_a)
                    Q_cj_sum += Q_cj[ii,jj,kk]
                for kk in range(6):
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    slope = (Q_a[ii,jj] + Q_th[ii,jj]) - (Q_a[nb_ii, nb_jj] + Q_th[nb_ii, nb_jj])
                    if slope < 0:
                        slope = 0
                        num_removed += 1
                    U += csqrt( 8 * g_prime * Q_cj_sum * Q_o[ii,jj,kk] * slope / (f * (1 + a)) )
                    if (cisnan(U)):
                        raise Exception("U is nan")
                if num_removed < 6:
                    nQ_v_view[ii,jj] = U/(6 - num_removed)
                if cisnan(nQ_v_view[ii, jj]):
                    raise ValueError
            if (Q_v[ii, jj] == 0) and (nQ_v_view[ii, jj] == 0):
                Q_v_is_zero_two_timesteps_view[ii, jj] = 1

    return nQ_v, Q_v_is_zero_two_timesteps

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def I_4(double[:,:] Q_d,int Ny,int Nx,int Nj,double dx,double reposeAngle,double[:,:,:] Q_cbj,double[:,:] Q_a,double[:,:,:] seaBedDiff):  # Toppling rule
    nQ_d = np.zeros((Ny,Nx), dtype=np.double, order='C')
    nQ_a = np.zeros((Ny,Nx), dtype=np.double, order='C')
    nQ_cbj = np.zeros((Ny,Nx, Nj), dtype=np.double, order='C')
    cdef double[:,:] nQ_d_view = nQ_d
    cdef double[:,:] nQ_a_view = nQ_a
    cdef double[:,:,:] nQ_cbj_view = nQ_cbj
    cdef int ii, jj, kk, num_recv, *give_dir, nb_no, nb_ii, nb_jj, ll
    cdef double *diff, angle, frac, deltaS

    # nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    cdef int nb_index[6][2]
    nb_index[0][:] = [-1, 0]
    nb_index[1][:] = [-1, 1]
    nb_index[2][:] = [0, 1]
    nb_index[3][:] = [1, 0]
    nb_index[4][:] = [1, -1]
    nb_index[5][:] = [0, -1]
    for ii in range(Ny):
        for jj in range(Nx):
            nQ_d_view[ii, jj] = Q_d[ii, jj]
            nQ_a_view[ii, jj] = Q_a[ii, jj]
            for kk in range(Nj):
                nQ_cbj_view[ii, jj, kk] = Q_cbj[ii, jj, kk]


    for ii in range(Ny):
        for jj in range(Nx):
            if (Q_d[ii,jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                num_recv = 0
                # give_dir = np.zeros((6),np.int, order='C')
                give_dir = <int*> calloc(6, sizeof(int))
                # diff = np.zeros((6), dtype=np.double, order='C')
                diff = <double*> calloc(6, sizeof(double))
                for kk in range(6):
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    diff[kk] = Q_d[ii,jj] - Q_d[nb_ii, nb_jj] + seaBedDiff[ii-1,jj-1,kk]
                    angle = carctan(diff[kk]/dx)
                    if angle > reposeAngle:
                        give_dir[num_recv] = kk
                        num_recv += 1
                if num_recv > 0:
                    for kk in range(num_recv):
                        nb_no = give_dir[kk]
                        nb_ii = ii + nb_index[nb_no][0]
                        nb_jj = jj + nb_index[nb_no][1]
                        frac = 0.5 * (diff[nb_no] - dx * ctan(reposeAngle)) / Q_d[ii,jj]
                        if frac > 0.5:
                            frac = 0.5
                        deltaS = Q_d[ii,jj] * frac/num_recv

                        for ll in range(Nj):
                            nQ_cbj_view[nb_ii, nb_jj, ll] = (nQ_d_view[nb_ii, nb_jj] * nQ_cbj_view[nb_ii, nb_jj, ll] +
                                                       nQ_cbj_view[ii,jj, ll] * deltaS) / (nQ_d_view[nb_ii, nb_jj] + deltaS)

                        nQ_d_view[nb_ii,nb_jj] += deltaS
                        nQ_a_view[nb_ii,nb_jj] += deltaS

                        nQ_d_view[ii,jj] -= deltaS
                        nQ_a_view[ii,jj] -= deltaS




    return nQ_a, nQ_d, nQ_cbj
