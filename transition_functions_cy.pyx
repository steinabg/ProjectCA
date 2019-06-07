# distutils: language = c++
import numpy as np
cimport numpy as np
import T1functions as T1
import T2functions as T2
import mathfunk as ma
cimport cython
from libc.stdlib cimport calloc, free
from libc.math cimport sqrt as csqrt
from libc.math cimport atan as carctan
from libc.math cimport sin as csin
from libc.math cimport cos as ccos
from libc.math cimport tan as ctan
from libc.math cimport isnan as cisnan
from libc.math cimport pow as cpow
from libc.math cimport log2 as clog2
from libcpp.vector cimport vector as cvector

cdef double get_gprime(double[:] Q_cj, int[:] rho_j, int rho_a,int Nj, double g):
    cdef double g_prime = 0.0
    cdef double Q_cj_sum = 0.0
    for zz in range(Nj):
        g_prime += g * (Q_cj[zz] * (rho_j[zz] - rho_a) / rho_a)
        # g_prime += g * ( (rho_j[zz] - rho_a) / rho_a)
        Q_cj_sum += Q_cj[zz]
    # g_prime = g_prime / Q_cj_sum
    return g_prime

def killQth(double[:,:] Q_d, double[:,:] Q_a, double[:,:,:] Q_cbj,
            double[:,:,:] Q_cj, double[:,:] Q_th, double[:,:,:] Q_o, double[:,:] Q_v,
            double[:,:] Qcjsum, double[:,:] Qosum, int Ny, int Nx, int Nj):
    # nQ_a = np.zeros((Ny, Nx), order='C', dtype=np.double)
    # nQ_th = np.zeros((Ny, Nx), order='C', dtype=np.double)
    # nQ_v = np.zeros((Ny, Nx), order='C', dtype=np.double)
    # nQ_cj = np.zeros((Ny, Nx, Nj), order='C',dtype=np.double)
    # nQ_cbj = np.zeros((Ny, Nx, Nj), order='C',dtype=np.double)
    # nQ_d = np.zeros((Ny, Nx), order='C', dtype=np.double)
    # nQ_o = np.zeros((Ny, Nx, 6), order='C', dtype=np.double)
    # cdef double[:,:] nQ_a_view = nQ_a
    # cdef double[:,:] nQ_th_view = nQ_th
    # cdef double[:,:] nQ_v_view = nQ_v
    # cdef double[:,:] nQ_d_view = nQ_d
    # cdef double[:,:,:] nQ_cj_view = nQ_cj
    # cdef double[:,:,:] nQ_cbj_view = nQ_cbj
    # cdef double[:,:,:] nQ_o_view = nQ_o
    cdef int ii,jj,kk
    cdef double qd_old
    for ii in range(Ny):
        for jj in range(Nx):
            if (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                if (Qcjsum[ii,jj] < 1e-1 and Q_th[ii,jj] > 100) or (Qosum[ii,jj] < 1e-10 and Q_th[ii,jj]>0):
                    qd_old = Q_d[ii,jj]
                    Q_a[ii,jj] += Qcjsum[ii,jj]*Q_th[ii,jj]
                    Q_d[ii,jj] += Qcjsum[ii,jj]*Q_th[ii,jj]
                    for kk in range(Nj):
                        Q_cbj[ii,jj,kk] = (Q_cbj[ii,jj,kk]*qd_old + Q_cj[ii,jj,kk]*Q_th[ii,jj])/Q_d[ii,jj]
                        Q_cj[ii,jj, kk] = 0
                    Q_th[ii,jj] = 0
                    Q_v[ii,jj] = 0
                    for kk in range(6):
                        Q_o[ii,jj,kk] = 0


    a = np.asarray
    t='double'
    return a(Q_d,dtype=t),a(Q_a,dtype=t),a(Q_cbj,dtype=t),a(Q_cj,dtype=t),a(Q_th,dtype=t),a(Q_o,dtype=t),a(Q_v,dtype=t)




def correctQcbj(int Ny, int Nx, int Nj, double[:,:,:] Q_cbj, double[:,:] err):
    ''' This function attempts to subtract the deviation from 1 to Qcbj'''

    nQ_cbj = np.zeros((Ny,Nx,Nj), dtype='double', order='C')
    cdef double[:,:,:] nQ_cbj_view = nQ_cbj
    cdef int ii,jj,zz, fix
    for ii in range(Ny):
        for jj in range(Nx):
            fix=0
            for zz in range(Nj):
                nQ_cbj_view[ii, jj, zz] = Q_cbj[ii,jj,zz]
                if err[ii,jj] > 0 and Q_cbj[ii,jj,zz] > 100 * err[ii,jj] and fix==0:
                    fix=1
                    nQ_cbj_view[ii, jj, zz] -= err[ii,jj]

    return nQ_cbj

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
                # g_prime = 0.0
                # Q_cj_sum = 0.0
                # for zz in range(Nj):
                #     g_prime += g * (Q_cj[ii, jj, zz] * (rho_j[zz] - rho_a) / rho_a)
                #     # g_prime += g * ( (rho_j[zz] - rho_a) / rho_a)
                #     Q_cj_sum += Q_cj[ii, jj, zz]
                # # g_prime = g_prime / Q_cj_sum
                g_prime = get_gprime(Q_cj[ii,jj,:], rho_j, rho_a, Nj, g)
                Ri_number = g_prime * Q_th[ii, jj] / (Q_v[ii, jj] * Q_v[ii, jj])
                dimless_entrainment_rate = 0.075 / csqrt(1.0 + 718.0 * (Ri_number) ** (2.4))
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


cdef double cstd(double *arr, int length):
    cdef double sum = 0.0, mean, standardDeviation = 0.0
    cdef int i

    for i in range(length):
        sum += arr[i]

    mean = sum/length

    for i in range(length):
        standardDeviation += cpow(arr[i] - mean, 2)
    return csqrt(standardDeviation/length)


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
    cdef double *log_2_D_sj
    cdef double sum_q_cj, sediment_mean_size, kappa
    cdef double *f_sj
    cdef double f_sj_sum, q_cj_sum
    cdef double fall_velocity_dimless, near_bed_c, particle_reynolds, Z_mj, erosion_rate
    for ii in range(Ny):
        for jj in range(Nx):
            nQ_th_view[ii,jj] = Q_th[ii,jj]
            invalid = 0
            if (Q_th[ii, jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                num_cells += 1
                # Deposition initialization:
                sediment_mean_size = 1.0
                sum_q_cj = 0.0

                # Erosion initialization:
                # log_2_D_sj = np.zeros((Nj), dtype=np.double, order='C')
                log_2_D_sj = <double*> calloc(Nj,sizeof(double))

                for kk in range(Nj):
                    # Deposition part:
                    sum_q_cj += Q_cj[ii, jj, kk]
                    # sediment_mean_size *= Q_cj[ii, jj, kk] * D_sj[kk]
                    sediment_mean_size *= D_sj[kk] ** Q_cj[ii, jj, kk]

                    # Erosion part:
                    log_2_D_sj[kk] = clog2(D_sj[kk] * 1000)

                kappa = 1 - 0.288 * cstd(log_2_D_sj, Nj)
                # sediment_mean_size = sediment_mean_size ** (1.0 / Nj) / sum_q_cj
                try:
                    sediment_mean_size = sediment_mean_size ** (1.0 / sum_q_cj)
                except ZeroDivisionError:
                    raise ZeroDivisionError("sum_qcj={0}, Qth={1}".format(sum_q_cj, Q_th[ii,jj]))
                # f_sj = np.zeros((Nj), dtype=np.double, order='C')
                f_sj = <double*> calloc(Nj,sizeof(double))
                f_sj_sum = 0.0

                for kk in range(Nj):
                    # Deposition part:
                    # fall_velocity_dimless = v_sj[kk] ** (3) * rho_a / ((rho_j[kk] - rho_a) * g * nu)
                    fall_velocity_dimless = v_sj[kk]
                    near_bed_c = Q_cj[ii, jj, kk] * (0.40 * (D_sj[kk] / sediment_mean_size) ** (1.64) + 1.64)
                    deposition_rate = fall_velocity_dimless * near_bed_c

                    # Erosion part:
                    particle_reynolds = csqrt(g * (rho_j[kk] - rho_a) * D_sj[kk] / rho_a) * D_sj[kk] / nu
                    # print("g = {}, rho_j[kk] = {}, rho_a = {}, Dsj = {}, nu = {}".format(g,rho_j[kk],rho_a,D_sj[kk],nu))
                    # print(particle_reynolds)
                    if (particle_reynolds >= 3.5):
                        function_reynolds = particle_reynolds ** (0.6)
                    else:
                        function_reynolds = 0.586 * particle_reynolds ** (1.23)

                    Z_mj = kappa * csqrt(c_D * Q_v[ii, jj] * Q_v[ii, jj]) * function_reynolds / fall_velocity_dimless
                    erosion_rate = (1.3 * 1e-7 * Z_mj ** (5)) / (1 + 4.3 * 1e-7 * Z_mj ** (5)) * fall_velocity_dimless
                    # erosion_rate = (1.3 * 1e-7 * Z_mj ** (5)) / (1 + 4.3 * 1e-7 * Z_mj ** (5))

                    # Exner equation:
                    f_sj[kk] = deposition_rate - erosion_rate * Q_cbj[ii, jj, kk]
                    f_sj_sum += f_sj[kk]

                nQ_a_view[ii, jj] = Q_a[ii, jj] + dt / (1 - porosity) * f_sj_sum
                nQ_d_view[ii, jj] = Q_d[ii, jj] + dt / (1 - porosity) * f_sj_sum

                for kk in range(Nj):
                    nQ_cj_view[ii, jj, kk] = Q_cj[ii, jj,kk] - dt / ((1 - porosity) * Q_th[ii, jj]) * f_sj[kk]
                    nQ_cbj_view[ii, jj, kk] = Q_cbj[ii, jj, kk] + dt / ((1 - porosity) * nQ_d_view[ii, jj]) * \
                                         (f_sj[kk] - Q_cbj[ii, jj, kk] * f_sj_sum)
                    # If this operation leads to unphysical state, undo the rule:
                    if (nQ_cj_view[ii,jj,kk] < 0) or nQ_cj_view[ii,jj,kk] > 1 or (cisnan(nQ_cj_view[ii,jj,kk])) or nQ_d_view[ii, jj] <= 0\
                            or nQ_cbj_view[ii,jj,kk] < 0 or nQ_cbj_view[ii,jj,kk] > 1:
                        num_cells_invalid += 1
                        invalid = 1
                        break
                if(invalid == 1):
                    nQ_a_view[ii, jj] = Q_a[ii, jj]
                    nQ_d_view[ii, jj] = Q_d[ii, jj]
                    for ll in range(Nj):
                        nQ_cj_view[ii, jj, ll] = Q_cj[ii, jj, ll]
                        nQ_cbj_view[ii, jj, ll] = Q_cbj[ii, jj, ll]
                free(log_2_D_sj)
                free(f_sj)

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




def I_1(double[:,:] Q_th,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,double[:,:] Q_v,double[:,:] Q_a,
        int Ny,int Nx,double dx,double p_f, double p_adh,double dt, double g):
    '''
    This function calculates the turbidity current outflows.\
    IN: Q_a,Q_th,Q_v,Q_cj. OUT: Q_o
    p_f = np.deg2rad(1) # Height threshold friction angle

    '''
    nQ_o = np.zeros((Ny,Nx,6), dtype=np.double, order='C') # Reset outflow
    nQ_o_no_time = np.zeros((Ny,Nx,6), dtype=np.double, order='C') # Reset outflow
    cdef double [:,:,:] nQ_o_view = nQ_o
    cdef double [:,:,:] nQ_o_no_time_view = nQ_o_no_time
    cdef int ii,jj,dir, nb_i, nb_j, zz, index
    cdef list
    cdef double Average, r, h_k, g_prime, height_center, Q_o_sum
    cdef double *nb_h
    cdef double *f
    cdef double factor_n, factor_r, sum_nb_h_in_A, t
    # nb_index = [[-1,0],[-1,1],[0,1],[1,0],[1,-1],[0,-1]]
    cdef int nb_index[6][2]
    nb_index[0][:] = [-1, 0]
    nb_index[1][:] = [-1, 1]
    nb_index[2][:] = [0, 1]
    nb_index[3][:] = [1, 0]
    nb_index[4][:] = [1, -1]
    nb_index[5][:] = [0, -1]

    # Calculate global relaxation constant
    factor_r = 1.0
    for ii in range(1, Ny - 1):
        for jj in range(1, Nx - 1):
            if (Q_th[ii,jj] > 1e-2): # If cell has flow perform algorithm
                # g_prime = 0.0
                # Q_cj_sum = 0.0
                # for kk in range(Nj):
                #     g_prime += g * (Q_cj[ii, jj, kk] * (rho_j[kk] - rho_a) / rho_a)
                #     Q_cj_sum += Q_cj[ii, jj, kk]
                    # g_prime += g * ( (rho_j[kk] - rho_a) / rho_a)
                # g_prime = g_prime / Q_cj_sum
                g_prime = get_gprime(Q_cj[ii,jj,:], rho_j, rho_a, Nj, g)
                h_k = 0.5 * Q_v[ii,jj] * Q_v[ii,jj] / g_prime
                r = Q_th[ii,jj] + h_k
                t = np.sqrt(2 * r * g_prime) * dt / dx
                if t < factor_r:
                    factor_r = t
    if factor_r < 0.2: factor_r = 0.2

    for ii in range(1, Ny - 1):
        for jj in range(1, Nx - 1):
            if (Q_th[ii,jj] > 1e-2): # If cell has flow perform algorithm
                # factor_r = 1.0
                # g_prime = 0.0
                # Q_cj_sum = 0.0
                # for kk in range(Nj):
                #     # g_prime += g * ( (rho_j[kk] - rho_a) / rho_a)
                #     g_prime += g * (Q_cj[ii, jj, kk] * (rho_j[kk] - rho_a) / rho_a)
                #     Q_cj_sum += Q_cj[ii, jj, kk]
                # g_prime = g_prime / Q_cj_sum
                g_prime = get_gprime(Q_cj[ii,jj,:], rho_j, rho_a, Nj, g)
                h_k = 0.5 * Q_v[ii,jj] * Q_v[ii,jj] / g_prime
                r = Q_th[ii,jj] + h_k
                if np.isinf(r):
                    raise ValueError('Q_th={0}, h_k={1}\n'
                                     'Q_v={2}, g_prime={3}'.format(Q_th[ii,jj], h_k,Q_v[ii,jj], g_prime))
                # t = csqrt(2 * r * g_prime) * dt / dx
                # if t < factor_r:
                #     factor_r = t
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
                    sum_nb_h_in_A = 0.0
                    for zz in range(len(A)):
                        dir = A[zz]
                        sum_nb_h_in_A += nb_h[dir]
                    Average = ( (r-p_adh) + sum_nb_h_in_A) / len(A)
                    if np.isinf(Average):
                        raise ValueError('r={0}, p_adh={1}, sum_nb_={2}, len(A)={3}'.format(r,p_adh,sum_nb_h_in_A,len(A)))
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
                # factor_r = 1
                Q_o_sum = 0.0
                Qosumnotime = 0.0
                for zz in range(len(A)):
                    dir = A[zz]
                    f[dir] = Average - nb_h[dir]
                    nQ_o_view[ii,jj,dir] = f[dir] * factor_n * factor_r
                    nQ_o_no_time_view[ii, jj, dir] = f[dir] * factor_n
                    Qosumnotime += nQ_o_no_time_view[ii, jj, dir]
                    Q_o_sum += nQ_o_view[ii,jj,dir]
                    if cisnan(nQ_o_view[ii, jj, dir]):
                        raise ValueError('f[dir]={0}, factor_n={1}, factor_r={2}\n'
                                         'average={3}, nb_h[dir]={4}'.format(f[dir],factor_n,factor_r,Average,nb_h[dir]))
                if Q_o_sum > Q_th[ii, jj]:
                    raise ValueError("Qosum = {0}, Qth={1}\n"
                                     "Qosumnotime = {2}, ".format(Q_o_sum, Q_th[ii,jj], Qosumnotime))
                free(f)
                free(nb_h)
    return nQ_o, nQ_o_no_time


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
    cdef double Q_o_from_center_sum
    cdef double *Q_o_Q_cj_neighbors
    # Only update interior cells:
    for ii in range(Ny):
        for jj in range(Nx):
            if (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                Q_o_from_center_sum = 0.0
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
                        nQ_cj_view[ii, jj, kk] = 1.0 / nQ_th_view[ii, jj] * ((Q_th[ii, jj] - Q_o_from_center_sum) * Q_cj[ii, jj, kk] +
                                                                 Q_o_Q_cj_neighbors[kk])
                        if (cisnan(nQ_cj_view[ii,jj,kk])) or (nQ_cj_view[ii,jj,kk] < 0):
                            raise ValueError
                free(Q_o_Q_cj_neighbors)
            else:
                nQ_th_view[ii,jj] = Q_th[ii,jj]
                for kk in range(Nj):
                    nQ_cj_view[ii,jj,kk] = Q_cj[ii,jj,kk]

    return nQ_th, nQ_cj

def I_3(double g,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,int Ny,int Nx,double[:,:] Q_a,
    double[:,:] Q_th,double[:,:,:] Q_o,double f,double a,double[:,:] Q_v,double dt, double dx, method='salles'):  # Should be done
    if method=='salles':
        nQ_v, Q_v_is_zero_two_timesteps = salles_speed(g,Nj, Q_cj, rho_j,rho_a, Ny, Nx, Q_a,
     Q_th, Q_o, f, a, Q_v, dt,  dx)
    elif method=='mulder1':
        nQ_v, Q_v_is_zero_two_timesteps = mulder1_speed(g,Nj, Q_cj, rho_j,rho_a, Ny, Nx, Q_a,
     Q_th, Q_o, f, a, Q_v, dt,  dx)
    elif method=='mulder2':
        nQ_v, Q_v_is_zero_two_timesteps = mulder2_speed(g,Nj, Q_cj, rho_j,rho_a, Ny, Nx, Q_a,
     Q_th, Q_o, f, a, Q_v, dt,  dx)
    else:
        raise Exception("Invalid method!")

    return nQ_v, Q_v_is_zero_two_timesteps


def salles_speed(double g,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,int Ny,int Nx,double[:,:] Q_a,
    double[:,:] Q_th,double[:,:,:] Q_o,double f,double a,double[:,:] Q_v,double dt, double dx):  # Should be done
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
    cdef double U, Q_cj_sum, g_prime, slope, Q_o_sum
    for ii in range(1, Ny-1):
        for jj in range(1, Nx - 1):
            if (Q_th[ii,jj] > 0):
                U = 0
                Q_cj_sum = 0
                g_prime = 0
                Q_o_sum = 0.0
                num_removed = 0
                S = []
                for kk in range(Nj):
                    # g_prime += g * (Q_cj[ii, jj, kk] * (rho_j[kk] - rho_a) / rho_a)
                    # g_prime += g * ( (rho_j[kk] - rho_a) / rho_a)
                    Q_cj_sum += Q_cj[ii,jj,kk]
                g_prime = get_gprime(Q_cj[ii,jj,:], rho_j, rho_a, Nj, g)
                # g_prime = g_prime / Q_cj_sum
                # print("g_prime", g_prime)
                for kk in range(6):
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    slope = ((Q_a[ii,jj] + Q_th[ii,jj]) - (Q_a[nb_ii, nb_jj] + Q_th[nb_ii, nb_jj]))/dx
                    S.append(slope)
                    if Q_o[ii,jj,kk] <= 1e-3 or np.isinf(slope) or slope<0:
                        slope = 0
                        num_removed += 1
                    U +=csqrt( 8 * g_prime * Q_o[ii,jj,kk] * slope / (f * (1 + a)) )
                    # Q_o_sum += Q_o[ii,jj,kk]
                    # U += csqrt( 8 * g_prime * Q_o[ii,jj,kk] * slope / (f * (1 + a)) )
                    if (cisnan(U)):
                        raise Exception("U is nan", slope, g_prime)
                if num_removed < 6:
                    # U = U/Q_o_sum
                    nQ_v_view[ii,jj] = U/(6 - num_removed)
                    # nQ_v_view[ii,jj] = U
                    # if nQ_v_view[ii,jj] < 1e-1:
                    #     raise Exception("U={0}, slopes={1}, Q_o = {2}".format(U, S, np.asarray(Q_o[ii,jj,:])))
                if cisnan(nQ_v_view[ii, jj]):
                    raise ValueError
            if (Q_v[ii, jj] == 0) and (nQ_v_view[ii, jj] == 0):
                Q_v_is_zero_two_timesteps_view[ii, jj] = 1

    return nQ_v, Q_v_is_zero_two_timesteps

def mulder1_speed(double g,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,int Ny,int Nx,double[:,:] Q_a,
    double[:,:] Q_th,double[:,:,:] Q_o,double f,double a,double[:,:] Q_v,double dt, double dx):
    '''
    Ref: Mulder (1998), eq. (3)
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
                # g_prime = 0
                num_removed = 0
                ave_rho_j=0
                aveSlope=0
                for kk in range(Nj):
                    ave_rho_j += Q_cj[ii,jj,kk]*rho_j[kk]
                #     g_prime += g * (Q_cj[ii, jj, kk]**2 * (rho_j[kk] - rho_a) / rho_a)
                    # g_prime += g * ( (rho_j[kk] - rho_a) / rho_a)
                    Q_cj_sum += Q_cj[ii,jj,kk]
                ave_rho_j = ave_rho_j/Q_cj_sum
                g_prime = g*(ave_rho_j-rho_a)/rho_a
                # g_prime = g_prime / Q_cj_sum
                # g_prime = get_gprime(Q_cj[ii,jj,:], rho_j, rho_a, Nj, g)
                for kk in range(6):
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    slope = ((Q_a[ii,jj] + Q_th[ii,jj]) - (Q_a[nb_ii, nb_jj] + Q_th[nb_ii, nb_jj]))/dx
                    if Q_o[ii,jj,kk] <= 0 or np.isinf(slope) or slope < 0:
                        slope = 0
                        num_removed += 1
                        continue
                    aveSlope += slope
                    # U += csqrt( g_prime * Q_cj_sum * Q_o[ii,jj,kk] * csin(slope) / (0.004 * (1 + a)) ) #  Mulder (1998)
                    # if (cisnan(U)):
                    #     raise Exception("U is nan", slope)
                if num_removed < 6:
                    aveSlope = aveSlope/(6-num_removed)
                    nQ_v_view[ii,jj] = csqrt( g_prime * Q_cj_sum * Q_th[ii,jj] * csin(aveSlope) / (0.004 * (1 + a)) )
                    # nQ_v_view[ii,jj] = U/(6 - num_removed)
                if cisnan(nQ_v_view[ii, jj]):
                    raise ValueError
            if (Q_v[ii, jj] == 0) and (nQ_v_view[ii, jj] == 0):
                Q_v_is_zero_two_timesteps_view[ii, jj] = 1

    return nQ_v, Q_v_is_zero_two_timesteps

def mulder2_speed(double g,int Nj,double[:,:,:] Q_cj,int[:] rho_j,int rho_a,int Ny,int Nx,double[:,:] Q_a,
    double[:,:] Q_th,double[:,:,:] Q_o,double f,double a,double[:,:] Q_v,double dt, double dx):  # Should be done
    '''
    Ref. Mulder (1998), eq. (8)
    Update of turbidity flow velocity (speed!). IN: Q_a,Q_th,Q_o,Q_cj. OUT: Q_v.
    '''
    # nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    cdef double repose = np.deg2rad(30)
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
            if (Q_th[ii,jj] > 1e-10):
                ave_slope=0
                U = 0
                Q_cj_sum = 0
                # g_prime = 0
                num_removed = 0
                for kk in range(Nj):
                    # g_prime += g * (Q_cj[ii, jj, kk] * (rho_j[kk] - rho_a) / rho_a)
                    # g_prime += g * ( (rho_j[kk] - rho_a) / rho_a)
                    Q_cj_sum += Q_cj[ii,jj,kk]
                # g_prime = g_prime / Q_cj_sum
                g_prime = get_gprime(Q_cj[ii,jj,:], rho_j, rho_a, Nj, g)
                for kk in range(6):
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    slope = ((Q_a[ii,jj] + Q_th[ii,jj]) - (Q_a[nb_ii, nb_jj] + Q_th[nb_ii, nb_jj]))/dx
                    if np.isinf(slope) or Q_o[ii,jj,kk] <= 1e-10:
                        # slope = 0
                        num_removed += 1
                        continue
                    U += (g_prime * csin(slope) - (1.0 + 0.43)*0.004*Q_v[ii,jj]**2.0/(Q_o[ii,jj,kk]) -
                                       g_prime*ccos(slope)*ctan(repose) * (2.718**Q_cj_sum-1.0)/(2.718-1.0))*dt #  Mulder (1998)
                    if (cisnan(U)):
                        raise Exception("U is nan", slope)
                # if U < 0: U = 0
                if num_removed < 6:
                    nQ_v_view[ii,jj] = Q_v[ii,jj] + U/(6 - num_removed)
                else:
                    nQ_v_view[ii,jj] = Q_v[ii,jj]
                if nQ_v_view[ii,jj] < 0: nQ_v_view[ii,jj] = 0
                if cisnan(nQ_v_view[ii, jj]):
                    raise ValueError
            if (Q_v[ii, jj] == 0) and (nQ_v_view[ii, jj] == 0):
                Q_v_is_zero_two_timesteps_view[ii, jj] = 1

    return nQ_v, Q_v_is_zero_two_timesteps



def I_4(double[:,:] Q_d,int Ny,int Nx,int Nj,double dx,double reposeAngle,double[:,:,:] Q_cbj,double[:,:] Q_a,double[:,:,:] seaBedDiff, debug=0):  # Toppling rule
    nQ_d = np.zeros((Ny,Nx), dtype=np.double, order='C')
    nQ_a = np.zeros((Ny,Nx), dtype=np.double, order='C')
    nQ_cbj = np.zeros((Ny,Nx, Nj), dtype=np.double, order='C')
    cdef double[:,:] nQ_d_view = nQ_d
    cdef double[:,:] nQ_a_view = nQ_a
    cdef double[:,:,:] nQ_cbj_view = nQ_cbj
    cdef int ii, jj, kk, num_recv
    cdef int *give_dir
    cdef int nb_no, nb_ii, nb_jj, ll
    cdef double *diff
    cdef double angle, frac, deltaS, err, qcbj_sum
    #### MPI: For sending deltaS back to neighbor rank
    top = np.zeros((Nx, Nj+1), dtype=np.double, order='C')
    bot = np.zeros((Nx, Nj+1), dtype=np.double, order='C')
    left = np.zeros((Ny, Nj+1), dtype=np.double, order='C')
    right = np.zeros((Ny, Nj+1), dtype=np.double, order='C')


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
                    if debug and ii == Ny-2 and jj == 5:
                        print("diff[kk] = ", diff[kk] , ". Q_d: ", Q_d[ii,jj], " + Q_nb: ", Q_d[nb_ii,nb_jj], " sbdiff = ", seaBedDiff[ii-1,jj-1,kk],
                              "Q_a: ", Q_a[ii,jj], " Q_anb = ", Q_a[nb_ii, nb_jj])
                        # print("Q_d = ", Q_d[ii,jj], "Q_d_south = ", Q_d[ii+1,jj], " diff = ", diff[kk], " kk = ", kk)
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
                        qcbj_sum = 0
                        for ll in range(Nj):
                            try:
                                nQ_cbj_view[nb_ii, nb_jj, ll] = (nQ_d_view[nb_ii, nb_jj] * nQ_cbj_view[nb_ii, nb_jj, ll] +
                                                       nQ_cbj_view[ii,jj, ll] * deltaS) / (nQ_d_view[nb_ii, nb_jj] + deltaS)
                                qcbj_sum += nQ_cbj_view[nb_ii, nb_jj, ll]
                            except ZeroDivisionError:
                                raise ZeroDivisionError("Qd = {0}, frac = {1}, num = {2} ".format(Q_d[ii,jj], frac, num_recv))
                        if qcbj_sum-1e-15 > 1: # Try to correct numerical error
                            if qcbj_sum-1e-4 > 1:
                                raise ValueError(qcbj_sum-1) # Raise error for large diffs
                            else:
                                err = qcbj_sum-1
                                for ll in range(Nj):
                                    if nQ_cbj_view[nb_ii, nb_jj, ll] > 100*err:
                                        nQ_cbj_view[nb_ii, nb_jj, ll] -= err
                                        break
                        nQ_d_view[nb_ii,nb_jj] += deltaS
                        nQ_a_view[nb_ii,nb_jj] += deltaS

                        nQ_d_view[ii,jj] -= deltaS
                        nQ_a_view[ii,jj] -= deltaS

                        # Mass to be transferred through MPI
                        if (nb_ii == 0):  # Row = 0
                            top[nb_jj][0] += deltaS
                            for ll in range(Nj):
                                top[nb_jj][ll+1] += nQ_cbj_view[ii,jj, ll] * deltaS
                        elif (nb_ii == Ny - 1):
                            bot[nb_jj][0] += deltaS
                            for ll in range(Nj):
                                bot[nb_jj][ll+1] += nQ_cbj_view[ii,jj, ll] * deltaS
                        elif (nb_jj == 0):
                            left[nb_ii][0] += deltaS
                            for ll in range(Nj):
                                left[nb_ii][ll+1] += nQ_cbj_view[ii,jj, ll] * deltaS
                        elif(nb_jj == Nx - 1):
                            right[nb_ii][0] += deltaS
                            for ll in range(Nj):
                                right[nb_ii][ll+1] += nQ_cbj_view[ii,jj, ll] * deltaS

                free(give_dir)
                free(diff)



    return nQ_a, nQ_d, nQ_cbj, top, bot, left, right
