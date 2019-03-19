import numpy as np
import T1functions as T1
import T2functions as T2
import mathfunk as ma


def T_1(Ny, Nx, Nj, Q_cj, rho_j, rho_a, Q_th, Q_v, dt, g,
        ):  # Water entrainment. IN: Q_a,Q_th,Q_cj,Q_v. OUT: Q_vj,Q_th
    '''
    This function calculates the water entrainment.\
    Entrainment is the transport of fluid across an interface\
    between two bodies of fluid by a shear induced turbulent flux.
    I.e. the 'mixing' of two fluids across their interface. \

    '''
    nQ_th = Q_th.copy()
    nQ_cj = Q_cj.copy()
    for ii in range(1,Ny-1):
        for jj in range(1,Nx-1):
            if (Q_th[ii, jj] > 0) and (Q_v[ii,jj] > 0):
                g_prime = 0
                for zz in range(Nj):
                    g_prime += g * (Q_cj[ii, jj, zz] * (rho_j[zz] - rho_a) / rho_a)
                Ri_number = g_prime * Q_th[ii, jj] / (Q_v[ii, jj] * Q_v[ii, jj])
                dimless_entrainment_rate = 0.075 / np.sqrt(1 + 718 * (Ri_number) ** (2.4))
                entrainment_rate = Q_v[ii, jj] * dimless_entrainment_rate

                nQ_th[ii, jj] = Q_th[ii, jj] + entrainment_rate * dt
                if np.isnan(nQ_th[ii,jj]):
                    raise ValueError
                for zz in range(Nj):
                    nQ_cj[ii, jj, zz] = Q_cj[ii, jj, zz] * Q_th[ii, jj] / nQ_th[ii, jj]

    return nQ_cj, nQ_th


def T_2(Ny, Nx, Nj, rho_j, rho_a, D_sj, nu, g, c_D, Q_v, v_sj, Q_cj, Q_cbj, Q_th, Q_d, dt, porosity, Q_a,
        Q_v_is_zero_two_timesteps):
    '''
    This function updates Q_a,Q_d,Q_cj and Q_cbj. According to erosion and deposition rules.\
    IN: Q_a,Q_th,Q_cj,Q_cbj,Q_v. OUT:
    '''

    nQ_a = np.zeros((Ny, Nx), dtype=np.double, order='C')
    nQ_d = np.zeros((Ny, Nx), dtype=np.double, order='C')
    nQ_cj = np.zeros((Ny, Nx, Nj), dtype=np.double, order='C')
    nQ_cbj = np.zeros((Ny, Nx, Nj), dtype=np.double, order='C')
    nQ_th = np.zeros((Ny, Nx), dtype=np.double, order='C')
    num_cells = 0
    num_cells_invalid = 0
    for ii in range(Ny):
        for jj in range(Nx):
            nQ_th[ii, jj] = Q_th[ii, jj]
            # If (interior cell) && (velocity > 0) -> Normal rule
            if (Q_th[ii, jj] > 0) and (Q_v[ii, jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                num_cells += 1
                # Deposition initialization:
                sediment_mean_size = 1
                sum_q_cj = 0

                # Erosion initialization:
                log_2_D_sj = np.zeros((Nj), dtype=np.double, order='C')

                for kk in range(Nj):
                    # Deposition part:
                    sum_q_cj += Q_cj[ii, jj, kk]
                    sediment_mean_size *= Q_cj[ii, jj, kk] * D_sj[kk]

                    # Erosion part:
                    log_2_D_sj[kk] = np.log2(D_sj[kk])

                kappa = 1 - 0.288 * np.std(log_2_D_sj)
                sediment_mean_size = sediment_mean_size ** (1 / Nj) / sum_q_cj
                f_sj = np.zeros((Nj), dtype=np.double, order='C')
                f_sj_sum = 0

                for kk in range(Nj):
                    # Deposition part:
                    fall_velocity_dimless = v_sj[kk] ** (3) * rho_a / ((rho_j[kk] - rho_a) * g * nu)
                    near_bed_c = Q_cj[ii, jj, kk] * (0.40 * (D_sj[kk] / sediment_mean_size) ** (1.64) + 1.64)
                    deposition_rate = fall_velocity_dimless * near_bed_c

                    # Erosion part:
                    particle_reynolds = np.sqrt(g * (rho_j[kk] - rho_a) * D_sj[kk] / rho_a) * D_sj[kk] / nu
                    # print("g = {}, rho_j[kk] = {}, rho_a = {}, Dsj = {}, nu = {}".format(g,rho_j[kk],rho_a,D_sj[kk],nu))
                    # print(particle_reynolds)
                    if (particle_reynolds >= 3.5):
                        function_reynolds = particle_reynolds ** (0.6)
                    elif (particle_reynolds > 1) and (particle_reynolds < 3.5):
                        function_reynolds = 0.586 * particle_reynolds ** (1.23)
                    else:
                        raise Exception('Eq. (40) (Salles) not defined for R_pj = {0}'.format(particle_reynolds))
                    Z_mj = kappa * np.sqrt(c_D * Q_v[ii, jj]) * function_reynolds / fall_velocity_dimless
                    erosion_rate = (1.3 * 10 ** (-7) * Z_mj ** (5)) / (1 + 4.3 * 10 ** (-7) * Z_mj ** (5))

                    # Exner equation:
                    f_sj[kk] = deposition_rate - erosion_rate * Q_cbj[ii, jj, kk]
                    f_sj_sum += f_sj[kk]

                nQ_a[ii, jj] = Q_a[ii, jj] + dt / (1 - porosity) * f_sj_sum
                nQ_d[ii, jj] = Q_d[ii, jj] + dt / (1 - porosity) * f_sj_sum

                for kk in range(Nj):
                    nQ_cj[ii, jj, kk] = Q_cj[ii, jj, kk] - dt / ((1 - porosity) * Q_th[ii, jj]) * f_sj[kk]
                    nQ_cbj[ii, jj, kk] = Q_cbj[ii, jj, kk] + dt / ((1 - porosity) * Q_d[ii, jj]) * \
                                         (f_sj[kk] - Q_cbj[ii, jj, kk] * f_sj_sum)
                    # If this operation leads to unphysical state, undo the rule:
                    if (nQ_cj[ii,jj,kk] < 0) or (np.isnan(nQ_cj[ii,jj,kk])):
                        num_cells_invalid += 1
                        nQ_a[ii, jj] = Q_a[ii, jj]
                        nQ_d[ii, jj] = Q_d[ii, jj]
                        for kk in range(Nj):
                            nQ_cj[ii, jj, kk] = Q_cj[ii, jj, kk]
                            nQ_cbj[ii, jj, kk] = Q_cbj[ii, jj, kk]
            # If (interior cell) && (velocity has been zero for two time steps): sediment everything; remove t current
            elif (Q_v_is_zero_two_timesteps[ii, jj] == 1) and (Q_v[ii, jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                nQ_a[ii, jj] = Q_a[ii, jj] + np.sum(Q_cj[ii, jj,:]) * Q_th[ii, jj]
                nQ_d[ii, jj] = Q_d[ii, jj] + np.sum(Q_cj[ii, jj,:]) * Q_th[ii, jj]

                for kk in range(Nj):
                    nQ_cbj[ii, jj, kk] = Q_cbj[ii, jj, kk] + Q_cj[ii, jj, kk] * Q_th[ii, jj]
                    nQ_cj[ii, jj, kk] = 0
                nQ_th[ii, jj] = 0


            # Mainly for border cells or cells away from "action"
            else:
                nQ_a[ii, jj] = Q_a[ii, jj]
                nQ_d[ii, jj] = Q_d[ii, jj]
                for kk in range(Nj):
                    nQ_cj[ii, jj, kk] = Q_cj[ii, jj, kk]
                    nQ_cbj[ii, jj, kk] = Q_cbj[ii, jj, kk]
    # if num_cells == num_cells_invalid:
    #     raise Exception("No cell changed")

    return nQ_a, nQ_d, nQ_cj, nQ_cbj, nQ_th


def I_1(Q_th, Nj, Q_cj, rho_j, rho_a, Q_v, Q_a,
        Ny, Nx, dx, p_f, p_adh, dt, g):
    '''
    This function calculates the turbidity current outflows.\
    IN: Q_a,Q_th,Q_v,Q_cj. OUT: Q_o
    p_f = np.deg2rad(1) # Height threshold friction angle

    '''
    nQ_o = np.zeros((Ny, Nx, 6), dtype=np.double, order='C')  # Reset outflow
    for ii in range(1,Ny-1):
        for jj in range(1,Nx-1):
            if (Q_th[ii, jj] > 0):  # If cell has flow perform algorithm
                g_prime = g * np.sum(Q_cj[ii, jj, :] * (rho_j - rho_a) / rho_a)
                h_k = 0.5 * (Q_v[ii, jj] * Q_v[ii, jj]) / g_prime
                r = Q_th[ii, jj] + h_k
                height_center = Q_a[ii, jj] + r

                A = [0, 1, 2, 3, 4, 5]
                nb_h = []

                nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
                for dir in range(6):
                    nb_i = ii + nb_index[dir][0]
                    nb_j = jj + nb_index[dir][1]

                    nb_h.append(Q_a[nb_i, nb_j] + Q_th[nb_i, nb_j])
                    if np.isnan(nb_h[-1]):
                        raise ValueError
                    if (np.arctan((height_center - nb_h[-1]) / dx) < p_f):
                        del A[dir - (6 - len(A))]

                eliminated = True
                Average = 0
                while eliminated and len(A) > 0:
                    eliminated = False
                    sum_nb_h_in_A = 0
                    for dir in A:
                        sum_nb_h_in_A += nb_h[dir]
                    Average = ((r - p_adh) + sum_nb_h_in_A) / len(A)

                    A_copy = A.copy()
                    for index, dir in enumerate(A):
                        if (nb_h[dir] >= Average):
                            del A_copy[index - (len(A) - len(A_copy))]
                            eliminated = True
                    A = A_copy.copy()

                f = np.zeros((6), dtype=np.double, order='C')
                factor_n = Q_th[ii, jj] / r
                factor_r = np.sqrt(2 * r * g_prime) * dt / (dx / 2)

                for dir in A:
                    f[dir] = Average - nb_h[dir]
                    nQ_o[ii, jj, dir] = f[dir] * factor_n * factor_r
                    if (np.isnan(nQ_o[ii,jj,dir])):
                        raise ValueError

    return nQ_o


def I_2(Ny, Nx, Nj, Q_o, Q_th, Q_cj):
    '''Update thickness and concentration. IN: Q_th,Q_cj,Q_o. OUT: Q_th,Q_cj'''
    nQ_th = Q_th.copy()
    nQ_cj = np.zeros((Ny, Nx, Nj), dtype=np.double, order='C')
    nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    nb_flow_dir = [3, 4, 5, 0, 1, 2]
    # Only update interior cells:
    for ii in range(1, Ny - 1):
        for jj in range(1, Nx - 1):
            Q_o_from_center_sum = 0
            Q_o_Q_cj_neighbors = np.zeros((Nj), dtype=np.double, order='C')
            for kk in range(6):
                # For thickness:
                nb_ii = ii + nb_index[kk][0]
                nb_jj = jj + nb_index[kk][1]
                nQ_th[ii, jj] += Q_o[nb_ii, nb_jj, nb_flow_dir[kk]] - Q_o[ii, jj, kk]

                # For concentration:
                Q_o_from_center_sum += Q_o[ii, jj, kk]
                for ll in range(Nj):
                    Q_o_Q_cj_neighbors[ll] += Q_o[nb_ii, nb_jj, nb_flow_dir[kk]] * Q_cj[nb_ii, nb_jj, ll]
            if (nQ_th[ii, jj] < 0) or np.isnan(nQ_th[ii,jj]):
                raise Exception("I_2: Negative sediment due to excessive outflow!")

            if (nQ_th[ii, jj] > 0):
                for kk in range(Nj):
                    nQ_cj[ii, jj, kk] = 1 / nQ_th[ii, jj] * ((Q_th[ii, jj] - Q_o_from_center_sum) * Q_cj[ii, jj, kk] +
                                                             Q_o_Q_cj_neighbors[kk])
                    if (np.isnan(nQ_cj[ii,jj,kk])) or (nQ_cj[ii,jj,kk] < 0):
                        raise ValueError

    return nQ_th, nQ_cj


def I_3(g, Nj, Q_cj, rho_j, rho_a, Ny, Nx, Q_a, Q_th, Q_o, f, a, Q_v):
    '''
    Update of turbidity flow velocity (speed!). IN: Q_a,Q_th,Q_o,Q_cj. OUT: Q_v.
    '''
    nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    nQ_v = np.zeros((Ny, Nx), dtype=np.double, order='C')
    Q_v_is_zero_two_timesteps = np.zeros((Ny, Nx), dtype=np.int, order='C')
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
                    U += np.sqrt( 8 * g_prime * Q_cj_sum * Q_o[ii,jj,kk] * slope / (f * (1 + a)) )
                    if (np.isnan(U)):
                        raise Exception("U is nan")
                if num_removed < 6:
                    nQ_v[ii,jj] = U/(6 - num_removed)
                if np.isnan(nQ_v[ii, jj]):
                    raise ValueError
            if (Q_v[ii, jj] == 0) and (nQ_v[ii, jj] == 0):
                Q_v_is_zero_two_timesteps[ii, jj] = 1

    return nQ_v, Q_v_is_zero_two_timesteps


def I_4(Q_d, Ny, Nx, Nj, dx, reposeAngle, Q_cbj, Q_a, seaBedDiff):  # Toppling rule
    nQ_d = np.zeros((Ny,Nx), dtype=np.double, order='C')
    nQ_a = np.zeros((Ny,Nx), dtype=np.double, order='C')
    nQ_cbj = np.zeros((Ny,Nx, Nj), dtype=np.double, order='C')

    nb_index = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]
    for ii in range(Ny):
        for jj in range(Nx):
            if (Q_d[ii,jj] > 0) and (ii > 0) and (ii < Ny - 1) and (jj > 0) and (jj < Nx - 1):
                nQ_d[ii, jj] += Q_d[ii, jj]
                nQ_a[ii, jj] += Q_a[ii, jj]
                num_recv = 0
                give_dir = np.zeros((6),np.int, order='C')
                diff = np.zeros((6), dtype=np.double, order='C')
                for kk in range(6):
                    nb_ii = ii + nb_index[kk][0]
                    nb_jj = jj + nb_index[kk][1]
                    diff[kk] = Q_d[ii,jj] - Q_d[nb_ii, nb_jj] + seaBedDiff[ii-1,jj-1,kk]
                    angle = np.arctan2(diff[kk], dx)
                    if angle > reposeAngle:
                        give_dir[num_recv] = kk
                        num_recv += 1
                if num_recv > 0:
                    for kk in range(num_recv):
                        nb_no = give_dir[kk]
                        nb_ii = ii + nb_index[nb_no][0]
                        nb_jj = jj + nb_index[nb_no][1]
                        frac = 0.5 * (diff[nb_no] - dx * np.tan(reposeAngle)) / Q_d[ii,jj]
                        if frac > 0.5:
                            frac = 0.5
                        deltaS = Q_d[ii,jj] * frac/num_recv
                        nQ_d[nb_ii,nb_jj] += Q_d[nb_ii, nb_jj] + deltaS
                        nQ_a[nb_ii,nb_jj] += Q_a[nb_ii, nb_jj] + deltaS

                        nQ_d[ii,jj] -= deltaS
                        nQ_a[ii,jj] -= deltaS

                        for ll in range(Nj):
                            nQ_cbj[nb_ii, nb_jj, ll] = (Q_d[nb_ii, nb_jj] * Q_cbj[nb_ii, nb_jj, ll] +
                                                       Q_cbj[ii,jj, ll] * deltaS) / nQ_d[nb_ii, nb_jj]
                else:
                    for kk in range(Nj):
                        nQ_cbj[ii, jj, kk] = Q_cbj[ii, jj, kk]
            else:
                nQ_d[ii, jj] = Q_d[ii, jj]
                nQ_a[ii, jj] = Q_a[ii, jj]
                for kk in range(Nj):
                    nQ_cbj[ii, jj, kk] = Q_cbj[ii, jj, kk]



    return nQ_a, nQ_d, nQ_cbj
