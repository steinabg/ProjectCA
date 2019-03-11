import numpy as np
import T1functions as T1
import T2functions as T2
import mathfunk as ma

def T_1(Ny,Nx,Nj, Q_cj, rho_j, rho_a, Q_th, Q_v, dt, g,  DEBUG=None):  # Water entrainment. IN: Q_a,Q_th,Q_cj,Q_v. OUT: Q_vj,Q_th
    '''
    This function calculates the water entrainment.\
    Entrainment is the transport of fluid across an interface\
    between two bodies of fluid by a shear induced turbulent flux.
    I.e. the 'mixing' of two fluids across their interface. \

    '''
    nQ_th = np.zeros((Ny,Nx),dtype=np.double,order='C')
    nQ_cj = np.zeros((Ny,Nx,Nj),dtype=np.double,order='C')
    for ii in range(Ny):
        for jj in range(Nx):
            if(Q_th[ii,jj] > 0):
                g_prime = 0
                for zz in range(Nj):
                    g_prime += g*(Q_cj[ii,jj,zz] * (rho_j[zz] - rho_a)/rho_a)
                Ri_number = g_prime*Q_th[ii,jj]/(Q_v[ii,jj] * Q_v[ii,jj])
                dimless_entrainment_rate = 0.075 / np.sqrt(1 + 718 * (Ri_number)**(2.4))
                entrainment_rate = Q_v[ii,jj]*dimless_entrainment_rate

                nQ_th[ii,jj] = Q_th[ii,jj] + entrainment_rate * dt
                for zz in range(Nj):
                    nQ_cj[ii,jj,zz] = Q_cj[ii,jj,zz] * Q_th[ii,jj] / nQ_th[ii,jj]


    return nQ_cj, nQ_th


def T_2(rho_j, rho_a, D_sj, nu, g, c_D, Q_v, v_sj, Q_cj, Q_cbj, Q_th, Q_d, dt, porosity, Q_a, Erosionrate, Depositionrate):
    '''
    This function updates Q_a,Q_d,Q_cj and Q_cbj. According to erosion and deposition rules.\
    IN: Q_a,Q_th,Q_cj,Q_cbj,Q_v. OUT:
    '''

    R_pj = T2.calc_Rpj(rho_j, rho_a, D_sj, nu, g=g)  # Assume rho = rho_ambient.
    f = T2.calc_fofR(R_pj)
    kappa = T2.calc_kappa(D_sj)
    Ustar = T2.calc_Ustar(c_D, Q_v)
    v_sjSTAR = v_sj**3 * rho_a / ((rho_j - rho_a) * g * nu) # Eq. (5) Dietrich
    D_sg = T2.calc_averageSedimentSize(Q_cj, D_sj)
    c_nbj = T2.calc_nearBedConcentration_SusSed(D_sj, D_sg, Q_cj)

    D_j = np.nan_to_num(T2.calc_depositionRate(v_sjSTAR, c_nbj))
    Z_mj = T2.calc_Z_mj(kappa, Ustar, v_sjSTAR, f)
    E_j = T2.calc_erotionRate(Z_mj)*v_sjSTAR

    # Calculate f_sj: f_sj > 0 => Deposition. f_sj < 0 => Erosion
    f_sj = (D_j - Q_cbj*E_j)[1:-1,1:-1,:]

    f_sj[np.logical_and(f_sj > 0, Q_th[1:-1,1:-1,None] == 0)] = 0
    f_sj[np.logical_and(f_sj < 0, Q_d[1:-1,1:-1,None] == 0)] = 0
    f_sj = np.where((dt * f_sj/((1-porosity)*Q_th[1:-1,1:-1, None])) >= 0.1*Q_cj[1:-1,1:-1,:],
                    0.1*Q_cj[1:-1,1:-1,:]*(1-porosity)*Q_th[1:-1,1:-1, None]/dt, f_sj)
    f_s = np.sum(f_sj, axis=2)
    f_sj = f_sj
    f_s = f_s

    # Use old values in equations!
    oldQ_d = Q_d.copy()

    # Rescale D_j and E_j to prevent too much material being moved
    # D_j, E_j = T2.rescale_Dj_E_j(D_j, dt, porosity, Q_th, Q_cj, p_adh, Q_cbj, E_j,
    #                              Q_d)

    # IF Q_cj = 1 increase the deposition rate D_j to compensate?
    #         temp_delta_qa = T2.T2_calc_change_qd(dt,D_j,Q_cbj,E_j,porosity, oldQ_th, oldQ_cj)

    #         D_j = np.where(Q_cj>=0.85, D_j*100,D_j)

    # DEBUGGING!
    Erosionrate.append(np.amax(E_j.flatten()))
    Depositionrate.append(np.amax(D_j.flatten()))
    #######

    Q_a[1:-1, 1:-1] += dt*f_s/(1-porosity)
    Q_d[1:-1, 1:-1] += dt*f_s/(1-porosity)
    Q_cj[1:-1, 1:-1, :] -= np.round(np.nan_to_num(dt*f_sj/((1-porosity)*Q_th[1:-1,1:-1, None])),16)
    Q_cbj[1:-1, 1:-1, :] += np.nan_to_num(dt/((1-porosity) * oldQ_d[1:-1,1:-1, None]) *
                                               (f_sj - Q_cbj[1:-1,1:-1,:] * f_s[:,:,None]))

    # Fail-safe
    # Q_cbj[Q_cbj > 1] = 1
    # Q_cbj[Q_cbj < 0] = 0
    # Q_cj[Q_cj < 0] = 0
    # if np.any( Q_th[np.sum(Q_cj,axis=2) == 0] > 0 ):
    #     raise Exception("Concentration set to 0 in cell with nonzero thickness!")
    Q_th[np.sum(Q_cj,axis=2) == 0] = 0 # Cant have thickness if no concentration...
    return Q_a, Q_d, Q_cj, Q_cbj


def I_1(Q_th, Nj, Q_cj, rho_j, rho_a, Q_v, Q_a,
         Ny, Nx, dx, p_f, NEIGHBOR, p_adh, dt, Q_o,  g, DEBUG=None):
    '''
    This function calculates the turbidity current outflows.\
    IN: Q_a,Q_th,Q_v,Q_cj. OUT: Q_o
    p_f = np.deg2rad(1) # Height threshold friction angle

    '''
    Q_o = np.zeros((Ny,Nx,6), dtype=np.double, order='C') # Reset outflow
    for ii in range(Ny):
        for jj in range(Nx):
            if (Q_th[ii,jj] > 0): # If cell has flow perform algorithm
                g_prime = g*np.sum(Q_cj[ii,jj,:] * (rho_j - rho_a)/rho_a)
                h_k = 0.5* Q_v[ii,jj]**2/g_prime
                r = Q_th[ii,jj] + h_k
                height_center = Q_a[ii,jj] + r

                A = [0,1,2,3,4,5]
                nb_h = []

                nb_index = [[-1,0],[-1,1],[0,1],[1,0],[1,-1],[0,-1]]
                for dir in range(6):
                    nb_i = ii + nb_index[dir][0]
                    nb_j = jj + nb_index[dir][1]

                    nb_h.append(Q_a[nb_i,nb_j] + Q_th[nb_i,nb_j])
                    if (np.arctan((height_center - nb_h[-1])/dx) < p_f):
                        del A[dir - (6-len(A))]

                eliminated = True
                Average = 0
                while eliminated and len(A) > 0:
                    eliminated = False
                    sum_nb_h_in_A = 0
                    for dir in A:
                        sum_nb_h_in_A += nb_h[dir]
                    Average = ( (r-p_adh) + sum_nb_h_in_A) / len(A)

                    A_copy = A.copy()
                    for index, dir in enumerate(A):
                        if(nb_h[dir] >= Average):
                            del A_copy[index - (len(A) - len(A_copy))]
                            eliminated = True
                    A = A_copy.copy()

                f = np.zeros((6),dtype=np.double, order='C')
                factor_n = Q_th[ii,jj]/r
                factor_r = np.sqrt(2*r*g_prime)*dt/(dx/2)

                for dir in A:
                    f[dir] = Average - nb_h[dir]
                    Q_o[ii,jj,dir] = f[dir] * factor_n * factor_r

    return Q_o


def I_2(Ny,Nx, Nj, Q_o, NEIGHBOR, Q_th, Q_cj):
    '''Update thickness and concentration. IN: Q_th,Q_cj,Q_o. OUT: Q_th,Q_cj'''
    outflowNo = np.array([3, 4, 5, 0, 1, 2])  # Used to find "inflow" to cell from neighbors
    s = np.zeros((Ny - 2, Nx - 2))
    term2 = np.zeros((Ny - 2, Nx - 2, Nj))
    for i in range(6):
        inn = (Q_o[NEIGHBOR[i] + (outflowNo[i],)])
        out = Q_o[1:-1, 1:-1, i]
        s += (inn - out)
    newq_th = Q_th[1:-1, 1:-1] + np.nan_to_num(s)
    term1 = ((Q_th - np.sum(Q_o, axis=2))[:, :, np.newaxis] * Q_cj)[1:-1, 1:-1, :]
    for j in range(Nj):
        for i in range(6):
            term2[:, :, j] += Q_o[NEIGHBOR[i] + (outflowNo[i],)] * Q_cj[NEIGHBOR[i] + (j,)]
    with np.errstate(invalid='ignore'):
        newq_cj = (term1 + term2) / newq_th[:, :, np.newaxis]
    newq_cj[np.isinf(newq_cj)] = 0
    Q_th[1:-1, 1:-1] = np.round(np.nan_to_num(newq_th),15)
    Q_cj[1:-1, 1:-1, :] = np.round(np.nan_to_num(newq_cj),15)
    return Q_th, Q_cj

def I_3(Nj, Q_cj, rho_j, rho_a, Ny, Nx, Q_a, Q_th, NEIGHBOR, Q_o, Q_v, f, a, DEBUG = None):  # Should be done
    '''
    Update of turbidity flow velocity (speed!). IN: Q_a,Q_th,Q_o,Q_cj. OUT: Q_v.
    '''
    #         ipdb.set_trace()
    # g_prime = np.ndarray(Ny,Nx)
    g_prime = ma.calc_g_prime(Nj, Q_cj, rho_j, rho_a)
    if DEBUG is True:
        g_prime[:,:] = 1
    #         print("Q_cj=\n",Q_cj)
    #         print("g_prime.shape=",g_prime.shape)
    #         print("Q_cj.shape=",Q_cj.shape)
    #         print("g_prime I_3 = ", g_prime)
    #         print("g_prime =\n", g_prime)

    sum_q_cj = np.sum(Q_cj, axis=2)  # TCurrent sediment volume concentration
    # #         print("sum_q_cj = ", sum_q_cj)
    # #         q_o = Q_o[1:-1,1:-1,:]
    # #         print("q_o = ", q_o)
    # #         calc_Hdiff()

    U_k = np.zeros((Ny - 2, Nx - 2, 6))
    # #         print("diff=\n",diff[:,:,0])
    # #         diff[np.isinf(diff)] = 0
    diff = np.zeros((Ny - 2, Nx - 2, 6))
    sum1 = Q_a + Q_th
    for i in range(6):
        diff[:, :, i] = np.abs((sum1)[1:-1, 1:-1] - (sum1)[NEIGHBOR[i]])
    diff[np.isinf(diff)] = 0  # For borders. diff = 0 => U_k = 0. ok.
    # diff[diff<0] = 0 # To avoid negative values in np.sqrt()

    for i in range(6):
        comp1 = (8 * g_prime * sum_q_cj)[1:-1, 1:-1] / (f * (1 + a))
        comp2 = (Q_o[1:-1, 1:-1, i] * diff[:, :, i])
        comp2[comp2<0] = 0 # TODO: Test om denne kan fjernes
        with np.errstate(invalid='raise'):
            temp = np.sqrt(comp1 * comp2)
        U_k[:, :, i] = temp
    # #             print("U_k[:,:,i]=\n",U_k[:,:,i])
    Q_v[1:-1, 1:-1] = np.round(np.nan_to_num(ma.average_speed_hexagon(U_k)),15)
    return Q_v

def I_4(Q_d, Ny, Nx, dx, reposeAngle, Q_cbj, Q_a, seaBedDiff):  # Toppling rule
    # angle = np.zeros((Ny - 2, Ny - 2, 6))
    indices = np.zeros((Ny - 2, Nx - 2, 6))
    NoOfTrans = np.zeros((Ny - 2, Nx - 2))
    frac = np.zeros((Ny - 2, Nx - 2, 6))
    deltaS = np.zeros((Ny - 2, Nx - 2, 6))
    deltaSSum = np.zeros((Ny - 2, Nx - 2))
    diff = np.zeros((Ny - 2, Nx - 2, 6))

    interiorH = Q_d[1:-1, 1:-1]
    old_height = Q_d.copy()
    # Calculate height differences of all neighbors
    diff[:, :, 0] = interiorH - old_height[0:Ny - 2, 1:Nx - 1] + seaBedDiff[:, :, 0]
    diff[:, :, 1] = interiorH - old_height[0:Ny - 2, 2:Nx] + seaBedDiff[:, :, 1]
    diff[:, :, 2] = interiorH - old_height[1:Ny - 1, 2:Nx] + seaBedDiff[:, :, 2]
    diff[:, :, 3] = interiorH - old_height[2:Ny, 1:Nx - 1] + seaBedDiff[:, :, 3]
    diff[:, :, 4] = interiorH - old_height[2:Ny, 0:Nx - 2] + seaBedDiff[:, :, 4]
    diff[:, :, 5] = interiorH - old_height[1:Ny - 1, 0:Nx - 2] + seaBedDiff[:, :, 5]



    # Find angles
    angle = np.arctan2(diff,dx)

    # (Checks if cell (i,j) has angle > repose angle and that it has mass > 0. For all directions.)
    # Find cells (i,j) for which to transfer mass in the direction given
    for i in np.arange(6):
        indices[:, :, i] = np.logical_and(angle[:, :, i] > reposeAngle, (
                interiorH > 0))  # Gives indices (i,j) where the current angle > repose angle and where height is > 0

    # Count up the number of cells (i,j) will be transfering mass to. If none, set (i,j) to infinity so that division works.
    #         NoOfTrans = np.sum(indices,axis=2)  # Gir tregere resultat?
    for i in np.arange(6):
        NoOfTrans += indices[:, :, i]
    NoOfTrans[NoOfTrans == 0] = np.inf

    # Calculate fractions of mass to be transfered
    for i in np.arange(6):
        frac[(indices[:, :, i] > 0), i] = (
                0.5 * (diff[(indices[:, :, i] > 0), i] - dx * np.tan(reposeAngle)) / (
            interiorH[(indices[:, :, i] > 0)]))
    frac[frac > 0.5] = 0.5
    #         print("frac.shape=",frac.shape)

    for i in np.arange(6):
        deltaS[(indices[:, :, i] > 0), i] = interiorH[(indices[:, :, i] > 0)] * frac[(indices[:, :, i] > 0), i] / \
                                            NoOfTrans[(indices[:, :,
                                                       i] > 0)]  # Mass to be transfered from index [i,j] to index [i-1,j]

    # Lag en endringsmatrise deltaSSum som kan legges til Q_d
    # Trekk fra massen som skal sendes ut fra celler
    deltaSSum = -np.sum(deltaS, axis=2)

    # Legg til massen som skal tas imot. BRUK NEIGHBOR
    deltaSSum += np.roll(np.roll(deltaS[:, :, 0], -1, 0), 0, 1)
    deltaSSum += np.roll(np.roll(deltaS[:, :, 1], -1, 0), 1, 1)
    deltaSSum += np.roll(np.roll(deltaS[:, :, 2], 0, 0), 1, 1)
    deltaSSum += np.roll(np.roll(deltaS[:, :, 3], 1, 0), 0, 1)
    deltaSSum += np.roll(np.roll(deltaS[:, :, 4], 1, 0), -1, 1)
    deltaSSum += np.roll(np.roll(deltaS[:, :, 5], 0, 0), -1, 1)

    oldQ_d = Q_d.copy()
    Q_d[1:-1, 1:-1] += deltaSSum
    Q_a[1:-1, 1:-1] += deltaSSum
    # Legg inn endring i volum fraksjon Q_cbj
    prefactor = 1 / Q_d[1:-1, 1:-1, np.newaxis]
    prefactor[np.isinf(prefactor)] = 0
    nq_cbj = np.nan_to_num(prefactor *
                           (oldQ_d[1:-1, 1:-1, np.newaxis] * Q_cbj[1:-1, 1:-1, :] + deltaSSum[:, :, None]))
    nq_cbj = np.round(nq_cbj,15)
    Q_cbj[1:-1, 1:-1] = nq_cbj
    Q_cbj[Q_cbj < 1e-15] = 0
    if (Q_d < -1e-7).sum() > 0:
        print('height', Q_d[1, 6])
        raise RuntimeError('Negative sediment thickness!')
    return Q_a, Q_d, Q_cbj