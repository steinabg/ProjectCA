import matplotlib.pyplot as plt

plt.style.use('bmh')
import numpy as np
import numexpr as ne
from datetime import datetime
# from scipy.ndimage import imread
import mathfunk as ma
import T1functions as T1
import T2functions as T2
import scipy.io
import numba as nb
from mpldatacursor import datacursor




class Hexgrid():
    '''Simulates a turbidity current using a CA. '''

    def __init__(self, Ny, Nx, ICstates=None, reposeAngle=np.deg2rad(0), dx=1, terrain=None,
                 global_grid = True):
        ################ Constants ######################
        self.g = 9.81  # Gravitational acceleration
        self.f = 0.04  # Darcy-Weisbach coeff
        self.a = 0.43  # Empirical coefficient (used in I_3)
        self.rho_a = 1000  # ambient density
        self.rho_j = np.array([2650])  # List of current sediment densities
        self.D_sj = np.array([0.00011])  # List of sediment-particle diameters
        self.Nj = 1  # Number of sediment types
        self.c_D = np.sqrt(0.003)  # Bed drag coefficient (table 3)
        self.nu = 1.5182e-06  # Kinematic viscosity of water at 5 degrees celcius
        self.porosity = 0.3
        self.v_sj = ma.calc_settling_speed(self.D_sj, self.rho_a, self.rho_j, self.g,
                                           self.nu)  # List of sediment-particle fall-velocities
        self.dt = 0  # Some initial value

        # Constants used in I_1:
        self.p_f = np.deg2rad(0)  # Height threshold friction angle
        self.p_adh = 0
        ############## Input variables ###################
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.reposeAngle = reposeAngle

        ################     Grid       ###################
        self.X = np.zeros((Ny, Nx, 2))  # X[:,:,0] = X coords, X[:,:,1] = Y coords
        for j in range(Ny):
            self.X[j, :, 0] = j * dx / 2 + np.arange(Nx) * dx
            self.X[j, :, 1] = -np.ones(Nx) * dx * np.sqrt(3) / 2 * j

        ################# Cell substate storage ####################
        self.Q_a   = np.zeros((self.Ny,self.Nx)) # Cell altitude (bathymetry at t = 0)
        self.Q_th = np.zeros((self.Ny, self.Nx))  # Turbidity current thickness
        self.Q_v = np.zeros((self.Ny, self.Nx))  # Turbidity current speed (scalar)
        self.Q_cj = np.zeros((self.Ny, self.Nx, self.Nj))  # jth current sediment volume concentration
        self.Q_cbj = np.zeros((self.Ny, self.Nx, self.Nj))  # jth bed sediment volume fraction
        if global_grid == True:
            self.Q_d = np.ones((self.Ny, self.Nx)) * np.inf  # Thickness of soft sediment
            self.Q_d[1:-1, 1:-1] = 0
        else:
            self.Q_d = np.zeros((self.Ny,self.Nx))
        if global_grid == True:
            self.Q_a = self.Q_d.copy()  # Bathymetry legges til Q_a i self.setBathymetry(terrain)
        # print(self.Q_a)
        self.Q_o = np.zeros((self.Ny, self.Nx, 6))  # Density current outflow

        ################### Set Initial conditions #####################
        if ICstates is not None: self.set_substate_ICs(ICstates)
        self.CellArea = ma.calc_hexagon_area(dx)
        if global_grid == True:
            self.setBathymetry(terrain)
        self.diff = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        self.seaBedDiff = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        self.calc_bathymetryDiff()

        #         self.totalheight = self.Q_d + self.Q_a

        self.defineNeighbors()

        # FOR DEBUGGING
        self.i = 0
        self.my_rank = 0
        self.unphysical_substate = {'Q_th': 0, 'Q_v': 0, 'Q_cj':0,'Q_cbj':0,'Q_d':0,'Q_o':0}
        self.Erosionrate = []
        self.Depositionrate = []

        self.find_channel_bot() # Find the indices of the channel minima in axial direction

        ################################################################
        ##########################  Methods ############################
        ################################################################

    def find_channel_bot(self):
        self.bot_indices = []
        for i in range(self.Ny):
            self.bot_indices.append((i,np.min(np.where(np.min(self.Q_a[i,:])==self.Q_a[i,:]))))

        # with open(
        #         'bot_indices.txt', 'w') as f:
        #     for item in self.bot_indices:
        #         f.write("%s, %s\n" % (item[0],item[1]))

    def set_substate_ICs(self, ICstates):
        self.Q_th = ICstates[0].copy()
        self.Q_v = ICstates[1].copy()
        self.Q_cj = ICstates[2].copy()
        self.Q_cbj = ICstates[3].copy()
        self.Q_d = ICstates[4].copy()
        self.Q_o = ICstates[5].copy()
        self.Q_a = self.Q_d.copy()

    def defineNeighbors(self):  # Note to self: This works as intended. See testfile in "Testing of functions"
        '''
        This function defines indices that can be used to reference the neighbors of a cell.\
        Use: self.Q_v[self.NEIGHBOR[0]] = NW neighbors' value of Q_v
        '''
        self.NEIGHBOR = []  # y,i                         x,j
        self.NEIGHBOR.append(np.ix_(np.arange(0, self.Ny - 2), np.arange(1, self.Nx - 1)))  # NW
        self.NEIGHBOR.append(np.ix_(np.arange(0, self.Ny - 2), np.arange(2, self.Nx)))  # NE
        self.NEIGHBOR.append(np.ix_(np.arange(1, self.Ny - 1), np.arange(2, self.Nx)))  # E
        self.NEIGHBOR.append(np.ix_(np.arange(2, self.Ny), np.arange(1, self.Nx - 1)))  # SE
        self.NEIGHBOR.append(np.ix_(np.arange(2, self.Ny), np.arange(0, self.Nx - 2)))  # SW
        self.NEIGHBOR.append(np.ix_(np.arange(1, self.Ny - 1), np.arange(0, self.Nx - 2)))  # W
        self.indexMat = np.zeros((self.Ny, self.Nx, 6), dtype=bool)
        for i in range(6):
            self.indexMat[self.NEIGHBOR[i] + (i,)] = 1

    def time_step(self, global_grid = True):
        if global_grid is True:
            self.dt = self.calc_dt()  # Works as long as all ICs are given

        self.T_1()  # Water entrainment.
        self.sanityCheck()
        # self.printSubstates_to_screen('T_1')
        self.T_2()  # Erosion and deposition TODO: Fix cause of instability
        self.sanityCheck()
        # self.printSubstates_to_screen('T_2')
        self.I_1()  # Turbidity c. outflows
        self.sanityCheck()
        # self.printSubstates_to_screen('I_1')
        self.I_2()  # Update thickness and concentration
        self.sanityCheck()
        # self.printSubstates_to_screen('I_2')
        self.I_3()  # Update of turbidity flow velocity
        self.sanityCheck()
        # self.printSubstates_to_screen('I_3')
        self.I_4()  # Toppling rule
        self.sanityCheck()
        # self.printSubstates_to_screen('I_4')



    def sanityCheck(self):
        # Round off numerical error
        # self.Q_cbj[self.Q_cbj > 0 & self.Q_cbj < (1-1e-16)] = 1

        # DEBUGGING: Check for unphysical values
        if (np.any(self.Q_th < 0)):
            self.unphysical_substate['Q_th'] = 1
            raise Exception('unphysical_substate[Q_th]')
        if (np.any(self.Q_v < 0)):
            self.unphysical_substate['Q_v'] = 1
            raise Exception('unphysical_substate[Q_v]')
        if (np.any(self.Q_cj > 1)) | (np.any(self.Q_cj < 0)):
            self.unphysical_substate['Q_cj'] = 1
            raise Exception('unphysical_substate[Q_cj]')
        if (np.any(self.Q_cbj > 1)) | (np.any(self.Q_cbj < 0)):
            self.unphysical_substate['Q_cbj'] = 1
            raise Exception('unphysical_substate[Q_cbj]')
        if (np.any(self.Q_d < 0)):
            self.unphysical_substate['Q_d'] = 1
            raise Exception('unphysical_substate[Q_d]')
        if (np.any(self.Q_o < 0)):
            self.unphysical_substate['Q_o'] = 1
            raise Exception('unphysical_substate[Q_o]')

        t1 = np.sum(self.Q_cbj,2)
        if (np.any(t1) != 1) and np.any(t1) != 0:
            print(np.where(np.sum(self.Q_cbj,2) != 1))
            raise Exception("should always be 1 with nj=1!")


        # If velocity goes to zero. Sediment everything.
        # index = (self.Q_v < 1e-16) & (self.Q_v>0) & (self.Q_th > 0)
        # dep_material = np.zeros((self.Ny,self.Nx,self.Nj))
        # dep_material[index,:] += self.Q_th[index,None] * self.Q_cj[index]
        #
        # for i in range(self.Nj):
        #     self.Q_cbj[:,:,i] += (dep_material[:,:,i] - self.Q_cbj[:,:,i]*np.sum(dep_material,axis=2)) /\
        #                             self.Q_d
        #     self.Q_cj[index, i] = 0
        #
        # self.Q_d += np.sum(dep_material,axis=2)
        # self.Q_a += np.sum(dep_material,axis=2)
        # self.Q_th[index] = 0


        self.Q_cj[self.Q_cj < 1e-16] = 0
        self.Q_cbj[self.Q_cbj < 1e-16] = 0
        self.Q_o[self.Q_o < 1e-16] = 0
        self.Q_d[self.Q_d < 1e-16] = 0
        self.Q_v[self.Q_v < 1e-16] = 0

    def printSubstates_to_screen(self, text):
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(2, 2, i, aspect='equal') for i in range(1, 5)]
        ind = np.unravel_index(np.argmax(self.Q_th, axis=None), self.Q_th.shape)

        points = ax[0].scatter(self.X[:, :, 0].flatten(), self.X[:, :, 1].flatten(), marker='h',
                               c=self.Q_cj[:, :, 0].flatten())

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_cj[:,:,0]. n = ' + text)

        points = ax[1].scatter(self.X[:, :, 0].flatten(), self.X[:, :, 1].flatten(), marker='h',
                               c=self.Q_th.flatten())
        ax[1].scatter(self.X[ind[0],ind[1],0], self.X[ind[0],ind[1],1], c='r')  # Targeting
        plt.colorbar(points, shrink=0.6, ax=ax[1])
        ax[1].set_title('Q_th')

        points = ax[2].scatter(self.X[1:-1, 1:-1, 0].flatten(), self.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.Q_cbj[1:-1, 1:-1, 0].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[2])
        ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

        points = ax[3].scatter(self.X[1:-1, 1:-1, 0].flatten(), self.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.Q_d[1:-1, 1:-1].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[3])
        datacursor(bbox=dict(alpha=1))
        ax[3].set_title('Q_d[1:-1,1:-1]')
        plt.tight_layout()
        plt.show()


    def T_1(self, DEBUG = None):  # Water entrainment. IN: Q_a,Q_th,Q_cj,Q_v. OUT: Q_vj,Q_th
        '''
        This function calculates the water entrainment.\
        Entrainment is the transport of fluid across an interface\
        between two bodies of fluid by a shear induced turbulent flux.
        I.e. the 'mixing' of two fluids across their interface. \

        '''
        #         ipdb.set_trace()
        g_prime = ma.calc_g_prime(self.Nj, self.Q_cj, self.rho_j, self.rho_a, g=self.g)
        if DEBUG is True:
            g_prime[:,:] = 0.5
        Ri = T1.calc_RichardsonNo(g_prime, self.Q_th, self.Q_v)
        # Ri[np.isnan(Ri)] = 0
        # Ri[Ri == 0] = np.inf
        E_wStar = T1.calc_dimlessIncorporationRate(Ri)  # Dimensionless incorporation rate
        E_w = T1.calc_rateOfSeaWaterIncorp(self.Q_v, E_wStar)  # Rate of seawater incorporation
        nQ_th = self.Q_th + T1.calc_changeIn_q_th(E_w, self.dt)  # Update cell current thickness
        # nQ_th[np.isnan(nQ_th)] = 0

        tempQ_cj = T1.calc_new_qcj(self.Q_cj, self.Q_th, nQ_th)
        # tempQ_cj[np.isnan(tempQ_cj)] = 0
        # if (tempQ_cj.sum() - self.Q_cj.sum() > 1e+03):
        #     print("break")

        self.Q_cj[1:-1, 1:-1] = np.round(tempQ_cj[1:-1, 1:-1],15)
        self.Q_th[1:-1, 1:-1] = np.round(nQ_th[1:-1, 1:-1],15)

    def T_2(self):
        '''
        This function updates Q_a,Q_d,Q_cj and Q_cbj. According to erosion and deposition rules.\
        IN: Q_a,Q_th,Q_cj,Q_cbj,Q_v. OUT:
        '''

        R_pj = T2.calc_Rpj(self.rho_j, self.rho_a, self.D_sj, self.nu, g=self.g)  # Assume rho = rho_ambient.
        f = T2.calc_fofR(R_pj)
        kappa = T2.calc_kappa(self.D_sj)
        Ustar = T2.calc_Ustar(self.c_D, self.Q_v)
        v_sjSTAR = self.v_sj**3 * self.rho_a / ((self.rho_j - self.rho_a) * self.g * self.nu) # Eq. (5) Dietrich
        D_sg = T2.calc_averageSedimentSize(self.Q_cj, self.D_sj)
        c_nbj = T2.calc_nearBedConcentration_SusSed(self.D_sj, D_sg, self.Q_cj)

        D_j = np.nan_to_num(T2.calc_depositionRate(v_sjSTAR, c_nbj))
        Z_mj = T2.calc_Z_mj(kappa, Ustar, v_sjSTAR, f)
        E_j = T2.calc_erotionRate(Z_mj)*v_sjSTAR

        # Calculate f_sj: f_sj > 0 => Deposition. f_sj < 0 => Erosion
        f_sj = (D_j - self.Q_cbj*E_j)[1:-1,1:-1,:]

        f_sj[np.logical_and(f_sj > 0, self.Q_th[1:-1,1:-1,None] == 0)] = 0
        f_sj[np.logical_and(f_sj < 0, self.Q_d[1:-1,1:-1,None] == 0)] = 0
        f_sj = np.where((self.dt * f_sj/((1-self.porosity)*self.Q_th[1:-1,1:-1, None])) >= 0.1*self.Q_cj[1:-1,1:-1,:],
                        0.1*self.Q_cj[1:-1,1:-1,:]*(1-self.porosity)*self.Q_th[1:-1,1:-1, None]/self.dt, f_sj)
        f_s = np.sum(f_sj, axis=2)
        self.f_sj = f_sj
        self.f_s = f_s

        # Use old values in equations!
        oldQ_d = self.Q_d.copy()

        # Rescale D_j and E_j to prevent too much material being moved
        # D_j, E_j = T2.rescale_Dj_E_j(D_j, self.dt, self.porosity, self.Q_th, self.Q_cj, self.p_adh, self.Q_cbj, E_j,
        #                              self.Q_d)

        # IF Q_cj = 1 increase the deposition rate D_j to compensate?
        #         temp_delta_qa = T2.T2_calc_change_qd(self.dt,D_j,self.Q_cbj,E_j,self.porosity, oldQ_th, oldQ_cj)

        #         D_j = np.where(self.Q_cj>=0.85, D_j*100,D_j)

        # DEBUGGING!
        self.Erosionrate.append(np.amax(E_j.flatten()))
        self.Depositionrate.append(np.amax(D_j.flatten()))
        #######

        self.Q_a[1:-1, 1:-1] += self.dt*f_s/(1-self.porosity)
        self.Q_d[1:-1, 1:-1] += self.dt*f_s/(1-self.porosity)
        self.Q_cj[1:-1, 1:-1, :] -= np.round(np.nan_to_num(self.dt*f_sj/((1-self.porosity)*self.Q_th[1:-1,1:-1, None])),16)
        self.Q_cbj[1:-1, 1:-1, :] += np.nan_to_num(self.dt/((1-self.porosity) * oldQ_d[1:-1,1:-1, None]) *
                                                   (f_sj - self.Q_cbj[1:-1,1:-1,:] * f_s[:,:,None]))

        # Fail-safe
        # self.Q_cbj[self.Q_cbj > 1] = 1
        # self.Q_cbj[self.Q_cbj < 0] = 0
        # self.Q_cj[self.Q_cj < 0] = 0
        # if np.any( self.Q_th[np.sum(self.Q_cj,axis=2) == 0] > 0 ):
        #     raise Exception("Concentration set to 0 in cell with nonzero thickness!")
        self.Q_th[np.sum(self.Q_cj,axis=2) == 0] = 0 # Cant have thickness if no concentration...



    def I_1(self, DEBUG=None):
        '''
        This function calculates the turbidity current outflows.\
        IN: Q_a,Q_th,Q_v,Q_cj. OUT: Q_o
        self.p_f = np.deg2rad(1) # Height threshold friction angle

        '''
        eligableCells = self.Q_th[1:-1, 1:-1] > 0
        # Step (i): angles beta_i
        g_prime = ma.calc_g_prime(self.Nj, self.Q_cj, self.rho_j, self.rho_a)
        if DEBUG is True:
            g_prime[:,:] = 1
        # h_k = self.calc_BFroudeNo(g_prime)
        g: np.ndarray = g_prime.copy()
        g[g == 0] = np.inf
        h_k = 0.5 * self.Q_v ** 2 / g
        r = self.Q_th + h_k
        #         print("run up height = \n", r)
        central_cell_height = (self.Q_a + r)[1:-1, 1:-1]
        q_i = (self.Q_a + self.Q_th)
        delta = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        delta[:, :, 0] = central_cell_height - q_i[0:self.Ny - 2, 1:self.Nx - 1]
        delta[:, :, 1] = central_cell_height - q_i[0:self.Ny - 2, 2:self.Nx]
        delta[:, :, 2] = central_cell_height - q_i[1:self.Ny - 1, 2:self.Nx]
        delta[:, :, 3] = central_cell_height - q_i[2:self.Ny, 1:self.Nx - 1]
        delta[:, :, 4] = central_cell_height - q_i[2:self.Ny, 0:self.Nx - 2]
        delta[:, :, 5] = central_cell_height - q_i[1:self.Ny - 1, 0:self.Nx - 2]
        delta[np.isinf(delta)] = 0  # q_i is inf at borders. delta = 0 => angle =0 => no transfer
        # print("my rank = {0}\nmy delta[:,:,0]=\n{1}".format(self.my_rank,delta[:,:,0]))

        # debug_delta = np.zeros((6, self.Ny - 2, self.Nx - 2))
        # for i in range(6):
        #     debug_delta[i, :, :] = delta[:, :, i]
        dx = self.dx
        angle = ne.evaluate('arctan2(delta, dx)')
        # debug_angle = np.zeros((6, self.Ny - 2, self.Nx - 2))
        # for i in range(6):
        #     debug_angle[i, :, :] = angle[:, :, i]
        #         print("angle =\n", np.rad2deg(angle))
        indices = angle > self.p_f  # indices(Ny,Nx,6). Dette er basically set A.
        indices *= eligableCells[:, :, np.newaxis]
        #         print("indices\n",indices)
        # debug_indices = np.zeros((6, self.Ny - 2, self.Nx - 2))
        # for i in range(6):
        #     debug_indices[i, :, :] = indices[:, :, i]


        # Average, indices = I_1_doubleforloop(indices, q_i, r, self.Ny, self.Nx, self.p_adh, self.indexMat) # numba function
        for ii in range(6):  # Step (iii) says to go back to step (ii) if a cell is removed.
            NumberOfCellsInA = np.sum(indices, axis=2)  # Cardinality of set A
            #             print("NumberOfCellsInA =\n",NumberOfCellsInA)

            # Step (ii) calculate average
            neighborValues = np.zeros((self.Ny - 2, self.Nx - 2))
            #         print("neighbors=\n", self.NEIGHBOR[0])
            for i in range(6):
                q_i_nb = q_i[self.NEIGHBOR[i]]
                with np.errstate(invalid='ignore'):
                    indices_for_ne = indices[:, :, i]
                    neighborValues_thread = q_i_nb * indices_for_ne
                    # neighborValues_thread = vectorMultiply(q_i_nb, indices_for_ne)
                    neighborValues_thread[np.isnan(neighborValues_thread)] = 0
                    neighborValues += neighborValues_thread
                    # neighborValues += np.nan_to_num(q_i_nb * indices[:, :,i])
                    # Vi vil bare legge til verdier hvor angle>self.p_f
            #             print("neighborValues=\n", neighborValues)
            p = (r - self.p_adh)[1:-1, 1:-1]
            # p[p<0]=0
            with np.errstate(divide='ignore', invalid='ignore'):
                Average = ne.evaluate('(p + neighborValues) / NumberOfCellsInA')
            Average[np.isinf(Average)] = 0  # for når NumberOfCellsInA =0
            Average[np.isnan(Average)] = 0
            #             print("Average=\n", Average)
            #             print("indices=\n", indices)

            # Step (iii) Eliminate adjacent cells i with q_i >= Average from A.
            for i in range(6):  # Skal sette posisjoner (j) hvor q_i (til nabocelle) > average (i celle j) til 0
                nb = q_i[self.NEIGHBOR[i]]
                itemp = (nb >= Average)
                indices[itemp, i] = 0

        # Step (iv)
        Average[np.isinf(Average)] = 0
        nonNormalizedOutFlow = np.ones((Average.shape + (6,))) * Average[:, :, np.newaxis]
        for i in range(6):
            with np.errstate(invalid='ignore'):
                nonNormalizedOutFlow[:, :, i] -= np.nan_to_num(q_i[self.NEIGHBOR[i]] * indices[:, :, i])
        nonNormalizedOutFlow *= indices
        # Step (v)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalization = self.Q_th / r  # nu_nf
            normalization[np.isnan(normalization)] = 0
        #         print("normalization=\n", normalization)
        with np.errstate(invalid='ignore'):
            relaxation = np.sqrt(2 * r * g_prime) * self.dt / (self.dx/2) # dx is intercellular distance. apothem is used in Salles.
            relaxation[relaxation >0.8] = 0.8 # Demper outflow litt
        if np.any(relaxation > 1) or np.any(relaxation < 0): # D'Ambrosio demand
            raise Exception("Warning! I_1 Relaxation > 1!")

        factor = (normalization * relaxation)[1:-1, 1:-1]
        self.Q_o[1:-1, 1:-1,:] = np.round(np.nan_to_num(factor[:,:,None] * nonNormalizedOutFlow), 15)


        if ((np.sum(self.Q_o, axis=2) > self.Q_th)[1:-1,1:-1].sum() > 0):
            ii, jj = np.where(np.sum(self.Q_o,axis=2)>self.Q_th)
            s = ''.join("\nsum(Q_o[%i,%i]) = %.10f > Q_th = %.10f\n" % (ii[x],jj[x],sum(self.Q_o[ii[x],jj[x]]),self.Q_th[ii[x],jj[x]]) for x in range(len(ii)))
            s = s + ''.join("Relaxation = %03f, Normalization = %03f\n" %(relaxation[ii[x],jj[x]], normalization[ii[x],jj[x]]) for x in range(len(ii)))
            raise Exception("I_1 warning! More outflow Q_o than thickness Q_th!" + s)




    def I_2(self):
        '''Update thickness and concentration. IN: Q_th,Q_cj,Q_o. OUT: Q_th,Q_cj'''
        outflowNo = np.array([3, 4, 5, 0, 1, 2])  # Used to find "inflow" to cell from neighbors
        s = np.zeros((self.Ny - 2, self.Nx - 2))
        term2 = np.zeros((self.Ny - 2, self.Nx - 2, self.Nj))
        for i in range(6):
            inn = (self.Q_o[self.NEIGHBOR[i] + (outflowNo[i],)])
            out = self.Q_o[1:-1, 1:-1, i]
            s += (inn - out)
        newq_th = self.Q_th[1:-1, 1:-1] + np.nan_to_num(s)
        term1 = ((self.Q_th - np.sum(self.Q_o, axis=2))[:, :, np.newaxis] * self.Q_cj)[1:-1, 1:-1, :]
        for j in range(self.Nj):
            for i in range(6):
                term2[:, :, j] += self.Q_o[self.NEIGHBOR[i] + (outflowNo[i],)] * self.Q_cj[self.NEIGHBOR[i] + (j,)]
        with np.errstate(invalid='ignore'):
            newq_cj = (term1 + term2) / newq_th[:, :, np.newaxis]
        newq_cj[np.isinf(newq_cj)] = 0
        self.Q_th[1:-1, 1:-1] = np.round(np.nan_to_num(newq_th),15)
        self.Q_cj[1:-1, 1:-1, :] = np.round(np.nan_to_num(newq_cj),15)


    def I_3(self, DEBUG = None):  # Should be done
        '''
        Update of turbidity flow velocity (speed!). IN: Q_a,Q_th,Q_o,Q_cj. OUT: Q_v.
        '''
        #         ipdb.set_trace()
        # g_prime = np.ndarray(Ny,Nx)
        g_prime = ma.calc_g_prime(self.Nj, self.Q_cj, self.rho_j, self.rho_a)
        if DEBUG is True:
            g_prime[:,:] = 1
        #         print("self.Q_cj=\n",self.Q_cj)
        #         print("g_prime.shape=",g_prime.shape)
        #         print("self.Q_cj.shape=",self.Q_cj.shape)
        #         print("g_prime I_3 = ", g_prime)
        #         print("g_prime =\n", g_prime)

        sum_q_cj = np.sum(self.Q_cj, axis=2)  # TCurrent sediment volume concentration
        # #         print("sum_q_cj = ", sum_q_cj)
        # #         q_o = self.Q_o[1:-1,1:-1,:]
        # #         print("q_o = ", q_o)
        # #         self.calc_Hdiff()

        U_k = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        # #         print("diff=\n",diff[:,:,0])
        # #         diff[np.isinf(diff)] = 0
        diff = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        sum1 = self.Q_a + self.Q_th
        for i in range(6):
            diff[:, :, i] = np.abs((sum1)[1:-1, 1:-1] - (sum1)[self.NEIGHBOR[i]])
        diff[np.isinf(diff)] = 0  # For borders. diff = 0 => U_k = 0. ok.
        # diff[diff<0] = 0 # To avoid negative values in np.sqrt()

        for i in range(6):
            comp1 = (8 * g_prime * sum_q_cj)[1:-1, 1:-1] / (self.f * (1 + self.a))
            comp2 = (self.Q_o[1:-1, 1:-1, i] * diff[:, :, i])
            comp2[comp2<0] = 0 # TODO: Test om denne kan fjernes
            with np.errstate(invalid='raise'):
                temp = np.sqrt(comp1 * comp2)
            U_k[:, :, i] = temp
        # #             print("U_k[:,:,i]=\n",U_k[:,:,i])
        self.Q_v[1:-1, 1:-1] = np.round(np.nan_to_num(ma.average_speed_hexagon(U_k)),15)


    def I_4(self):  # Toppling rule
        interiorH = self.Q_d[1:self.Ny - 1, 1:self.Nx - 1]

        # angle = np.zeros((self.Ny - 2, self.Ny - 2, 6))
        indices = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        NoOfTrans = np.zeros((self.Ny - 2, self.Nx - 2))
        frac = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        deltaS = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        deltaSSum = np.zeros((self.Ny - 2, self.Nx - 2))

        self.calc_Hdiff()
        diff = self.diff

        # Find angles
        dx = self.dx
        angle = ne.evaluate('arctan2(diff,dx)')

        # (Checks if cell (i,j) has angle > repose angle and that it has mass > 0. For all directions.)
        # Find cells (i,j) for which to transfer mass in the direction given
        for i in np.arange(6):
            indices[:, :, i] = np.logical_and(angle[:, :, i] > self.reposeAngle, (
                    interiorH > 0))  # Gives indices (i,j) where the current angle > repose angle and where height is > 0

        # Count up the number of cells (i,j) will be transfering mass to. If none, set (i,j) to infinity so that division works.
        #         NoOfTrans = np.sum(indices,axis=2)  # Gir tregere resultat?
        for i in np.arange(6):
            NoOfTrans += indices[:, :, i]
        NoOfTrans[NoOfTrans == 0] = np.inf

        # Calculate fractions of mass to be transfered
        for i in np.arange(6):
            frac[(indices[:, :, i] > 0), i] = (
                    0.5 * (diff[(indices[:, :, i] > 0), i] - self.dx * np.tan(self.reposeAngle)) / (
                interiorH[(indices[:, :, i] > 0)]))
        frac[frac > 0.5] = 0.5
        #         print("frac.shape=",frac.shape)

        for i in np.arange(6):
            deltaS[(indices[:, :, i] > 0), i] = interiorH[(indices[:, :, i] > 0)] * frac[(indices[:, :, i] > 0), i] / \
                                                NoOfTrans[(indices[:, :,
                                                           i] > 0)]  # Mass to be transfered from index [i,j] to index [i-1,j]

        # Lag en endringsmatrise deltaSSum som kan legges til self.Q_d
        # Trekk fra massen som skal sendes ut fra celler
        deltaSSum = -np.sum(deltaS, axis=2)

        # Legg til massen som skal tas imot. BRUK self.NEIGHBOR
        deltaSSum += np.roll(np.roll(deltaS[:, :, 0], -1, 0), 0, 1)
        deltaSSum += np.roll(np.roll(deltaS[:, :, 1], -1, 0), 1, 1)
        deltaSSum += np.roll(np.roll(deltaS[:, :, 2], 0, 0), 1, 1)
        deltaSSum += np.roll(np.roll(deltaS[:, :, 3], 1, 0), 0, 1)
        deltaSSum += np.roll(np.roll(deltaS[:, :, 4], 1, 0), -1, 1)
        deltaSSum += np.roll(np.roll(deltaS[:, :, 5], 0, 0), -1, 1)

        oldQ_d = self.Q_d.copy()
        self.Q_d[1:-1, 1:-1] += deltaSSum
        self.Q_a[1:-1, 1:-1] += deltaSSum
        # Legg inn endring i volum fraksjon Q_cbj
        prefactor = 1 / self.Q_d[1:-1, 1:-1, np.newaxis]
        prefactor[np.isinf(prefactor)] = 0
        nq_cbj = np.nan_to_num(prefactor *
                               (oldQ_d[1:-1, 1:-1, np.newaxis] * self.Q_cbj[1:-1, 1:-1, :] + deltaSSum[:, :,
                                                                                             None]))  # TODO usikker på om denne blir rett!
        nq_cbj = np.round(nq_cbj,15)
        self.Q_cbj[1:-1, 1:-1] = nq_cbj
        self.Q_cbj[self.Q_cbj < 1e-15] = 0
        if (self.Q_d < -1e-7).sum() > 0:
            print('height', self.Q_d[1, 6])
            raise RuntimeError('Negative sediment thickness!')

    def setBathymetry(self, terrain):
        if terrain is not None:
            x = np.linspace(0, 100, self.Nx)
            y = np.linspace(0, 100, self.Ny)
            X = np.array(np.meshgrid(x, y))
            temp = np.zeros((self.Ny, self.Nx))
            if terrain == 'river':
                temp = -2 * X[1, :] + 5 * np.abs(X[0, :] - 50 + 10 * np.sin(X[1, :] / 10))
                #                 temp = 2*self.X[:,:,1] + 5*np.abs(self.X[:,:,0] + 10*np.sin(self.X[:,:,1]/10))
                self.Q_a += temp  # BRUK MED RIVER
            elif terrain == 'river_shallow':
                temp = -1 * X[1, :] + 1 * np.abs(X[0, :] - 50 + 5 * np.sin(X[1, :] / 10))
                #                 temp = 2*self.X[:,:,1] + 5*np.abs(self.X[:,:,0] + 10*np.sin(self.X[:,:,1]/10))
                self.Q_a += temp  # BRUK MED RIVER
            elif terrain == 'pit':
                temp = np.sqrt((X[0, :] - 50) * (X[0, :] - 50) + (X[1, :] - 50) * (X[1, :] - 50))
                self.Q_a += 10 * temp
            elif terrain == 'rupert':
                # mat = scipy.io.loadmat('rupert_inlet_200x200.mat')
                # mat['X'] = np.transpose(mat['X'])
                # self.Q_a += mat['X']
                temp, junk = ma.generate_rupert_inlet_bathymetry(self.reposeAngle, self.dx, self.Ny,self.Nx)
                self.Q_a += np.transpose(temp)

    def calc_bathymetryDiff(self):
        with np.errstate(invalid='ignore'):
            temp = self.Q_a - self.Q_d
        self.seaBedDiff[:, :, 0] = temp[1:-1, 1:-1] - temp[0:self.Ny - 2, 1:self.Nx - 1]
        self.seaBedDiff[:, :, 1] = temp[1:-1, 1:-1] - temp[0:self.Ny - 2, 2:self.Nx]
        self.seaBedDiff[:, :, 2] = temp[1:-1, 1:-1] - temp[1:self.Ny - 1, 2:self.Nx]
        self.seaBedDiff[:, :, 3] = temp[1:-1, 1:-1] - temp[2:self.Ny, 1:self.Nx - 1]
        self.seaBedDiff[:, :, 4] = temp[1:-1, 1:-1] - temp[2:self.Ny, 0:self.Nx - 2]
        self.seaBedDiff[:, :, 5] = temp[1:-1, 1:-1] - temp[1:self.Ny - 1, 0:self.Nx - 2]
        self.seaBedDiff[np.isnan(self.seaBedDiff)] = 0

    def calc_Hdiff(self):
        ''' Calculates the height difference between center cell and neighbors.
            diff[i,j,k] is the '''
        old_height = self.Q_d
        interiorH = old_height[1:-1, 1:-1]
        # Calculate height differences of all neighbors
        self.diff[:, :, 0] = interiorH - old_height[0:self.Ny - 2, 1:self.Nx - 1] + self.seaBedDiff[:, :, 0]
        self.diff[:, :, 1] = interiorH - old_height[0:self.Ny - 2, 2:self.Nx] + self.seaBedDiff[:, :, 1]
        self.diff[:, :, 2] = interiorH - old_height[1:self.Ny - 1, 2:self.Nx] + self.seaBedDiff[:, :, 2]
        self.diff[:, :, 3] = interiorH - old_height[2:self.Ny, 1:self.Nx - 1] + self.seaBedDiff[:, :, 3]
        self.diff[:, :, 4] = interiorH - old_height[2:self.Ny, 0:self.Nx - 2] + self.seaBedDiff[:, :, 4]
        self.diff[:, :, 5] = interiorH - old_height[1:self.Ny - 1, 0:self.Nx - 2] + self.seaBedDiff[:, :, 5]

    def calc_BFroudeNo(self, g_prime):  # out: Bulk Froude No matrix
        U = self.Q_v
        g: np.ndarray = g_prime.copy()
        # g_prime[g_prime == 0] = np.inf
        g[g == 0] = np.inf
        return 0.5 * U ** 2 / g

    def calc_RunUpHeight(self, g_prime):  # out: Run up height matrix
        h_k = self.calc_BFroudeNo(g_prime)
        return self.Q_th + h_k

    def calc_MaxRelaxationTime(self):  # out: matrix
        g_prime = ma.calc_g_prime(self.Nj, self.Q_cj, self.rho_j, self.rho_a, g=self.g)
        r_j = self.calc_RunUpHeight(g_prime)
        r_j[r_j == 0] = np.inf
        g_prime[g_prime == 0] = np.inf
        return (self.dx / 2) / np.sqrt(2 * r_j * g_prime)

    def calc_dt(self, global_grid=True):
        temp = self.calc_MaxRelaxationTime()
        try:
            dt = np.min([np.amin(temp[np.isfinite(temp) & (~np.isnan(temp)) & (temp > 0)]), 0.2]) # Better stability
        except:
            if global_grid is True:
                dt = 0.01
            else:
                dt = 9999999 # Set a large number so we can use MPI.Reduce MIN.
        return dt

    def printCA(self):
        outflowNo = np.array(['NW', 'NE', 'E', 'SE', 'SW', 'W'])
        try:
            print("Time step = ", self.dt)
        except:
            print("No dt defined yet!")
        print("self.Q_th =\n", self.Q_th)
        print("self.Q_v  =\n", self.Q_v)

        for i in range(1):
            print("self.Q_cj =\n", self.Q_cj[:, :, i])
        for i in range(1):
            print("self.Q_cbj=\n", self.Q_cbj[:, :, i])
        print("self.Q_d  =\n", self.Q_d)
        print("self.Q_a  =\n", self.Q_a)
        # for i in range(6):
        #     print("self.Q_o[",outflowNo[i],"]=\n", self.Q_o[:,:,i])

@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.boolean[:,:,:]))(nb.boolean[:,:,:],nb.float64[:,:], nb.float64[:,:], nb.int32, nb.int32, nb.float32, nb.boolean[:,:,:]), nopython=True)
def I_1_doubleforloop(indices, q_i, r, Ny, Nx, p_adh, neighborIndexMat):
    for ii in range(6):  # Step (iii) says to go back to step (ii) if a cell is removed.
        NumberOfCellsInA = np.sum(indices, axis=2)  # Cardinality of set A
        neighborValues = np.zeros((Ny - 2, Nx - 2))
        q_i_nb = np.zeros((Ny - 2, Nx - 2, 6))
        for i in range(6):

            x = 0
            y = 0
            # if (ii == 0) and (i == 0):
            #     print(neighborIndexMat[:,:,0])
            for jj in range(Ny):
                # if (ii == 0) and (i == 0):
                #     print(q_i_nb)
                for mm in range(Nx):
                    # if (ii == 0) and (i==0):
                    #     print(jj,mm)
                    if (neighborIndexMat[jj,mm,i] == 1):

                        q_i_nb[y,x,i] = q_i[jj,mm] # Build matrix

                        # if (ii == 0) and (i == 0):
                        #     print("q_i_nb[",y,",",x,"] = q_i[",jj,",",mm,"] = ", q_i[jj,mm])

                        if ((mm < Nx-1)):
                            if (neighborIndexMat[jj,mm,i] == 1) and (neighborIndexMat[jj,mm+1,i] == 0):
                                y = y + 1
                                x = 0
                            else:
                                x += 1
                        elif(mm >= Nx-1):
                            y = y + 1
                            x = 0
                        else:
                            x += 1

            # if(i ==0) and (ii==0): print(q_i_nb)
            # q_i_nb = q_i[neighborIndexMat[:,:,i]]

            indices_for_ne = indices[:, :, i]
            neighborValues_thread = q_i_nb[:,:,i] * indices_for_ne
            for jj in nb.prange(Ny-2):
                for mm in range(Nx-2):
                    if np.isnan(neighborValues_thread[jj,mm]):
                        neighborValues_thread[jj,mm] = 0
            # neighborValues_thread[np.isnan(neighborValues_thread)] = 0
            neighborValues += neighborValues_thread
        p = (r - p_adh)[1:-1, 1:-1]
        Average = (p + neighborValues) / NumberOfCellsInA
        for jj in range(Ny-2):
            for mm in range(Nx-2):
                if (np.isinf(Average[jj,mm])) or (np.isnan(Average[jj,mm])):
                    Average[jj,mm] = 0
        # Average[np.isinf(Average)] = 0  # for når NumberOfCellsInA =0
        # Average[np.isnan(Average)] = 0
        # Step (iii) Eliminate adjacent cells i with q_i >= Average from A.
        for j in nb.prange(Ny-2):
            for k in range(Nx-2):
                for i in range(6):
                    if (q_i_nb[j,k,i] >= Average[j,k]):
                        indices[j,k,i] = 0



    return Average, indices