import matplotlib.pyplot as plt

plt.style.use('bmh')
import numpy as np
# from scipy.ndimage import imread
import mathfunk as ma
import transition_functions_cy as tra
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
        self.Q_a   = np.zeros((self.Ny,self.Nx),order='C', dtype=np.double) # Cell altitude (bathymetry at t = 0)
        self.Q_th = np.zeros((self.Ny, self.Nx),order='C', dtype=np.double)  # Turbidity current thickness
        self.Q_v = np.zeros((self.Ny, self.Nx), order='C',dtype=np.double)  # Turbidity current speed (scalar)
        self.Q_cj = np.zeros((self.Ny, self.Nx, self.Nj), order='C',dtype=np.double)  # jth current sediment volume concentration
        self.Q_cbj = np.zeros((self.Ny, self.Nx, self.Nj),order='C', dtype=np.double)  # jth bed sediment volume fraction
        if global_grid == True:
            self.Q_d = np.ones((self.Ny, self.Nx),order='C', dtype=np.double) * np.inf  # Thickness of soft sediment
            self.Q_d[1:-1, 1:-1] = 0
        else:
            self.Q_d = np.zeros((self.Ny,self.Nx),order='C', dtype=np.double)
        if global_grid == True:
            self.Q_a = self.Q_d.copy()  # Bathymetry legges til Q_a i self.setBathymetry(terrain)
        # print(self.Q_a)
        self.Q_o = np.zeros((self.Ny, self.Nx, 6),order='C', dtype=np.double)  # Density current outflow

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

        self.Q_cj, self.Q_th = tra.T_1(self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_th, self.Q_v, self.dt, self.g)  # Water entrainment.
        self.sanityCheck()
        # self.printSubstates_to_screen('T_1')
        # Erosion and deposition TODO: Fix cause of instability
        self.Q_a, self.Q_d, self.Q_cj, self.Q_cbj = tra.T_2(self.rho_j, self.rho_a, self.D_sj, self.nu, self.g,
                                                            self.c_D, self.Q_v, self.v_sj, self.Q_cj, self.Q_cbj,
                                                            self.Q_th, self.Q_d, self.dt, self.porosity, self.Q_a,
                                                            self.Erosionrate, self.Depositionrate)
        self.sanityCheck()
        # self.printSubstates_to_screen('T_2')
        # Turbidity c. outflows
        self.Q_o = tra.I_1(self.Q_th, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_v, self.Q_a, self.Ny, self.Nx,
                           self.dx, self.p_f, self.NEIGHBOR, self.p_adh, self.dt, self.Q_o, self.indexMat, self.g)
        self.sanityCheck()
        # self.printSubstates_to_screen('I_1')
        # Update thickness and concentration
        self.Q_th, self.Q_cj = tra.I_2(self.Ny, self.Nx, self.Nj, self.Q_o, self.NEIGHBOR, self.Q_th, self.Q_cj)
        self.sanityCheck()
        # self.printSubstates_to_screen('I_2')
        self.Q_v = tra.I_3(self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Ny, self.Nx, self.Q_a, self.Q_th,
                           self.NEIGHBOR, self.Q_o, self.Q_v, self.f, self.a)  # Update of turbidity flow velocity
        self.sanityCheck()
        # self.printSubstates_to_screen('I_3')
        self.Q_a, self.Q_d, self.Q_cbj = tra.I_4(self.Q_d, self.Ny, self.Nx, self.dx, self.reposeAngle, self.Q_cbj,
                                                 self.Q_a, self.seaBedDiff)  # Toppling rule
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




    # def I_4(self):  # Toppling rule
    #     interiorH = self.Q_d[1:self.Ny - 1, 1:self.Nx - 1]
    #
    #     # angle = np.zeros((self.Ny - 2, self.Ny - 2, 6))
    #     indices = np.zeros((self.Ny - 2, self.Nx - 2, 6))
    #     NoOfTrans = np.zeros((self.Ny - 2, self.Nx - 2))
    #     frac = np.zeros((self.Ny - 2, self.Nx - 2, 6))
    #     deltaS = np.zeros((self.Ny - 2, self.Nx - 2, 6))
    #     deltaSSum = np.zeros((self.Ny - 2, self.Nx - 2))
    #
    #     self.calc_Hdiff()
    #     diff = self.diff
    #
    #     # Find angles
    #     dx = self.dx
    #     angle = ne.evaluate('arctan2(diff,dx)')
    #
    #     # (Checks if cell (i,j) has angle > repose angle and that it has mass > 0. For all directions.)
    #     # Find cells (i,j) for which to transfer mass in the direction given
    #     for i in np.arange(6):
    #         indices[:, :, i] = np.logical_and(angle[:, :, i] > self.reposeAngle, (
    #                 interiorH > 0))  # Gives indices (i,j) where the current angle > repose angle and where height is > 0
    #
    #     # Count up the number of cells (i,j) will be transfering mass to. If none, set (i,j) to infinity so that division works.
    #     #         NoOfTrans = np.sum(indices,axis=2)  # Gir tregere resultat?
    #     for i in np.arange(6):
    #         NoOfTrans += indices[:, :, i]
    #     NoOfTrans[NoOfTrans == 0] = np.inf
    #
    #     # Calculate fractions of mass to be transfered
    #     for i in np.arange(6):
    #         frac[(indices[:, :, i] > 0), i] = (
    #                 0.5 * (diff[(indices[:, :, i] > 0), i] - self.dx * np.tan(self.reposeAngle)) / (
    #             interiorH[(indices[:, :, i] > 0)]))
    #     frac[frac > 0.5] = 0.5
    #     #         print("frac.shape=",frac.shape)
    #
    #     for i in np.arange(6):
    #         deltaS[(indices[:, :, i] > 0), i] = interiorH[(indices[:, :, i] > 0)] * frac[(indices[:, :, i] > 0), i] / \
    #                                             NoOfTrans[(indices[:, :,
    #                                                        i] > 0)]  # Mass to be transfered from index [i,j] to index [i-1,j]
    #
    #     # Lag en endringsmatrise deltaSSum som kan legges til self.Q_d
    #     # Trekk fra massen som skal sendes ut fra celler
    #     deltaSSum = -np.sum(deltaS, axis=2)
    #
    #     # Legg til massen som skal tas imot. BRUK self.NEIGHBOR
    #     deltaSSum += np.roll(np.roll(deltaS[:, :, 0], -1, 0), 0, 1)
    #     deltaSSum += np.roll(np.roll(deltaS[:, :, 1], -1, 0), 1, 1)
    #     deltaSSum += np.roll(np.roll(deltaS[:, :, 2], 0, 0), 1, 1)
    #     deltaSSum += np.roll(np.roll(deltaS[:, :, 3], 1, 0), 0, 1)
    #     deltaSSum += np.roll(np.roll(deltaS[:, :, 4], 1, 0), -1, 1)
    #     deltaSSum += np.roll(np.roll(deltaS[:, :, 5], 0, 0), -1, 1)
    #
    #     oldQ_d = self.Q_d.copy()
    #     self.Q_d[1:-1, 1:-1] += deltaSSum
    #     self.Q_a[1:-1, 1:-1] += deltaSSum
    #     # Legg inn endring i volum fraksjon Q_cbj
    #     prefactor = 1 / self.Q_d[1:-1, 1:-1, np.newaxis]
    #     prefactor[np.isinf(prefactor)] = 0
    #     nq_cbj = np.nan_to_num(prefactor *
    #                            (oldQ_d[1:-1, 1:-1, np.newaxis] * self.Q_cbj[1:-1, 1:-1, :] + deltaSSum[:, :,
    #                                                                                          None]))  # TODO usikker på om denne blir rett!
    #     nq_cbj = np.round(nq_cbj,15)
    #     self.Q_cbj[1:-1, 1:-1] = nq_cbj
    #     self.Q_cbj[self.Q_cbj < 1e-15] = 0
    #     if (self.Q_d < -1e-7).sum() > 0:
    #         print('height', self.Q_d[1, 6])
    #         raise RuntimeError('Negative sediment thickness!')

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
