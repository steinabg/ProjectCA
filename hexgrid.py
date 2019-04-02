import matplotlib.pyplot as plt

plt.style.use('bmh')
import numpy as np
import mathfunk as ma
import transition_functions_cy as tra
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
        self.rho_j = np.array([2650], dtype=np.dtype("i"), order='C')  # List of current sediment densities
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
        self.Q_v_is_zero_two_timesteps = np.zeros((self.Ny, self.Nx), dtype=np.intc, order='C') # update in I3 read in T2

        ################### Set Initial conditions #####################
        if ICstates is not None: self.set_substate_ICs(ICstates)
        self.CellArea = ma.calc_hexagon_area(dx)
        if global_grid == True:
            self.setBathymetry(terrain)
        self.diff = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        self.seaBedDiff = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        self.calc_bathymetryDiff()





        # FOR DEBUGGING
        self.i = 0
        self.my_rank = 0
        self.unphysical_substate = {'Q_th': 0, 'Q_v': 0, 'Q_cj':0,'Q_cbj':0,'Q_d':0,'Q_o':0}
        self.Erosionrate_sample = []
        self.Depositionrate_sample = []

        self.find_channel_bot() # Find the indices of the channel minima in axial direction

        ################################################################
        ##########################  Methods ############################
        ################################################################

    def find_channel_bot(self):
        self.bot_indices = []
        for i in range(self.Ny):
            self.bot_indices.append((i,np.min(np.where(np.min(self.Q_a[i,:])==self.Q_a[i,:]))))


    def set_substate_ICs(self, ICstates):
        self.Q_th = ICstates[0].copy()
        self.Q_v = ICstates[1].copy()
        self.Q_cj = ICstates[2].copy()
        self.Q_cbj = ICstates[3].copy()
        self.Q_d = ICstates[4].copy()
        self.Q_o = ICstates[5].copy()
        self.Q_a = self.Q_d.copy()



    def time_step(self, global_grid = True):
        if global_grid is True:
            self.dt = self.calc_dt()  # Works as long as all ICs are given

        self.Q_cj, self.Q_th = tra.T_1(self.Ny, self.Nx, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_th, self.Q_v, self.dt, self.g)  # Water entrainment.
        self.sanityCheck()
        # self.printSubstates_to_screen('T_1')
        # Erosion and deposition
        self.Q_a, self.Q_d, self.Q_cj, self.Q_cbj, self.Q_th = tra.T_2(self.Ny,self.Nx,self.Nj,self.rho_j, self.rho_a,
                                                                       self.D_sj, self.nu, self.g,
                                                                self.c_D, self.Q_v, self.v_sj, self.Q_cj, self.Q_cbj,
                                                                self.Q_th, self.Q_d, self.dt, self.porosity, self.Q_a,
                                                                self.Q_v_is_zero_two_timesteps)
        self.sanityCheck()
        # self.printSubstates_to_screen('T_2')
        # Turbidity c. outflows
        self.Q_o = tra.I_1(self.Q_th, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_v, self.Q_a,
                           self.Ny, self.Nx, self.dx, self.p_f, self.p_adh, self.dt, self.g)
        self.sanityCheck()
        # self.printSubstates_to_screen('I_1')
        # Update thickness and concentration
        self.Q_th, self.Q_cj = tra.I_2(self.Ny, self.Nx, self.Nj, self.Q_o, self.Q_th, self.Q_cj)
        self.sanityCheck()
        # self.printSubstates_to_screen('I_2')
        self.Q_v, self.Q_v_is_zero_two_timesteps = tra.I_3(self.g, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Ny, self.Nx, self.Q_a,
                           self.Q_th, self.Q_o, self.f, self.a, self.Q_v)  # Update of turbidity flow velocity
        self.sanityCheck()
        # self.printSubstates_to_screen('I_3')
        self.Q_a, self.Q_d, self.Q_cbj = tra.I_4(self.Q_d, self.Ny, self.Nx, self.Nj, self.dx, self.reposeAngle, self.Q_cbj,
                                                 self.Q_a, self.seaBedDiff)  # Toppling rule
        self.sanityCheck()
        # self.printSubstates_to_screen('I_4')

    def time_step_compare_cy_py(self, global_grid=True, tol=1e-12):
        # Load unloaded version of transition function file
        import transition_functions as tra
        import transition_functions_cy as tra2

        if global_grid is True:
            self.dt = self.calc_dt()
        # Water entrainment
        t_Q_cj, t_Q_th = tra2.T_1(self.Ny, self.Nx, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_th,
                                       self.Q_v, self.dt, self.g)

        self.Q_cj, self.Q_th = tra.T_1(self.Ny, self.Nx, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_th,
                                       self.Q_v, self.dt, self.g)
        ma.compare_ndarray(t_Q_cj,self.Q_cj, tol)
        ma.compare_ndarray(t_Q_th,self.Q_th, tol)

        self.sanityCheck()

        # Erosion and deposition
        t_Q_a, t_Q_d, t_Q_cj, t_Q_cbj, t_Q_th = tra2.T_2(self.Ny, self.Nx, self.Nj, self.rho_j,
                                                                       self.rho_a,
                                                                       self.D_sj, self.nu, self.g,
                                                                       self.c_D, self.Q_v, self.v_sj, self.Q_cj,
                                                                       self.Q_cbj,
                                                                       self.Q_th, self.Q_d, self.dt, self.porosity,
                                                                       self.Q_a,
                                                                       self.Q_v_is_zero_two_timesteps)

        self.Q_a, self.Q_d, self.Q_cj, self.Q_cbj, self.Q_th = tra.T_2(self.Ny, self.Nx, self.Nj, self.rho_j,
                                                                       self.rho_a,
                                                                       self.D_sj, self.nu, self.g,
                                                                       self.c_D, self.Q_v, self.v_sj, self.Q_cj,
                                                                       self.Q_cbj,
                                                                       self.Q_th, self.Q_d, self.dt, self.porosity,
                                                                       self.Q_a,
                                                                       self.Q_v_is_zero_two_timesteps)
        ma.compare_ndarray(t_Q_a[1:-1,1:-1], self.Q_a[1:-1,1:-1], tol)
        ma.compare_ndarray(t_Q_d[1:-1,1:-1], self.Q_d[1:-1,1:-1], tol)
        ma.compare_ndarray(t_Q_cj, self.Q_cj, tol)
        ma.compare_ndarray(t_Q_cbj, self.Q_cbj, tol)
        ma.compare_ndarray(t_Q_th, self.Q_th, tol)
        self.sanityCheck()

        # Turbidity c. outflows
        t_Q_o = tra2.I_1(self.Q_th, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_v, self.Q_a,
                           self.Ny, self.Nx, self.dx, self.p_f, self.p_adh, self.dt, self.g)

        self.Q_o = tra.I_1(self.Q_th, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_v, self.Q_a,
                           self.Ny, self.Nx, self.dx, self.p_f, self.p_adh, self.dt, self.g)
        ma.compare_ndarray(t_Q_o, self.Q_o, tol)
        self.sanityCheck()

        t_Q_th, t_Q_cj = tra2.I_2(self.Ny, self.Nx, self.Nj, self.Q_o, self.Q_th, self.Q_cj)

        self.Q_th, self.Q_cj = tra.I_2(self.Ny, self.Nx, self.Nj, self.Q_o, self.Q_th, self.Q_cj)
        ma.compare_ndarray(t_Q_th, self.Q_th, tol)
        ma.compare_ndarray(t_Q_cj, self.Q_cj, tol)
        self.sanityCheck()

        # TC speed
        t_Q_v, t_Q_v_is_zero_two_timesteps = tra2.I_3(self.g, self.Nj, self.Q_cj, self.rho_j, self.rho_a,
                                                           self.Ny, self.Nx, self.Q_a,
                                                           self.Q_th, self.Q_o, self.f, self.a,
                                                           self.Q_v)  # Update of turbidity flow velocity

        self.Q_v, self.Q_v_is_zero_two_timesteps = tra.I_3(self.g, self.Nj, self.Q_cj, self.rho_j, self.rho_a,
                                                           self.Ny, self.Nx, self.Q_a,
                                                           self.Q_th, self.Q_o, self.f, self.a,
                                                           self.Q_v)  # Update of turbidity flow velocity
        ma.compare_ndarray(t_Q_v, self.Q_v, tol)
        ma.compare_ndarray(t_Q_v_is_zero_two_timesteps, self.Q_v_is_zero_two_timesteps, tol)
        self.sanityCheck()

        # Bed toppling rule
        t_Q_a, t_Q_d, t_Q_cbj = tra2.I_4(self.Q_d, self.Ny, self.Nx, self.Nj, self.dx, self.reposeAngle,
                                                 self.Q_cbj,
                                                 self.Q_a, self.seaBedDiff)

        self.Q_a, self.Q_d, self.Q_cbj = tra.I_4(self.Q_d, self.Ny, self.Nx, self.Nj, self.dx, self.reposeAngle,
                                                 self.Q_cbj,
                                                 self.Q_a, self.seaBedDiff)
        ma.compare_ndarray(t_Q_a, self.Q_a, tol)
        ma.compare_ndarray(t_Q_d, self.Q_d, tol)
        ma.compare_ndarray(t_Q_cbj, self.Q_cbj, tol)
        self.sanityCheck()

    def T_1(self):
        self.Q_cj, self.Q_th = tra.T_1(self.Ny, self.Nx, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_th,
                                       self.Q_v, self.dt, self.g)  # Water entrainment.
        self.sanityCheck()

    def T_2(self):
        self.Q_a, self.Q_d, self.Q_cj, self.Q_cbj, self.Q_th = tra.T_2(self.Ny, self.Nx, self.Nj, self.rho_j, self.rho_a,
                                                            self.D_sj, self.nu, self.g,
                                                            self.c_D, self.Q_v, self.v_sj, self.Q_cj, self.Q_cbj,
                                                            self.Q_th, self.Q_d, self.dt, self.porosity, self.Q_a,
                                                            self.Q_v_is_zero_two_timesteps)
        self.sanityCheck()
    def I_1(self):
        self.Q_o = tra.I_1(self.Q_th, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_v, self.Q_a,
                           self.Ny, self.Nx, self.dx, self.p_f, self.p_adh, self.dt, self.g)
        self.sanityCheck()
    def I_2(self):
        self.Q_th, self.Q_cj = tra.I_2(self.Ny, self.Nx, self.Nj, self.Q_o, self.Q_th, self.Q_cj)
        self.sanityCheck()
    def I_3(self):
        self.Q_v, self.Q_v_is_zero_two_timesteps = tra.I_3(self.g, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Ny, self.Nx, self.Q_a,
                           self.Q_th, self.Q_o, self.f, self.a, self.Q_v)  # Update of turbidity flow velocity
        self.sanityCheck()
    def I_4(self):
        self.Q_a, self.Q_d, self.Q_cbj = tra.I_4(self.Q_d, self.Ny, self.Nx, self.Nj, self.dx, self.reposeAngle, self.Q_cbj,
                                                 self.Q_a, self.seaBedDiff)  # Toppling rule
        self.sanityCheck()


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

    def setBathymetry(self, terrain):
        if terrain is not None:
            x = np.linspace(0, 100, self.Nx)
            y = np.linspace(0, 100, self.Ny)
            X = np.array(np.meshgrid(x, y))
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
                temp, junk = ma.generate_rupert_inlet_bathymetry(self.reposeAngle, self.dx, self.Ny,self.Nx)
                temp = ma.gen_sloped_plane(self.Ny,self.Nx,self.dx,np.deg2rad(-5),mat=temp.transpose())
                self.Q_a += temp
            elif terrain == 'sloped_plane':
                self.Q_a += ma.gen_sloped_plane(self.Ny, self.Nx, self.dx, np.arctan(0.08))
            else:
                terrain_path = './Bathymetry/' + terrain + '.npy'
                try:
                    temp = np.load(terrain_path)
                except FileNotFoundError:
                    raise FileNotFoundError('Could not find bathymetry with name {0}'.format(terrain))
                assert temp.shape == self.Q_a.shape
                self.Q_a += temp

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
