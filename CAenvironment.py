import numpy as np
import mathfunk as ma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('bmh')
# import sys
import os.path
import os
import transition_functions_cy as tra
# from mpldatacursor import datacursor
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

# sys.path.append('..')
np.set_printoptions(suppress=True, precision=3)


class CAenvironment:

    def __init__(self, parameters, global_grid=True, mpi=False):
        #     plt.ioff()
        self.mpi = mpi
        self.global_grid = global_grid  # If False this environment describes a local CA (part of a grid)
        self.parameters = parameters
        self.Nx = parameters['nx']
        self.Ny = parameters['ny']
        self.dx = parameters['dx']
        self.dt = 0
        self.reposeAngle = parameters['theta_r']
        self.g = parameters['g']  # Gravitational acceleration
        self.f = parameters['f']  # Darcy-Weisbach coeff
        self.a = parameters['a']  # Empirical coefficient (used in I_3)
        self.rho_a = parameters['rho_a']  # ambient density
        self.rho_j = parameters['rho_j']  # List of current sediment densities
        self.D_sj = parameters['d_sj']  # List of sediment-particle diameters
        self.Nj = parameters['nj']  # Number of sediment types
        self.c_D = parameters['c_d']  # Bed drag coefficient (table 3)
        self.nu = parameters['nu']  # Kinematic viscosity of water at 5 degrees celcius
        self.porosity = parameters['porosity']
        self.p_f = parameters['p_f']  # Height threshold friction angle
        self.p_adh = parameters['p_adh']
        if mpi is False: self.save_dt = []

        s = parameters['sphere_settling_velocity']
        if type(s) != str:
            self.v_sj = s
        elif s.lower() == 'salles' or s.lower() == 'vanrijn':
            self.v_sj = ma.calc_settling_speed(self.D_sj, self.rho_a, self.rho_j,
                                               self.g, self.nu)
        elif s.lower() == 'soulsby':
            self.v_sj = ma.calc_settling_speed(self.D_sj, self.rho_a, self.rho_j,
                                               self.g, self.nu, method='soulsby')


        ################   Grid for plotting    ###################
        self.x0 = self.y0 = 0  # Offset
        self.X = np.zeros((self.Ny, self.Nx, 2))  # X[:,:,0] = X coords, X[:,:,1] = Y coords
        for jj in range(self.Ny):
            self.X[jj, :, 0] = self.x0 + jj * self.dx / 2 + np.arange(self.Nx) * self.dx
            self.X[jj, :, 1] = self.y0 + np.ones(self.Nx) * self.dx * np.sqrt(3) / 2 * jj

        ################# Cell substate buffers ####################
        self.Q_a = np.zeros((self.Ny, self.Nx), order='C', dtype=np.double)  # Cell altitude (bathymetry at t = 0)
        self.Q_th = np.zeros((self.Ny, self.Nx), order='C', dtype=np.double)  # Turbidity current thickness
        self.Q_v = np.zeros((self.Ny, self.Nx), order='C', dtype=np.double)  # Turbidity current speed (scalar)
        self.Q_cj = np.zeros((self.Ny, self.Nx, self.Nj), order='C',
                             dtype=np.double)  # jth current sediment volume concentration
        self.Q_cbj = np.zeros((self.Ny, self.Nx, self.Nj), order='C',
                              dtype=np.double)  # jth bed sediment volume fraction
        if global_grid:
            self.Q_d = np.ones((self.Ny, self.Nx), order='C', dtype=np.double) * np.inf  # Thickness of soft sediment
            self.Q_d[1:-1, 1:-1] = 0
        else:
            self.Q_d = np.zeros((self.Ny, self.Nx), order='C', dtype=np.double)

        self.Q_o = np.zeros((self.Ny, self.Nx, 6), order='C', dtype=np.double)  # Density current outflow
        self.Q_v_is_zero_two_timesteps = np.zeros((self.Ny, self.Nx), dtype=np.intc,
                                                  order='C')  # update in I3 read in T2

        # SET BATHYMETRY
        self.terrain = parameters['terrain']
        if global_grid:
            try:
                self.Q_a += self.get_global_bathy(parameters['terrain'], slope=parameters['slope'])
            except KeyError:
                self.Q_a += self.get_global_bathy(parameters['terrain'])
        else: # Bathymetry gitt i parameters['bathymetry']
            try:
                self.Q_a[1:-1,1:-1] += parameters['bathymetry']
            except KeyError:
                raise KeyError("No bathymetry specified in parameters['bathymetry']")


        # SET INITIAL CONDITIONS
        # Interior sand
        for particle_type in range(self.Nj):
            parameter_string2 = 'q_cbj[interior,' + str(particle_type) + ']'
            self.Q_cbj[1:-1, 1:-1, particle_type] = parameters[parameter_string2]  # 1
        self.Q_d[1:-1, 1:-1] = parameters['q_d[interior]']  # 1.0

        # Source area (overwrite interior)
        # First try to convert from coordinates to indices
        try:
            N = parameters['n']
            E = parameters['e']
            y = parameters['y'] = ma.find_index_nearest(self.X[:, 0, 1], N)
            parameters['x'] = ma.find_index_nearest(self.X[y, :, 0], E)
        except KeyError:  # No n or e specified
            pass
        try:
            if (parameters['x'] is not None) and (parameters['y'] is not None):
                # By this time in mpi local grid, x y is None for relevant grids
                self.y, self.x = np.meshgrid(parameters['y'], parameters['x'])
                self.Q_th[self.y, self.x] = parameters['q_th[y,x]']  # 1.5
                self.Q_v[self.y, self.x] = parameters['q_v[y,x]']  # 0.2
                self.Q_d[self.y, self.x] = parameters['q_d[y,x]']  # 1
                for particle_type in range(self.Nj):
                    parameter_string1 = 'q_cj[y,x,' + str(particle_type) + ']'
                    parameter_string2 = 'q_cbj[y,x,' + str(particle_type) + ']'
                    self.Q_cj[self.y, self.x, particle_type] = parameters[parameter_string1]  # 0.003
                    self.Q_cbj[self.y, self.x, particle_type] = parameters[parameter_string2]  # 1
        except KeyError:
            print("Warning! No source area specified. Either specify a coordinate using (N,E) or (x,y).")
        l1 = []
        l2 = []
        for particle_type in range(self.Nj):
            parameter_string1 = 'q_cj[y,x,' + str(particle_type) + ']'
            parameter_string2 = 'q_cbj[y,x,' + str(particle_type) + ']'
            l1.append(parameters[parameter_string1])
            l2.append(parameters[parameter_string2])
        self.sourcevalues = {'Q_th': parameters['q_th[y,x]'],
                             'Q_v': parameters['q_v[y,x]'],
                             'Q_d': parameters['q_d[y,x]'],
                             'Q_cj': l1, 'Q_cbj': l2}


        self.bathymetry = self.Q_a.copy()
        self.Q_a += self.Q_d.copy()  # Add sand layer to bed height
        self.seaBedDiff = self.calc_bathymetryDiff()

        # Run toppling rule until convergence, before starting simulation
        if global_grid is True and mpi is False:
            try:
                if parameters['converged_toppling'][0]:
                    tol = parameters['converged_toppling'][1]
                    print("Running toppling rule until converged with tol = {0}. This may take a while!".format(tol))
                    jj = 0
                    while True:
                        Q_d_old = self.Q_d.copy()
                        self.I_4()
                        jj += 1
                        if ma.two_norm(Q_d_old[1:-1, 1:-1], self.Q_d[1:-1, 1:-1]) <= tol:
                            break
                    print("Convergence achieved after {0} iterations. Continuing initialization.".format(jj))
            except KeyError:
                print("Toppling rule keyword not found! Skipping toppling before simulation.")

        # FOR DEBUGGING
        if global_grid == 0:
            self.my_rank = parameters['rank']
        else:
            self.my_rank = -1
        self.unphysical_substate = {'Q_th': 0, 'Q_v': 0, 'Q_cj': 0, 'Q_cbj': 0, 'Q_d': 0, 'Q_o': 0}
        self.Erosionrate_sample = []
        self.Depositionrate_sample = []

        self.find_channel_bot()  # Find the indices of the channel minima in axial direction

        self.Q_a_south = self.Q_a[-1, :].copy()  # For absorbing boundary

        self.time = []
        self.mass = []
        self.massBed = []
        self.density = []
        self.beddensity = []
        self.head_velocity = []

        self.save_path = './Data/'

        # For plotting in center of channel
        # self.ch_bot_thickness =

    def CAtimeStep_with_source(self):
        # Normal time step
        # self.time_step(self.global_grid)
        self.dt = self.calc_dt()
        self.add_source_constant_flow()
        self.T_1()
        self.add_source_constant_flow()
        self.T_2()
        self.add_source_constant_flow()
        self.I_1()
        self.add_source_constant_flow()
        self.I_2()
        self.add_source_constant_flow()
        self.I_3()
        self.add_source_constant_flow()
        self.I_4()
        self.add_source_constant_flow()

    ###### METHODS FROM HEXGRID #######

    def find_channel_bot(self):
        self.bot_indices = []
        for i in range(self.Ny):
            self.bot_indices.append((i, np.min(np.where(np.min(self.Q_a[i, :]) == self.Q_a[i, :]))))

    def set_substate_ICs(self, ICstates):
        self.Q_th = ICstates[0].copy()
        self.Q_v = ICstates[1].copy()
        self.Q_cj = ICstates[2].copy()
        self.Q_cbj = ICstates[3].copy()
        self.Q_d = ICstates[4].copy()
        self.Q_o = ICstates[5].copy()
        self.Q_a = self.Q_d.copy()

    def T_1(self):
        self.Q_cj, self.Q_th = tra.T_1(self.Ny, self.Nx, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_th,
                                       self.Q_v, self.dt, self.g)  # Water entrainment.
        self.sanityCheck()

    def T_2(self):
        self.Q_a, self.Q_d, self.Q_cj, self.Q_cbj, self.Q_th = tra.T_2(self.Ny, self.Nx, self.Nj, self.rho_j,
                                                                       self.rho_a,
                                                                       self.D_sj, self.nu, self.g,
                                                                       self.c_D, self.Q_v, self.v_sj, self.Q_cj,
                                                                       self.Q_cbj,
                                                                       self.Q_th, self.Q_d, self.dt, self.porosity,
                                                                       self.Q_a,
                                                                       self.Q_v_is_zero_two_timesteps)
        self.sanityCheck()

    def I_1(self):
        self.Q_o, self.Q_o_no_time = tra.I_1(self.Q_th, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Q_v, self.Q_a,
                           self.Ny, self.Nx, self.dx, self.p_f, self.p_adh, self.dt, self.g)
        self.sanityCheck()

    def I_2(self):
        self.Q_th, self.Q_cj = tra.I_2(self.Ny, self.Nx, self.Nj, self.Q_o, self.Q_th, self.Q_cj)
        self.sanityCheck()

    def I_3(self):
        self.Q_v, self.Q_v_is_zero_two_timesteps = tra.I_3(self.g, self.Nj, self.Q_cj, self.rho_j, self.rho_a, self.Ny,
                                                           self.Nx, self.Q_a,
                                                           self.Q_th, self.Q_o, self.f, self.a,
                                                           self.Q_v)  # Update of turbidity flow velocity
        self.Q_v = self.Q_v / np.sqrt(self.dt)
        # if (self.parameters['x'] is not None) and (self.parameters['y'] is not None):
        #     self.Q_v[self.y, self.x] = self.sourcevalues['Q_v']  # TODO, test
        self.sanityCheck()

    def I_4(self):
        self.Q_a, self.Q_d, self.Q_cbj, self.t, self.b, self.l, self.r = tra.I_4(self.Q_d, self.Ny, self.Nx, self.Nj,
                                                                                 self.dx, self.reposeAngle, self.Q_cbj,
                                                                                 self.Q_a,
                                                                                 self.seaBedDiff)  # Toppling rule
        self.sanityCheck()

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
        ma.compare_ndarray(t_Q_cj, self.Q_cj, tol)
        ma.compare_ndarray(t_Q_th, self.Q_th, tol)

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
        ma.compare_ndarray(t_Q_a[1:-1, 1:-1], self.Q_a[1:-1, 1:-1], tol)
        ma.compare_ndarray(t_Q_d[1:-1, 1:-1], self.Q_d[1:-1, 1:-1], tol)
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
        t_Q_a, t_Q_d, t_Q_cbj, _, _, _, _ = tra2.I_4(self.Q_d, self.Ny, self.Nx, self.Nj, self.dx, self.reposeAngle,
                                                     self.Q_cbj,
                                                     self.Q_a, self.seaBedDiff)

        self.Q_a, self.Q_d, self.Q_cbj = tra.I_4(self.Q_d, self.Ny, self.Nx, self.Nj, self.dx, self.reposeAngle,
                                                 self.Q_cbj,
                                                 self.Q_a, self.seaBedDiff)
        ma.compare_ndarray(t_Q_a, self.Q_a, tol)
        ma.compare_ndarray(t_Q_d, self.Q_d, tol)
        ma.compare_ndarray(t_Q_cbj, self.Q_cbj, tol)
        self.sanityCheck()

    def sanityCheck(self):
        # Round off numerical error
        # self.Q_cbj[self.Q_cbj > 0 & self.Q_cbj < (1-1e-16)] = 1

        # DEBUGGING: Check for unphysical values
        if np.any(self.Q_th < 0):
            self.unphysical_substate['Q_th'] = 1
            raise Exception('unphysical_substate[Q_th]')
        if np.any(self.Q_v < 0):
            self.unphysical_substate['Q_v'] = 1
            raise Exception('unphysical_substate[Q_v]')
        if (np.any(self.Q_cj > 1)) | (np.any(self.Q_cj < 0)):
            self.unphysical_substate['Q_cj'] = 1
            raise Exception('unphysical_substate[Q_cj]')
        if (np.any(self.Q_cbj > 1)) | (np.any(self.Q_cbj < 0)):
            self.unphysical_substate['Q_cbj'] = 1
            ii, jj, kk = np.where(np.logical_or(self.Q_cbj > 1,self.Q_cbj < 0))
            s = "unphysical_substate[Q_cbj]\n"
            if self.mpi: s = s + ''.join("rank = ".format(self.my_rank))
            s = self.create_substate_string(s, [ii,jj,kk])
            raise Exception(s)
        if np.any(self.Q_d < 0):
            self.unphysical_substate['Q_d'] = 1
            raise Exception('unphysical_substate[Q_d]')
        if np.any(self.Q_o < 0):
            self.unphysical_substate['Q_o'] = 1
            raise Exception('unphysical_substate[Q_o]')

        t1 = np.sum(self.Q_cbj, 2)
        if (np.any(t1) != 1) and np.any(t1) != 0:
            print(np.where(np.sum(self.Q_cbj, 2) != 1))
            raise Exception("should always be 1 with nj=1!")

    def create_substate_string(self, string, index):
        """ This function is used to print errors containing\
            information about unphysical substates."""
        ii = index[0]
        jj = index[1]
        kk = index[2]
        string = string + ''.join("q_cj[{0},{1},{2}] = {3}\n".format(
                ii[x], jj[x], kk[x], self.Q_cj[ii[x], jj[x],kk[x]]) for x in range(len(ii)))
        string = string + ''.join("q_cbj[{0},{1},{2}] = {3}\n".format(
            ii[x], jj[x], kk[x], self.Q_cbj[ii[x], jj[x], kk[x]]) for x in range(len(ii)))
        string = string + ''.join("q_th[{0},{1}] = {2}\n".format(
            ii[x], jj[x], self.Q_th[ii[x], jj[x]]) for x in range(len(ii)))
        string = string + ''.join("q_d[{0},{1}] = {2}\n".format(
            ii[x], jj[x], self.Q_d[ii[x], jj[x]]) for x in range(len(ii)))
        string = string + ''.join("q_v[{0},{1}] = {2}\n".format(
            ii[x], jj[x], self.Q_v[ii[x], jj[x]]) for x in range(len(ii)))


        return string

    def get_global_bathy(self, terrain, slope=0.08):
        """ Returns global bathymetry """
        if terrain is not None:
            x = np.linspace(0, 100, self.Nx)
            y = np.linspace(0, 100, self.Ny)
            X = np.array(np.meshgrid(x, y))
            if type(terrain) == str:
                if terrain == 'river':
                    temp = -2 * X[1, :] + 5 * np.abs(X[0, :] - 50 + 10 * np.sin(X[1, :] / 10))
                    #                 temp = 2*self.X[:,:,1] + 5*np.abs(self.X[:,:,0] + 10*np.sin(self.X[:,:,1]/10))
                    return temp  # BRUK MED RIVER
                elif terrain == 'river_shallow':
                    temp = -1 * X[1, :] + 1 * np.abs(X[0, :] - 50 + 5 * np.sin(X[1, :] / 10))
                    #                 temp = 2*self.X[:,:,1] + 5*np.abs(self.X[:,:,0] + 10*np.sin(self.X[:,:,1]/10))
                    return temp  # BRUK MED RIVER
                elif terrain == 'pit':
                    temp = np.sqrt((X[0, :] - 50) * (X[0, :] - 50) + (X[1, :] - 50) * (X[1, :] - 50))
                    return 10 * temp
                elif terrain == 'rupert':
                    print("Using slope={0}".format(slope))
                    temp, junk = ma.generate_rupert_inlet_bathymetry(self.reposeAngle, self.dx, self.Ny, self.Nx)
                    temp = ma.gen_sloped_plane(self.Ny, self.Nx, self.dx, -slope, mat=temp.transpose())
                    return temp
                elif terrain == 'sloped_plane':
                    print("Using slope={0}".format(slope))
                    return ma.gen_sloped_plane(self.Ny, self.Nx, self.dx, slope)
                elif terrain == 'ranfjorden':
                    #### Redefine X using x0 and y0 ######
                    Ny = self.Ny
                    Nx = self.Nx
                    self.x0 = x0 = 452000
                    self.y0 = y0 = 7350000
                    self.dx = dx = 50
                    self.X = np.zeros((Ny, Nx, 2))  # X[:,:,0] = X coords, X[:,:,1] = Y coords
                    for j in range(Ny):
                        self.X[j, :, 0] = x0 + j * dx / 2 + np.arange(Nx) * dx
                        self.X[j, :, 1] = y0 + np.ones(Nx) * dx * np.sqrt(3) / 2 * j

                    import xarray as xr
                    from scipy.interpolate import RectBivariateSpline
                    ncfile = './Bathymetry/ranfjorden_depth.nc'
                    with xr.open_dataset(ncfile) as d:
                        # Create spline interpolation object
                        b = RectBivariateSpline(d.yc, d.xc, d.depth)
                        # evaluate spline at positions of hexgrid cells
                        temp = -1*b(self.X[:, :, 1], self.X[:, :, 0], grid=False)
                        return temp # Subtract beacuse temp is 'depth'
                elif terrain == 'random':
                    return np.random.rand(self.Ny, self.Nx) * 10
                else:
                    terrain_path = './Bathymetry/' + terrain + '.npy'
                    try:
                        temp = np.load(terrain_path)
                    except FileNotFoundError:
                        raise FileNotFoundError('Could not find bathymetry with name {0}'.format(terrain))
                    assert temp.shape == self.Q_a.shape
                    return temp
            elif type(terrain) == np.ndarray:
                assert self.Q_a.shape == terrain.shape
                return terrain
        else:
            return np.zeros((self.Ny, self.Nx))

    def calc_bathymetryDiff(self):
        seaBedDiff = np.zeros((self.Ny - 2, self.Nx - 2, 6))
        with np.errstate(invalid='ignore'):
            temp = self.Q_a - self.Q_d
        seaBedDiff[:, :, 0] = temp[1:-1, 1:-1] - temp[0:self.Ny - 2, 1:self.Nx - 1]
        seaBedDiff[:, :, 1] = temp[1:-1, 1:-1] - temp[0:self.Ny - 2, 2:self.Nx]
        seaBedDiff[:, :, 2] = temp[1:-1, 1:-1] - temp[1:self.Ny - 1, 2:self.Nx]
        seaBedDiff[:, :, 3] = temp[1:-1, 1:-1] - temp[2:self.Ny, 1:self.Nx - 1]
        seaBedDiff[:, :, 4] = temp[1:-1, 1:-1] - temp[2:self.Ny, 0:self.Nx - 2]
        seaBedDiff[:, :, 5] = temp[1:-1, 1:-1] - temp[1:self.Ny - 1, 0:self.Nx - 2]
        seaBedDiff[np.isnan(seaBedDiff)] = 0
        return seaBedDiff

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
            # dt = np.min([np.amin(temp[np.isfinite(temp) & (~np.isnan(temp)) & (temp > 0)]), 0.02])  # Better stability
            dt = np.amin(temp[np.isfinite(temp) & (~np.isnan(temp)) & (temp > 0)])
        except:
            if global_grid is True:
                dt = 0.01 # TODO, burde kanskje la sim slutte hvis dette skjer
            else:
                dt = 9999999  # Set a large number so we can use MPI.Reduce MIN.
        if self.mpi is False: self.save_dt.append(dt)
        return dt

    ###### END OF METHODS FROM HEXGRID ##########

    def add_source_constant_flow(self):
        """Note: this may turn into a drain if Q_th[nb] > Q_th0 !"""
        if (self.parameters['x'] is not None) and (self.parameters['y'] is not None):
            self.Q_v[self.y, self.x] = self.parameters['q_v[y,x]']
            for i in range(parameters['nj']):
                s = 'q_cj[y,x,{0}]'.format(i)
                self.Q_cj[self.y, self.x, i] = self.parameters[s]
            self.Q_th[self.y, self.x] = self.parameters['q_th[y,x]']
            self.Q_o[self.y, self.x, :] = 0.0
            self.Q_o[self.y, self.x, 3] = self.parameters['q_o[y,x]']
        else:
            pass

    def CAtimeStep(self, compare_cy_py=False):
        self.dt = self.calc_dt()
        if compare_cy_py is False:
            # Normal time step
            self.T_1()
            self.sanityCheck()
            self.T_2()
            self.sanityCheck()
            self.I_1()
            self.sanityCheck()
            self.I_2()
            self.sanityCheck()
            self.I_1() # Try to get nonzero speed in "new" cells
            self.sanityCheck()
            self.I_3()
            self.sanityCheck()
            self.I_4()
            self.sanityCheck()
        elif compare_cy_py is True:
            self.time_step_compare_cy_py(self.global_grid)

    def plot3d(self, i):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X[:, :, 0], self.X[:, :, 1], self.Q_a)
        t = np.zeros((self.Ny, self.Nx))
        t[self.Q_th>0] += self.Q_a[self.Q_th>0]
        ax.plot_surface(self.X[:, :, 0], self.X[:, :, 1], self.Q_th+t, cmap='spring')
        # ma.axisEqual3D(ax)
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        ma.ensure_dir(self.parameters['save_dir'] + '3D/')
        d = self.parameters['save_dir'] + '3D/'
        plt.savefig(os.path.join(d, '%03ix%03i_%s_%03i_thetar%0.0f.png' % (
            self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r'])),
                    bbox_inches='tight', pad_inches=0, dpi=240)
        plt.close("all")

    def plot2d_current(self, i):
        N = self.Nj

        gs = gridspec.GridSpec(20, 10*N)
        fig = plt.figure(figsize=(18, 18))
        ax1 = fig.add_subplot(gs[:9, :-3])
        cbar2 = fig.add_subplot(gs[9:10, :-3])
        ax2 = fig.add_subplot(gs[:9, -3:], sharey=ax1)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax = []
        for n in range(N):
            try:
                ax.append(fig.add_subplot(gs[10:-2, n*10:(n+1)*10], sharey=ax[0]))
                plt.setp(ax[n].get_yticklabels(), visible=False)
            except IndexError:
                ax.append(fig.add_subplot(gs[10:-2, n*10:(n+1)*10]))
        cbar = fig.add_subplot(gs[-1,:])
        d = (self.Q_th[1:-1, 1:-1])
        d[d==0] = np.nan
        points = ax1.pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                                  d,
                                  cmap='Blues', vmin=0)
        ax1.contour(self.X[:, :, 0], self.X[:, :, 1],
                      self.bathymetry, colors='black', alpha=0.4)
        cbar2.set_title("TC height")
        plt.colorbar(points, shrink=0.8, cax=cbar2, orientation='horizontal')
        ax1.set_title('Q_th[1:-1,1:-1]. N = {0}'.format(i+1))

        aveQv = np.divide(self.Q_v.sum(1), (self.Q_v!=0).sum(1))
        ax2.plot(aveQv, self.X[0,:, 0])
        ax2.set_xlabel("Average speed")
        ax2.set_ylim([0, np.max(self.X[0,:,0])])



        sum_qcj = np.sum(self.Q_cj[1:-1,1:-1], axis=2)
        for n in range(N):
            ax[n].set_title("D_sj = {0}".format(self.D_sj[n]))
            p = ax[n].pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                                  self.Q_cj[1:-1,1:-1,n]/sum_qcj,
                                  cmap='Reds', vmin=0, vmax=1.)
            ax[n].contour(self.X[:, :, 0], self.X[:, :, 1],
                        self.bathymetry, colors='black', alpha=0.4)
        plt.colorbar(p, cax=cbar, orientation='horizontal', shrink=0.6)

        fig.tight_layout()
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        plt.savefig(os.path.join(self.parameters['save_dir'], 'TC_%03ix%03i_%s_thetar%0.0f_%03i.png' % (
            self.Nx, self.Ny, s1, self.parameters['theta_r'], i + 1)),
                    bbox_inches='tight', pad_inches=0, dpi=240)

    def plot2d_bed(self, i):
        N = self.Nj

        gs = gridspec.GridSpec(20, 10*N)
        fig = plt.figure(figsize=(18, 18))
        ax1 = fig.add_subplot(gs[:5, :-1])
        cbar2 = fig.add_subplot(gs[:5, -1:])
        ax = []
        for n in range(N):
            try:
                ax.append(fig.add_subplot(gs[5:-1, n*10:(n+1)*10], sharey=ax[0]))
                plt.setp(ax[n].get_yticklabels(), visible=False)
            except IndexError:
                ax.append(fig.add_subplot(gs[5:-1, n*10:(n+1)*10]))
        cbar = fig.add_subplot(gs[-1,:])
        d = (self.Q_d[1:-1, 1:-1] - self.parameters['q_d[interior]'])
        dd = np.abs(self.Q_d[1:-1, 1:-1] - self.parameters['q_d[interior]'])
        points = ax1.pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                                  d.transpose(),
                                  cmap='seismic', vmin=-1., vmax=1.)
        ax1.contour(self.X[:, :, 0], self.X[:, :, 1],
                      self.bathymetry.transpose(), colors='black', alpha=0.4)
        plt.colorbar(points, shrink=0.8, cax=cbar2, use_gridspec=True)
        ax1.set_title('\Delta Q_d[1:-1,1:-1]. N = {0}'.format(i + 1))


        for n in range(N):
            ax[n].set_title("D_sj = {0}".format(self.D_sj[n]))
            p = ax[n].pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                                  self.Q_cbj[1:-1,1:-1,n] * dd,
                                  cmap='Reds', vmin=0, vmax=1.)
            ax[n].contour(self.X[:, :, 0], self.X[:, :, 1],
                        self.bathymetry, colors='black', alpha=0.4)
        plt.colorbar(p, cax=cbar, orientation='horizontal', shrink=0.6)

        fig.tight_layout()
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        plt.savefig(os.path.join(self.parameters['save_dir'], 'BED_%03ix%03i_%s_thetar%0.0f_%03i.png' % (
            self.Nx, self.Ny, s1, self.parameters['theta_r'], i + 1)),
                    bbox_inches='tight', pad_inches=0, dpi=240)
        plt.close('all')

    def plot2d(self, i):
        self.plot2d_bed(i)
        self.plot2d_current(i)
        #fig = plt.figure(figsize=(14, 21))
        #ax = [fig.add_subplot(3, 2, i, aspect='equal') for i in range(1, 7)]
        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(21, 21), sharex=True, sharey=True, subplot_kw={"aspect": "equal"})
        ax = ax.flatten()
        ind = np.unravel_index(np.argmax(self.Q_th, axis=None), self.Q_th.shape)

        #points = ax[0].scatter(self.X[:, :, 0].flatten(), self.X[:, :, 1].flatten(),
         #                      c=self.Q_cj[:, :, 0].flatten())

        points = ax[0].pcolormesh(self.X[:, :, 0], self.X[:, :, 1],
                               self.Q_cj[:, :, 0])

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_cj[:,:,0]. n = ' + str(i + 1))

        points = ax[1].pcolormesh(self.X[:, :, 0], self.X[:, :, 1],
                               np.log10(self.Q_th))
        ax[1].contour(self.X[:, :, 0], self.X[:, :, 1],
                               self.bathymetry, colors='black', alpha=0.4)
        ax[1].scatter(self.X[ind[0], ind[1], 0], self.X[ind[0], ind[1], 1], c='r')  # Targeting
        plt.colorbar(points, shrink=0.6, ax=ax[1])
        ax[1].set_title('Q_th')

        points = ax[2].pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                               self.Q_cbj[1:-1, 1:-1, 0])
        plt.colorbar(points, shrink=0.6, ax=ax[2])
        ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

        points = ax[3].pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                               self.Q_d[1:-1, 1:-1]-self.parameters['q_d[interior]'],
                                  cmap='seismic', vmin=-1., vmax=1.)
        ax[3].contour(self.X[:, :, 0], self.X[:, :, 1],
                               self.bathymetry, colors='black', alpha=0.4)
        plt.colorbar(points, shrink=0.6, ax=ax[3])
        ax[3].set_title('\Delta Q_d[1:-1,1:-1]')

        try:
            points = ax[4].pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                                   self.Q_cbj[1:-1, 1:-1, 1])
            plt.colorbar(points, shrink=0.6, ax=ax[4])
            ax[4].set_title('Q_cbj[1:-1,1:-1,1]')

            points = ax[5].pcolormesh(self.X[1:-1, 1:-1, 0], self.X[1:-1, 1:-1, 1],
                                   self.Q_cj[1:-1, 1:-1, 1])
            plt.colorbar(points, shrink=0.6, ax=ax[5])
            ax[5].set_title('Q_cj[1:-1,1:-1,1]')
        except:
            pass
        plt.tight_layout()
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        plt.savefig(os.path.join(self.parameters['save_dir'], 'full_%03ix%03i_%s_%03i_thetar%0.0f.png' % (
            self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r'])),
                    bbox_inches='tight', pad_inches=0, dpi=240)
        plt.close('all')

    def plot1d(self, i):
        # Plot the 1D substates along the bottom of the channel
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(2, 2, i, aspect='auto') for i in range(1, 5)]
        lnns1 = ax[0].plot(np.arange(len(self.ch_bot_thickness)), self.ch_bot_thickness, label='Q_th', color='black')
        # ax[0].plot((5, 5), (0, 3), 'k-')
        ax[0].set_title('1D Q_th, time step = %03i' % (i + 1))
        # ax[0].set_ylim([0, 10])
        ax[0].set_ylabel('Q_{th}')
        ax3 = ax[0].twinx()
        lnns2 = []
        for ii in range(self.parameters['nj']):
            lnns2.append(ax3.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_thickness_cons[ii], linestyle='--',
                                  label='Q_cbj[y,x,{0}]'.format(ii)))
        # lnns2 = ax3.plot(np.arange(len(ch_bot_speed)), ch_bot_thickness_cons0, color='tab:red', linestyle='--', label='Q_cbj[y,x,0]')
        # lnns3 = ax3.plot(np.arange(len(ch_bot_speed)), ch_bot_thickness_cons1, color='tab:cyan', linestyle='-.', label='Q_cbj[y,x,1]')
        lnns = lnns1
        for z in range(len(lnns2)):
            lnns += lnns2[z]
        labbs = [l.get_label() for l in lnns]
        ax[0].legend(lnns, labbs, loc=0)
        maxconc = np.max([np.amax(self.ch_bot_thickness_cons)])
        upper = (0.1 * (maxconc // 0.1) + 0.1)
        # ax3.set_ylim([0, upper])
        ax3.set_ylabel('Concentration')
        for xx in range(0, 3):
            ax[xx].set_xlabel('y: Channel axis')
        # plt.savefig('ch_bot_thickness_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)

        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(len(self.ch_bot_thickness)), self.ch_bot_thickness, label='Thickness')
        # plt.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_speed, 'c.', label='Speed')
        # plt.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_outflow, 'r-.', label='Outflow')
        # plt.legend()
        # plt.plot((5, 5), (0, 3), 'k-')
        # plt.title('1D Q_th, time step = %03i' % (i+1))
        # plt.ylim([0, 3])
        # plt.savefig('ch_bot_combi_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)

        # plt.figure(figsize=(10, 6))
        ax[1].plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_speed)
        # ax[1].plot((5, 5), (0, 0.2), 'k-')
        ax[1].set_title('1D speed, time step = %03i' % (i + 1))
        # ax[1].set_ylim([0, 0.2])
        ax[1].set_ylabel('Q_{v}')
        # plt.savefig('ch_bot_speed_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)

        # ax[2].figure(figsize=(10, 6))
        ax[2].plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_outflow)
        # ax[2].plot((5, 5), (0, 2), 'k-')
        ax[2].set_title('Sum 1D outflow, time step = %03i' % (i + 1))
        # ax[2].set_ylim([0, 2])
        ax[2].set_ylabel('sum(Q_{o}[y,x])')

        lns1 = ax[3].plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_sediment, label='Q_{d}', color='black')
        # ax[3].plot((5, 5), (0, 2), 'k-')
        ax[3].set_title('Sediment, time step = %03i' % (i + 1))
        # ax[3].set_ylim([0, 2])
        ax[3].set_ylabel('Q_{d}[y,x])')
        ax2 = ax[3].twinx()
        lns2 = []
        for ii in range(self.parameters['nj']):
            lns2.append(
                ax2.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_sediment_cons[ii],
                         label='Q_cbj[y,x,{0}]'.format(ii)))
        # lns2 = ax2.plot(np.arange(len(ch_bot_speed)), ch_bot_sediment_cons0, color='tab:red', label='Q_cbj[y,x,0]')
        # lns3 = ax2.plot(np.arange(len(ch_bot_speed)), ch_bot_sediment_cons1, color='tab:cyan', label='Q_cbj[y,x,1]')
        lns = lns1
        for z in range(len(lns2)):
            lns += lns2[z]
        labs = [l.get_label() for l in lns]
        ax[3].legend(lns, labs, loc=0)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Concentration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.parameters['save_dir'], 'ch_bot_%03i.png' % (i + 1)),
                    bbox_inches='tight', pad_inches=0, dpi=240)

        plt.close('all')

    def printSubstates_to_screen(self, i):
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(2, 2, i, aspect='equal') for i in range(1, 5)]
        ind = np.unravel_index(np.argmax(self.Q_th, axis=None), self.Q_th.shape)

        points = ax[0].scatter(self.X[:, :, 0], self.X[:, :, 1], marker='h',
                               c=self.Q_cj[:, :, 0].flatten())

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_cj[:,:,0]. n = ' + str(i + 1))

        points = ax[1].scatter(self.X[:, :, 0].flatten(), self.X[:, :, 1].flatten(), marker='h',
                               c=self.Q_th.flatten())
        ax[1].scatter(self.X[ind[0], ind[1], 0], self.X[ind[0], ind[1], 1], c='r')  # Targeting
        plt.colorbar(points, shrink=0.6, ax=ax[1])
        ax[1].set_title('Q_th')

        points = ax[2].scatter(self.X[1:-1, 1:-1, 0].flatten(), self.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.Q_cbj[1:-1, 1:-1, 0].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[2])
        ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

        points = ax[3].scatter(self.X[1:-1, 1:-1, 0].flatten(), self.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.Q_d[1:-1, 1:-1].flatten())
        # datacursor(bbox=dict(alpha=1))
        plt.colorbar(points, shrink=0.6, ax=ax[3])
        ax[3].set_title('Q_d[1:-1,1:-1]')
        plt.tight_layout()
        plt.show()
        # s1 = str(self.terrain) if self.terrain is None else self.terrain
        # plt.savefig(os.path.join('./Data/','full_%03ix%03i_%s_%03i_thetar%0.0f.png' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']) ),
        #             bbox_inches='tight', pad_inches=0, dpi=240)
        # plt.close('all')

        # Plot the 1D substates along the bottom of the channel
        # fig = plt.figure(figsize=(10, 6))
        # ax = [fig.add_subplot(2, 2, i, aspect='auto') for i in range(1, 4)]
        # ax[0].plot(np.arange(len(self.ch_bot_thickness)), self.ch_bot_thickness)
        # ax[0].plot((5, 5), (0, 3), 'k-')
        # ax[0].set_title('1D Q_th, time step = %03i' % (i+1))
        # ax[0].set_ylim([0, 3])
        # ax[0].set_ylabel('Q_{th}')
        # for xx in range(0,3):
        #     ax[xx].set_xlabel('y: Channel axis')
        # # plt.savefig('ch_bot_thickness_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)
        #
        #
        # # plt.figure(figsize=(10, 6))
        # # plt.plot(np.arange(len(self.ch_bot_thickness)), self.ch_bot_thickness, label='Thickness')
        # # plt.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_speed, 'c.', label='Speed')
        # # plt.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_outflow, 'r-.', label='Outflow')
        # # plt.legend()
        # # plt.plot((5, 5), (0, 3), 'k-')
        # # plt.title('1D Q_th, time step = %03i' % (i+1))
        # # plt.ylim([0, 3])
        # # plt.savefig('ch_bot_combi_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)
        #
        #
        # # plt.figure(figsize=(10, 6))
        # ax[1].plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_speed)
        # ax[1].plot((5, 5),(0, 1), 'k-')
        # ax[1].set_title('1D speed, time step = %03i' % (i+1))
        # ax[1].set_ylim([0, 1])
        # ax[1].set_ylabel('Q_{v}')
        # # plt.savefig('ch_bot_speed_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)
        #
        # # ax[2].figure(figsize=(10, 6))
        # ax[2].plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_outflow)
        # ax[2].plot((5, 5), (0, 2), 'k-')
        # ax[2].set_title('Sum 1D outflow, time step = %03i' % (i + 1))
        # ax[2].set_ylim([0, 2])
        # ax[2].set_ylabel('sum(Q_{o}[y,x])')
        # plt.tight_layout()
        # plt.show()
        # # plt.savefig('./Data/ch_bot_%03i.png' % (i + 1), bbox_inches='tight', pad_inches=0, dpi=240)
        #
        #
        # plt.close('all')

    def sampleValues(self):
        self.time.append(self.dt)
        self.mass.append(self.Q_th[:, :, None] * self.Q_cj)
        self.massBed.append(np.sum(self.Q_d[1:-1, 1:-1].flatten(), axis=None))
        self.density.append(np.amax(self.Q_cj[:, :, 0].flatten()))
        self.beddensity.append(np.amax(self.Q_cbj[:, :, 0]))

        # Plot sub states in channel center
        bottom_indices = self.bot_indices
        self.ch_bot_thickness = [self.Q_th[self.bot_indices[i]] for i in range(len(self.bot_indices))]
        self.ch_bot_speed = [self.Q_v[self.bot_indices[i]] for i in range(len(self.bot_indices))]
        self.ch_bot_thickness_cons = []
        self.ch_bot_sediment_cons = []
        for jj in range(self.parameters['nj']):
            self.ch_bot_thickness_cons.append(
                [self.Q_cj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
            self.ch_bot_sediment_cons.append(
                [self.Q_cbj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
        self.ch_bot_sediment = [self.Q_d[bottom_indices[i]] for i in range(len(bottom_indices))]
        self.ch_bot_outflow = [sum(self.Q_o[self.bot_indices[i]]) for i in range(len(self.bot_indices))]

    def addSource(self):
        q_th0 = self.sourcevalues['Q_th']
        q_v0 = self.sourcevalues['Q_v']
        q_cj0: list = self.sourcevalues['Q_cj']
        # grid.Q_v[y, x] += 0.2
        # grid.Q_cj[y, x, 0] += 0.003
        # grid.Q_th[y, x] += 1.5
        if (self.parameters['x'] is not None) and (self.parameters['y'] is not None):
            if self.parameters['q_th[y,x]'] > 0:
                # self.Q_v[self.y, self.x] = q_v0 # TODO, test
                self.Q_v[self.y, self.x] = (self.Q_v[self.y, self.x] * self.Q_th[
                    self.y, self.x] + q_v0 * q_th0 * self.dt) / (q_th0 * self.dt + self.Q_th[self.y, self.x])
                self.Q_th[self.y, self.x] += q_th0 * self.dt
                for particle_type in range(self.Nj):
                    self.Q_cj[self.y, self.x, particle_type] = (self.Q_cj[self.y, self.x, particle_type] * self.Q_th[
                        self.y, self.x] + q_cj0[particle_type] * q_th0 * self.dt) / (
                                                                       q_th0 * self.dt + self.Q_th[self.y, self.x])
        else:
            pass

    def add_source_constant(self):
        """Note: this may turn into a drain if Q_th[nb] > Q_th0 !"""
        if (self.parameters['x'] is not None) and (self.parameters['y'] is not None):
            self.Q_v[self.y, self.x] = self.parameters['q_v[y,x]']
            for i in range(parameters['nj']):
                s = 'q_cj[y,x,{0}]'.format(i)
                self.Q_cj[self.y, self.x, i] = self.parameters[s]
            self.Q_th[self.y, self.x] = self.parameters['q_th[y,x]']
        else:
            pass
        # if((self.Q_th[self.y,self.x] < self.parameters['q_th[y,x]']).sum()):
        #     q_v0 = self.parameters['q_v[y,x]']
        #     q_cj0 = self.parameters['q_cj[y,x,0]']
        #     amount = self.parameters['q_th[y,x]'] - self.Q_th[self.y,self.x]
        #
        #     self.Q_v[self.y, self.x] = (self.Q_v[self.y, self.x] *
        #                                      self.Q_th[self.y, self.x] + q_v0 * amount) / \
        #                                     (amount * self.dt + self.Q_th[self.y, self.x])
        #     self.Q_cj[self.y, self.x, 0] = (self.Q_cj[self.y, self.x, 0] *
        #                                          self.Q_th[self.y, self.x] + q_cj0 * amount) /\
        #                                         (1.5 * self.dt + self.Q_th[self.y, self.x])
        #     self.Q_th[self.y, self.x] += amount

    def set_BC_absorb_bed(self):
        self.Q_d[-1, :] = parameters['q_d[interior]']
        self.Q_a[-1, :] = self.Q_a_south

    def set_BC_absorb_current(self):
        self.Q_th[-2, :] = 0.0
        self.Q_cj[-2, :, :] = 0.0
        self.Q_o[-2, :, :] = 0.0
        self.Q_v[-2, :] = 0.0

    def writeToTxt(self, i):
        d = self.parameters['save_dir'] + 'stability/'
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        time_file = os.path.join(d, 'full_%03ix%03i_%s_%03i_thetar%0.0f_time.txt'
                                 % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']))
        mass_file = os.path.join(d, 'full_%03ix%03i_%s_%03i_thetar%0.0f_mass.txt'
                                 % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']))
        maxdensity_file = os.path.join(d, 'full_%03ix%03i_%s_%03i_thetar%0.0f_maxdensity.txt'
                                       % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']))
        with open(time_file, 'w') as f:
            for item in self.time:
                f.write("%s\n" % item)

        savemass = np.sum(self.mass, axis=(1, 2, 3))
        with open(mass_file, 'w') as f:
            for item in savemass:
                f.write("%s\n" % item)

        with open(maxdensity_file, 'w') as f:
            for item in self.density:
                f.write("%s\n" % item)

        # with open('full_%03ix%03i_%s_%03i_thetar%0.0f_maxerosionrate.txt' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']),
        #           'w') as f:
        #     for item in self.Erosionrate:
        #         f.write("%s\n" % item)
        #
        # with open(
        #         'full_%03ix%03i_%s_%03i_thetar%0.0f_maxdepositionrate.txt' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']),
        #         'w') as f:
        #     for item in self.Depositionrate:
        #         f.write("%s\n" % item)
        #
        # mt = np.sum(self.mass, axis=(1, 2, 3))
        # mb = np.array(self.massBed)
        # with open('full_%03ix%03i_%s_%03i_thetar%0.0f_totalmass.txt' % (self.Nx,self.Ny, s1, i + 1, self.parameters['theta_r']),
        #           'w') as f:
        #     for item in (mt + mb):
        #         f.write("%s\n" % item)
        #
        # with open('full_%03ix%03i_%s_%03i_thetar%0.0f_head_velocity.txt' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']),
        #           'w') as f:
        #     for item in self.head_velocity:
        #         f.write("%s\n" % item)

    def plotStabilityCurves(self, i_values):
        """
        :param i_values: Sample iteration values
        """
        plt.figure(figsize=(15, 6))
        fontsize = 16
        ax1: plt.Axes = plt.subplot(131)
        ax1.plot(i_values, self.density)
        ax1.set_ylabel('$Q_{cj}^{(n)}$', fontsize=fontsize)
        ax1.set_xlabel('$n$', fontsize=fontsize)

        ax2: plt.Axes = plt.subplot(133)
        ax2.plot(i_values, self.time)
        ax2.set_xscale('log')
        ax2.set_ylabel('$\Delta t^{(n)}$', fontsize=fontsize)
        ax2.set_xlabel('$n$', fontsize=fontsize)
        plt.tight_layout()

        ax3: plt.Axes = plt.subplot(132)
        ax3.plot(i_values, np.sum(self.mass, axis=(1, 2, 3)))
        ax3.set_ylabel(' \"Mass\" = $Q_{th} \cdot Q_{cj}$', fontsize=fontsize)
        ax3.set_xlabel('$n$', fontsize=fontsize)

        s1 = str(self.terrain) if self.terrain is None else self.terrain
        d = self.parameters['save_dir'] + 'stability/'
        ma.ensure_dir(d)
        plt.savefig(os.path.join(d, 'full_%03ix%03i_%s_%03i_thetar%0.0f_stability.png'
                                 % (self.Nx, self.Ny, s1, i_values[-1], self.parameters['theta_r'])),
                    bbox_inches='tight', pad_inches=0)

    def print_npy(self, i, mpisize=0):
        """
        This function prints substates to .npy format
        """
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        d = self.parameters['save_dir'] + 'npy_files/'
        if self.mpi:
            d = d + ''.join("rank_{0}_of_{1}".format(self.my_rank, mpisize))
        ma.ensure_dir(d)
        d = d + ''.join('%03ix%03i_%s_%03i_thetar%0.0f_'
                        % (self.Nx, self.Ny, s1, i, self.parameters['theta_r']))

        np.save(d + str('self.Q_th'), self.Q_th)
        np.save(d + str('self.Q_v'),self.Q_v)
        np.save(d + str('self.Q_cj'),self.Q_cj)
        np.save(d + str('self.Q_cbj'),self.Q_cbj)
        np.save(d + str('self.Q_d'),self.Q_d)
        np.save(d + str('self.Q_o'),self.Q_o)
        np.save(d + str('self.Q_a'),self.Q_a)

    def load_npy(self, i_value):
        pass



    def plot_bathy(self):
        fig = plt.figure(figsize=(10, 6))
        ind = np.unravel_index(np.argmax(self.Q_th, axis=None), self.Q_th.shape)

        points = plt.scatter(self.X[:, :, 0].flatten(), self.X[:, :, 1].flatten(),
                               c=self.Q_a[:, :].flatten())
        plt.scatter(self.X[self.y, self.x, 0], self.X[self.y,self.x, 1], c='r', label='source')  # Targeting
        plt.legend()
        plt.colorbar(points, shrink=0.6)
        plt.savefig(os.path.join(self.parameters['save_dir'], 'bathymetry.png'),
                    bbox_inches='tight', pad_inches=0, dpi=240)
        plt.close('all')

    def print_log(self, loop: str):
        """ This function prints the variables used to a timestamped file.\
            "loop" is a string that will be added to the log file."""
        from time import localtime, strftime
        ts = localtime()
        s = (strftime("%Y-%m-%d %H:%M:%S", ts))
        d = self.parameters['save_dir'] + 'log/'
        ma.ensure_dir(d)
        d = d+''.join(s)
        d = d + ''.join('.txt')
        with open(d, 'w') as f:
            f.write(loop + "\n")
            ma.dump(self.parameters, output=f)
        f.close()

def read_which_configs():
    try:
        with open('./Config/configs.txt') as f:
            content = f.readlines()
    except:
        raise FileNotFoundError('Could not find config file of name configs.txt')
    content = [x.strip() for x in content]
    if len(content) == 0:
        raise RuntimeError('No .ini files specified in configs.txt!')
    return content


def set_which_plots(parameters):
    """
    This function tries to create a boolean list, specifying which plots to \\
    produce in the simulation.
    :param parameters: Parameters specifying the
    :return: A list of boolean values used to determine which plots to produce.
    """
    try:
        plot_bool = [0, 0, 0, 0, 0, 0]
        if type(parameters['output']) is str:
            if len(parameters['output'].split()) >= len(parameters['output'].split(',')):
                parameters['output'] = parameters['output'].split()
            else:
                parameters['output'] = parameters['output'].split(',')

        for p in parameters['output']:
            if p == '3d':
                plot_bool[2] = 1
            elif p == '2d':
                plot_bool[1] = 1
            elif p == '1d':
                plot_bool[0] = 1
            elif p == 'stability':
                plot_bool[3] = 1
            elif p == 'npy':
                plot_bool[4] = 1
            elif p == 'bathymetry':
                plot_bool[5] = 1
    except:
        plot_bool = [0, 0, 0, 0, 0, 0]
        print('Warning: Parameter \'output\' not specified. Please specify which types of output to print!'
              ' Eg. output = 3d,2d,1d,stability,txt')
    return plot_bool


def import_parameters(filename=None):
    if filename is None:
        filename = './Config/test.ini'
    else:
        filename = './Config/' + filename + '.ini'
    from configparser import ConfigParser, ExtendedInterpolation

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    if os.path.exists(filename):
        parser.read(filename)
    else:
        raise FileNotFoundError('Could not find file with name {0}'.format(filename))
    sections = parser.sections()

    items: list = parser.items(sections[0])

    parameters = {}
    for i in range(len(items)):
        try:
            parameters[items[i][0]] = eval(items[i][1])
        except:
            parameters[items[i][0]] = (items[i][1])

    return parameters

def end_signal(it, parameters, t):
    """ Takes in number of iterations and compares with either\
        specified number of iterations or elapsed simulation time 't'.\
        Returns True if simulation has lasted specified length."""
    try:
        tend = parameters['t_end']
        if t >= tend:
            return True
    except KeyError:
        num_iterations = parameters['num_iterations']
        if it > num_iterations:
            return True
    return False



if __name__ == "__main__":
    '''
    This script reads all configurations in configs.txt, and runs all the simulations\
    specified there!
    '''

    ma.ensure_dir('./Bathymetry/')
    ma.ensure_dir('./Data/')
    ma.ensure_dir('./Config/')
    ma.ensure_file('./Config/configs.txt')
    configs = read_which_configs()

    for config in configs:
        parameters = import_parameters(config)
        save_dir = './Data/' + config + '/'
        ma.ensure_dir(save_dir)
        parameters['save_dir'] = save_dir
        # q_th0 = parameters['q_th[y,x]']
        # q_cj0 = parameters['q_cj[y,x,0]']
        # q_v0 = parameters['q_v[y,x]']
        plot_bool = set_which_plots(parameters)

        sample_rate = parameters['sample_rate']
        CAenv = CAenvironment(parameters)
        if plot_bool[5]: CAenv.plot_bathy()
        start = timer()
        j_values = []
        j = 0
        while True:
        # for j in range(parameters['num_iterations']):
            if end_signal(j, parameters, sum(CAenv.save_dt)):
                break
            CAenv.addSource()
            # CAenv.add_source_constant()
            CAenv.CAtimeStep(compare_cy_py=False)
            CAenv.set_BC_absorb_bed()
            CAenv.set_BC_absorb_current()
            ind = np.unravel_index(np.argmax(CAenv.Q_th, axis=None), CAenv.Q_th.shape)
            CAenv.head_velocity.append(CAenv.Q_v[ind])
            if ((j + 1) % sample_rate == 0) and j > 0:
                j_values.append(j + 1)
                CAenv.sampleValues()
                if plot_bool[0]: CAenv.plot1d(j)
                if plot_bool[1]: CAenv.plot2d(j)
                if plot_bool[2]: CAenv.plot3d(j)
                if plot_bool[4]: CAenv.print_npy(j)
            j += 1

        if plot_bool[3]:
            CAenv.plotStabilityCurves(j_values)
            CAenv.writeToTxt(j)
        wtime = timer() - start
        stime = sum(CAenv.save_dt)
        CAenv.print_log("Wall time used = {0}, simulated time = {1}\n"
                        "n_cores = 1".format(wtime,stime))
        print('{0} is complete. Wall time elapsed = {1}\n'
              '{2} seconds simulation time.'.format(config, wtime, stime))

