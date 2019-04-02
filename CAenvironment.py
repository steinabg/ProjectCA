from hexgrid import *
import matplotlib.pyplot as plt
# import sys
import os.path
import os
# from mpldatacursor import datacursor
from timeit import default_timer as timer
# sys.path.append('..')
np.set_printoptions(suppress=True, precision=3)



# theta_r = 80
def import_parameters(filename = None):
    if filename is None:
        filename = './Config/test.ini'
    else:
        filename = './Config/' + filename + '.ini'
    from configparser import ConfigParser, ExtendedInterpolation
    import numpy as np # Must be here for imported expressions to be evaluated

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    if os.path.exists(filename):
        parser.read(filename)
    else:
        raise FileNotFoundError('Could not find file with name {0}'.format(filename))
    sections = parser.sections()

    items = parser.items(sections[0])

    parameters = {}
    for i in range(len(items)):
        try:
            parameters[items[i][0]] = eval(items[i][1])
        except:
            parameters[items[i][0]] = (items[i][1])

    return parameters



class CAenvironment():

    def __init__(self, parameters, global_grid = True):
        #     plt.ioff()
        self.global_grid = global_grid # If False this environment describes a local CA (part of a grid)
        self.parameters = parameters
        self.Ny = parameters['ny']
        self.Nx = parameters['nx']
        self.Nj = parameters['nj']
        self.dx = parameters['dx']



        self.Q_th = np.zeros((self.Ny, self.Nx))  # Turbidity current thickness
        self.Q_v = np.zeros((self.Ny, self.Nx))  # Turbidity current speed (scalar)
        self.Q_cj = np.zeros((self.Ny, self.Nx, self.Nj))  # jth current sediment volume concentration
        self.Q_cbj = np.zeros((self.Ny, self.Nx, self.Nj))  # jth bed sediment volume fraction
        if global_grid == True:
            self.Q_d = np.ones((self.Ny, self.Nx)) * np.inf  # Thickness of soft sediment
            self.Q_d[1:-1, 1:-1] = 0  # Part of the initialization
        else:
            self.Q_d = np.zeros((self.Ny,self.Nx))
        self.Q_o = np.zeros((self.Ny, self.Nx, 6))  # Density current outflow

        # Initial sand cover
        self.Q_cbj[1:-1, 1:-1, 0] = parameters['q_cbj[interior, 0]']  # 1
        self.Q_d[1:-1, 1:-1] = parameters['q_d[interior]']  # 1.0

        # Source area
        if (parameters['x'] is not None) and (parameters['y'] is not None):
            # if global_grid is True:
            self.y, self.x = np.meshgrid(parameters['y'],parameters['x'])
            self.Q_th[self.y, self.x] = parameters['q_th[y,x]']  # 1.5
            self.Q_v[self.y, self.x] = parameters['q_v[y,x]']  # 0.2
            self.Q_d[self.y, self.x] = parameters['q_d[y,x]']  # 1
            for particle_type in range(self.Nj):
                parameter_string1 = 'q_cj[y,x,' + str(particle_type) + ']'
                parameter_string2 = 'q_cbj[y,x,' + str(particle_type) + ']'
                self.Q_cj[self.y, self.x, particle_type] = parameters[parameter_string1]  # 0.003
                self.Q_cbj[self.y, self.x, particle_type] = parameters[parameter_string2]  # 1


        self.terrain = parameters['terrain']

        self.ICstates = [self.Q_th, self.Q_v, self.Q_cj, self.Q_cbj, self.Q_d, self.Q_o]

        self.grid = Hexgrid(self.Ny, self.Nx, ICstates=self.ICstates, reposeAngle=np.deg2rad(parameters['theta_r']),
                            dx=self.dx, terrain=self.terrain, global_grid=global_grid)

        self.grid.g = parameters['g']  # Gravitational acceleration
        self.grid.f = parameters['f']  # Darcy-Weisbach coeff
        self.grid.a = parameters['a']  # Empirical coefficient (used in I_3)
        self.grid.rho_a =parameters['rho_a'] # ambient density
        self.grid.rho_j =parameters['rho_j'] # List of current sediment densities
        self.grid.D_sj = parameters['d_sj'] # List of sediment-particle diameters
        self.grid.Nj = parameters['nj']  # Number of sediment types
        self.grid.c_D = parameters['c_d']  # Bed drag coefficient (table 3)
        self.grid.nu = parameters['nu']  # Kinematic viscosity of water at 5 degrees celcius
        self.grid.porosity = parameters['porosity']
        self.grid.p_f = parameters['p_f']  # Height threshold friction angle
        self.grid.p_adh = parameters['p_adh']

        if parameters['sphere_settling_velocity'] != 'salles':
            self.grid.v_sj = parameters['sphere_settling_velocity']
        else:
            self.grid.v_sj = ma.calc_settling_speed(self.grid.D_sj, self.grid.rho_a, self.grid.rho_j,
                                                    self.grid.g, self.grid.nu)
        self.time = []
        self.mass = []
        self.massBed = []
        self.density = []
        self.beddensity = []
        self.head_velocity = []


        self.save_path = './Data/'

        # For plotting in center of channel
        # self.ch_bot_thickness =



    def CAtimeStep(self, compare_cy_py=False):
        if compare_cy_py is False:
            # Normal time step
            self.grid.time_step(self.global_grid)
        elif compare_cy_py is True:
            self.grid.time_step_compare_cy_py(self.global_grid)

    def printSubstates(self, i):
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(3, 2, i, aspect='equal') for i in range(1, 7)]
        ind = np.unravel_index(np.argmax(self.grid.Q_th, axis=None), self.grid.Q_th.shape)

        points = ax[0].scatter(self.grid.X[:, :, 0].flatten(), self.grid.X[:, :, 1].flatten(), marker='h',
                               c=self.grid.Q_cj[:, :, 0].flatten())

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_cj[:,:,0]. n = ' + str(i + 1))

        points = ax[1].scatter(self.grid.X[:, :, 0].flatten(), self.grid.X[:, :, 1].flatten(), marker='h',
                               c=self.grid.Q_th.flatten())
        ax[1].scatter(self.grid.X[ind[0],ind[1],0], self.grid.X[ind[0],ind[1],1], c='r')  # Targeting
        plt.colorbar(points, shrink=0.6, ax=ax[1])
        ax[1].set_title('Q_th')

        points = ax[2].scatter(self.grid.X[1:-1, 1:-1, 0].flatten(), self.grid.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.grid.Q_cbj[1:-1, 1:-1, 0].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[2])
        ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

        points = ax[3].scatter(self.grid.X[1:-1, 1:-1, 0].flatten(), self.grid.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.grid.Q_d[1:-1, 1:-1].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[3])
        ax[3].set_title('Q_d[1:-1,1:-1]')

        try:
            points = ax[4].scatter(self.grid.X[1:-1, 1:-1, 0].flatten(), self.grid.X[1:-1, 1:-1, 1].flatten(), marker='h',
                                   c=self.grid.Q_cbj[1:-1, 1:-1,1].flatten())
            plt.colorbar(points, shrink=0.6, ax=ax[4])
            ax[4].set_title('Q_cbj[1:-1,1:-1,1]')

            points = ax[5].scatter(self.grid.X[1:-1, 1:-1, 0].flatten(), self.grid.X[1:-1, 1:-1, 1].flatten(), marker='h',
                                   c=self.grid.Q_cj[1:-1, 1:-1, 1].flatten())
            plt.colorbar(points, shrink=0.6, ax=ax[5])
            ax[5].set_title('Q_cj[1:-1,1:-1,1]')
        except:
            pass
        plt.tight_layout()
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        plt.savefig(os.path.join(self.parameters['save_dir'],'full_%03ix%03i_%s_%03i_thetar%0.0f.png' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']) ),
                    bbox_inches='tight', pad_inches=0, dpi=240)
        plt.close('all')

        # Plot the 1D substates along the bottom of the channel
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(2, 2, i, aspect='auto') for i in range(1, 5)]
        lnns1 = ax[0].plot(np.arange(len(self.ch_bot_thickness)), self.ch_bot_thickness, label='Q_th', color='black')
        ax[0].plot((5, 5), (0, 3), 'k-')
        ax[0].set_title('1D Q_th, time step = %03i' % (i+1))
        ax[0].set_ylim([0, 3])
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
        ax3.set_ylim([0, upper])
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
        ax[1].plot((5, 5), (0, 0.2), 'k-')
        ax[1].set_title('1D speed, time step = %03i' % (i + 1))
        ax[1].set_ylim([0, 0.2])
        ax[1].set_ylabel('Q_{v}')
        # plt.savefig('ch_bot_speed_%03i.png' %(i+1), bbox_inches='tight',pad_inches=0,dpi=240)

        # ax[2].figure(figsize=(10, 6))
        ax[2].plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_outflow)
        ax[2].plot((5, 5), (0, 2), 'k-')
        ax[2].set_title('Sum 1D outflow, time step = %03i' % (i + 1))
        ax[2].set_ylim([0, 2])
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
                ax2.plot(np.arange(len(self.ch_bot_speed)), self.ch_bot_sediment_cons[ii], label='Q_cbj[y,x,{0}]'.format(ii)))
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
        plt.savefig(os.path.join(self.parameters['save_dir'],'ch_bot_%03i.png' % (i + 1)),
                    bbox_inches='tight', pad_inches=0, dpi=240)


        plt.close('all')

    def printSubstates_to_screen(self, i):
        fig = plt.figure(figsize=(10, 6))
        ax = [fig.add_subplot(2, 2, i, aspect='equal') for i in range(1, 5)]
        ind = np.unravel_index(np.argmax(self.grid.Q_th, axis=None), self.grid.Q_th.shape)

        points = ax[0].scatter(self.grid.X[:, :, 0].flatten(), self.grid.X[:, :, 1].flatten(), marker='h',
                               c=self.grid.Q_cj[:, :, 0].flatten())

        plt.colorbar(points, shrink=0.6, ax=ax[0])
        ax[0].set_title('Q_cj[:,:,0]. n = ' + str(i + 1))

        points = ax[1].scatter(self.grid.X[:, :, 0].flatten(), self.grid.X[:, :, 1].flatten(), marker='h',
                               c=self.grid.Q_th.flatten())
        ax[1].scatter(self.grid.X[ind[0],ind[1],0], self.grid.X[ind[0],ind[1],1], c='r')  # Targeting
        plt.colorbar(points, shrink=0.6, ax=ax[1])
        ax[1].set_title('Q_th')

        points = ax[2].scatter(self.grid.X[1:-1, 1:-1, 0].flatten(), self.grid.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.grid.Q_cbj[1:-1, 1:-1, 0].flatten())
        plt.colorbar(points, shrink=0.6, ax=ax[2])
        ax[2].set_title('Q_cbj[1:-1,1:-1,0]')

        points = ax[3].scatter(self.grid.X[1:-1, 1:-1, 0].flatten(), self.grid.X[1:-1, 1:-1, 1].flatten(), marker='h',
                               c=self.grid.Q_d[1:-1, 1:-1].flatten())
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
        self.time.append(self.grid.dt)
        self.mass.append(self.grid.Q_th[:, :, None] * self.grid.Q_cj)
        self.massBed.append(np.sum(self.grid.Q_d[1:-1, 1:-1].flatten(), axis=None))
        self.density.append(np.amax(self.grid.Q_cj[:, :, 0].flatten()))
        self.beddensity.append(np.amax(self.grid.Q_cbj[:, :, 0]))


        # Plot sub states in channel center
        bottom_indices = self.grid.bot_indices
        self.ch_bot_thickness = [self.grid.Q_th[self.grid.bot_indices[i]] for i in range(len(self.grid.bot_indices))]
        self.ch_bot_speed = [self.grid.Q_v[self.grid.bot_indices[i]] for i in range(len(self.grid.bot_indices))]
        self.ch_bot_thickness_cons = []
        self.ch_bot_sediment_cons = []
        for jj in range(self.parameters['nj']):
            self.ch_bot_thickness_cons.append([self.grid.Q_cj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
            self.ch_bot_sediment_cons.append([self.grid.Q_cbj[bottom_indices[i] + (jj,)] for i in range(len(bottom_indices))])
        self.ch_bot_sediment = [self.grid.Q_d[bottom_indices[i]] for i in range(len(bottom_indices))]
        self.ch_bot_outflow = [sum(self.grid.Q_o[self.grid.bot_indices[i]]) for i in range(len(self.grid.bot_indices))]



    def addSource(self, q_th0, q_v0, q_cj0):
        # grid.Q_v[y, x] += 0.2
        # grid.Q_cj[y, x, 0] += 0.003
        # grid.Q_th[y, x] += 1.5
        if (self.parameters['x'] is not None) and (self.parameters['y'] is not None):
            if (self.parameters['q_th[y,x]'] > 0):
                self.grid.Q_v[self.y, self.x] = (self.grid.Q_v[self.y, self.x] * self.grid.Q_th[self.y,self.x] + q_v0 * q_th0*self.grid.dt) / (q_th0*self.grid.dt + self.grid.Q_th[self.y, self.x])
                self.grid.Q_th[self.y, self.x] += q_th0*self.grid.dt
                for particle_type in range(self.Nj):
                    self.grid.Q_cj[self.y, self.x, particle_type] = (self.grid.Q_cj[self.y, self.x, particle_type] * self.grid.Q_th[
                        self.y, self.x] + q_cj0 * q_th0 * self.grid.dt) / (
                                                                    1.5 * self.grid.dt + self.grid.Q_th[self.y, self.x])
        else:
            pass

    def add_source_constant(self, q_th0, q_v0, q_cj0):
        if (self.parameters['x'] is not None) and (self.parameters['y'] is not None):
            self.grid.Q_v[self.y, self.x] = 0.2
            self.grid.Q_cj[self.y, self.x, 0] = 0.003
            self.grid.Q_th[self.y, self.x] = 1.5
        else:
            pass
        # if((self.grid.Q_th[self.y,self.x] < self.parameters['q_th[y,x]']).sum()):
        #     q_v0 = self.parameters['q_v[y,x]']
        #     q_cj0 = self.parameters['q_cj[y,x,0]']
        #     amount = self.parameters['q_th[y,x]'] - self.Q_th[self.y,self.x]
        #
        #     self.grid.Q_v[self.y, self.x] = (self.grid.Q_v[self.y, self.x] *
        #                                      self.grid.Q_th[self.y, self.x] + q_v0 * amount) / \
        #                                     (amount * self.grid.dt + self.grid.Q_th[self.y, self.x])
        #     self.grid.Q_cj[self.y, self.x, 0] = (self.grid.Q_cj[self.y, self.x, 0] *
        #                                          self.grid.Q_th[self.y, self.x] + q_cj0 * amount) /\
        #                                         (1.5 * self.grid.dt + self.grid.Q_th[self.y, self.x])
        #     self.grid.Q_th[self.y, self.x] += amount


    def writeToTxt(self, i):
        s1 = str(self.terrain) if self.terrain is None else self.terrain
        time_file = os.path.join(self.parameters['save_dir'],'full_%03ix%03i_%s_%03i_thetar%0.0f_time.txt'
                                 % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']))
        mass_file = os.path.join(self.parameters['save_dir'],'full_%03ix%03i_%s_%03i_thetar%0.0f_mass.txt'
                                 % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']))
        maxdensity_file = os.path.join(self.parameters['save_dir'],'full_%03ix%03i_%s_%03i_thetar%0.0f_maxdensity.txt'
                                 % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']))
        with open(time_file, 'w') as f:
            for item in self.time:
                f.write("%s\n" % item)

        savemass = np.sum(self.mass, axis=(1, 2, 3))
        with open(mass_file, 'w') as f:
            for item in savemass:
                f.write("%s\n" % item)

        with open(maxdensity_file,'w') as f:
            for item in self.density:
                f.write("%s\n" % item)

        # with open('full_%03ix%03i_%s_%03i_thetar%0.0f_maxerosionrate.txt' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']),
        #           'w') as f:
        #     for item in self.grid.Erosionrate:
        #         f.write("%s\n" % item)
        #
        # with open(
        #         'full_%03ix%03i_%s_%03i_thetar%0.0f_maxdepositionrate.txt' % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r']),
        #         'w') as f:
        #     for item in self.grid.Depositionrate:
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


    def plotStabilityCurves(self, i):
        '''
        :param i: iteration
        '''
        plt.figure(figsize=(15, 6))
        fontsize = 16
        ax1 = plt.subplot(131)
        ax1.plot(self.density)
        ax1.set_ylabel('$Q_{cj}^{(n)}$', fontsize=fontsize)
        ax1.set_xlabel('$n$', fontsize=fontsize)

        ax2 = plt.subplot(133)
        ax2.plot(self.time)
        ax2.set_xscale('log')
        ax2.set_ylabel('$\Delta t^{(n)}$', fontsize=fontsize)
        ax2.set_xlabel('$n$', fontsize=fontsize)
        plt.tight_layout()

        ax3 = plt.subplot(132)
        ax3.plot(np.sum(self.mass, axis=(1, 2, 3)))
        ax3.set_ylabel(' \"Mass\" = $Q_{th} \cdot Q_{cj}$', fontsize=fontsize)
        ax3.set_xlabel('$n$', fontsize=fontsize)

        s1 = str(self.terrain) if self.terrain is None else self.terrain
        plt.savefig(os.path.join(self.parameters['save_dir'],'full_%03ix%03i_%s_%03i_thetar%0.0f_stability.png'
                                 % (self.Nx, self.Ny, s1, i + 1, self.parameters['theta_r'])),
                                 bbox_inches='tight', pad_inches=0)

def read_which_configs():
    try:
        with open('./Config/configs.txt') as f:
            content = f.readlines()
    except:
        raise FileNotFoundError('Could not find config file of name configs.txt')
    content = [x.strip() for x in content]
    return content

if __name__ == "__main__":
    '''
    This script reads all configurations in configs.txt, and runs all the simulations\
    specified there!
    '''
    ma.ensure_dir('./Bathymetry/')
    ma.ensure_dir('./Data/')
    configs = read_which_configs()

    for config in configs:
        parameters = import_parameters(config)
        save_dir = './Data/' + config + '/'
        ma.ensure_dir(save_dir)
        parameters['save_dir'] = save_dir
        q_th0 = parameters['q_th[y,x]']
        q_cj0 = parameters['q_cj[y,x,0]']
        q_v0 = parameters['q_v[y,x]']
        sample_rate = parameters['sample_rate']
        CAenv = CAenvironment(parameters)
        start = timer()
        for j in range(parameters['num_iterations']):

            CAenv.addSource(q_th0,q_v0, q_cj0)
            CAenv.CAtimeStep(compare_cy_py=False)
            ind = np.unravel_index(np.argmax(CAenv.grid.Q_th, axis=None), CAenv.grid.Q_th.shape)
            CAenv.head_velocity.append(CAenv.grid.Q_v[ind])
            if ( (j+1) % sample_rate == 0) and j > 0:
                CAenv.sampleValues()
                CAenv.printSubstates(j)
        CAenv.plotStabilityCurves(j)
        CAenv.writeToTxt(j)
        print('{0} is complete. Time elapsed = {1}'.format(config, timer()-start))




