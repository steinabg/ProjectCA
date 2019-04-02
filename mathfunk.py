import numpy as np
import warnings
import os

def ensure_file(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'a').close()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def compare_ndarray(array1: np.ndarray, array2: np.ndarray, tol):
    assert array1.shape == array2.shape
    array1[np.isinf(array1)] = 0
    array2[np.isinf(array2)] = 0
    temp = np.abs(array1 - array2)

    if (((array1 - array2) > tol).sum() >= 1):
        i = np.where(temp > 0)
        s = "Not equal!\n"
        for num in range(i[0].size):
            s = s + ''.join("array1[{0},{1},{2}] = {3}, array2[{4},{5},{6}] = {7} <-- diff = {8}\n"
                            .format(i[0][num],i[1][num],i[2][num], array1[i[0][num],i[1][num],i[2][num]],
                                    i[0][num],i[1][num],i[2][num], array2[i[0][num],i[1][num],i[2][num]],
                                    temp[i[0][num],i[1][num],i[2][num]]))
        raise Exception(s)
    else:
        return

def generate_rupert_inlet_bathymetry(repose_angle,dx, Ny=200, Nx=200, channel_amplitude=None, channel_width=None, channeldepth=None):
    # Nx = Ny = 200
    X = np.zeros((Nx, Ny))

    y_offset = np.round(Nx / 13.333)

    if channel_amplitude is None:
        channel_amplitude = np.round(Nx / 6.667)
    channel_amplitude = channel_amplitude/dx

    indicesX = np.arange(np.round(Ny / 3), Ny + 1)
    indicesY = int(np.round(Nx / 2) + y_offset) + np.round(
        channel_amplitude * np.sin(np.linspace(np.pi, 5 * np.pi / 2, num=int(Ny - indicesX[0]))))

    if channel_width is None:
        channel_width = int(np.round(Nx / 6.667))
    channel_width = int(np.round(channel_width/dx)) # Convert from meter to cells
    if channeldepth is None:
        channeldepth = np.minimum(int(np.round(Nx / 66.666)),3)
    cross_sectionY = channeldepth * np.sin(np.linspace(np.pi, 2 * np.pi, channel_width))
    # def x_func(arg):
    #     return arg*arg
    # cross_sectionY = channeldepth * x_func(np.linspace(-1, 1, channel_width))- channeldepth
    for i in range(len(indicesY)):
        for j in range(channel_width):
            X[int(indicesY[i] - channel_width + j), int(indicesX[i])] = cross_sectionY[j]

    for j in range(channel_width):
        X[int(np.round(Nx / 2) + y_offset - channel_width + j), np.arange(0, np.round(Ny / 3) + 1, dtype=np.int)] = \
        cross_sectionY[j]

    x = np.arange(channel_width)
    angle = [(np.arctan2(cross_sectionY[i], x[i])) for i in range(1, len(cross_sectionY))]
    angle = np.abs(angle)
    if np.max(angle) > repose_angle:
        warnings.warn("Bathymetry: Channel cross-section steepness {0} exceeds angle of repose {1}! "
                      "This will cause an avalanche.".format(np.max(angle), repose_angle))

    return X, cross_sectionY

def gen_sloped_plane(Ny: int, Nx: int, dx: float, angle: float, mat=None):
    '''
    :param Ny: Number of cells in y direction
    :param Nx: Number of cells in x direction
    :param dx: Intercellular distance
    :param angle: Angle of plane [radians]
    :param mat: (Optional) ndarray with some predefined "terrain", onto which\
                the slope angle will be added
    :return: Ny x Nx numpy array with height values

    Generates a plane where the height difference between cell rows\
    are defined by dx and angle. That is assuming hexagonal cells with\
    pointy heads.
    '''
    vec = np.arange(Ny, dtype=np.double)
    vec = vec * np.sqrt(3) / 2 * dx * np.sin(angle)
    if mat is None:
        vec = -np.array([vec, ] * Nx).transpose() + np.amax(vec)
    else:
        assert type(mat) == np.ndarray
        vec = mat + vec[:, None]
    return vec

def calc_settling_speed(D_sg: np.ndarray, rho_a, rho_j,g,nu):
    '''

    :param D_sg: Diameter of particles. np.array(Nj)
    :param rho_a: Ambient fluid density
    :param rho_j: Particle density [kg/m^3]
    :param g: Gravitational acceleration
    :param nu: Kinematic viscosity
    :return: Settling speed of particles
    '''
    v_s = []
    for i in range(D_sg.shape[0]):
        if(D_sg[i]<= 100e-06):
            v_s.append(1/18*(rho_j[i]/rho_a-1)*g*D_sg[i]**2/nu)
        elif(D_sg[i]<= 1000e-06):
            v_s.append(10*nu/D_sg[i]*(np.sqrt(1+0.01*((rho_j[i]/rho_a-1)*g*D_sg[i]**3)/nu**2)-1))
        else:
            v_s.append(1.1*np.sqrt((rho_j[i]/rho_a-1)*g*D_sg[i]))
    v_s = np.asarray(v_s)
    return v_s




def calc_g_prime(Nj, Q_cj, rho_j, rho_a, g = 9.81): 
    '''
    This function calculates the reduced gravity $g'$. Returns reduced gravity matrix numpy.ndarray(Ny,Nx).
    
    :type Nj: int
    :param Nj: Number of sediment layers used in simulation.
    
    :type Q_cj: numpy.ndarray((Ny,Nx,Nj))
    :param Q_cj: Concentration of j'th sediment layer. Fraction, i.e.\
    Q_cj :math \in \[0,1\]
    
    :type rho_j: numpy.ndarray((Nj))
    :param rho_j: $rho_j[j] =$ Density of j'th sediment layer
    
    :type rho_a: float
    :param rho_a: Density of ambient fluid. rho_a > 0.
    
    :type g: float
    :param g: Gravitational acceleration.
    
    Example:
    
    >>> import numpy as np
    >>> Nj = 1
    >>> Nx=Ny=3
    >>> Q_cj = np.ones((Ny,Nx,Nj))
    >>> rho_j = np.array([1])
    >>> rho_a = 0.5
    >>> calc_g_prime(Nj,Q_cj,rho_j,rho_a)
    ... array([[ 9.81,  9.81,  9.81],
    ...        [ 9.81,  9.81,  9.81],
    ...        [ 9.81,  9.81,  9.81]])
    
    '''
    sum = 0
    try:
        for j in range(Nj):
            sum += Q_cj[:,:,j]*(rho_j[j]-rho_a)/rho_a
        return g*sum
    except:
        print("Error: Could not calculate reduced gravity!")


