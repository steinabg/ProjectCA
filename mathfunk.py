import numpy as np
import warnings
import os
from sys import stdout
from functools import reduce

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


def generate_salles2_bathymetry(dx, alpha=np.pi / 2, Ny=200, Nx=200, c_p=80, c_amp=3, c_width=7, c_depth=2.75,
                                endspace=10):
    # Nx = Ny = 200
    X = np.zeros((Ny, Nx))
    dy = np.sqrt(3) / 2 * dx
    y = np.linspace(0, Ny - 1, num=int(Ny))

    y_offset = np.round(Nx / 13.333)

    if c_amp is None:
        c_amp = np.round(Nx / 6.667)
    c_amp = c_amp / dx

    indicesY = np.arange(0, Ny)
    indicesX = int(np.round(Nx / 2) + y_offset) + np.round(
        c_amp * np.sin(2 * np.pi / (c_p / dy) * (y - alpha)))

    if c_width is None:
        c_width = int(np.round(Nx / 6.667))
    c_width = int(np.round(c_width / dx))  # Convert from meter to cells
    if c_depth is None:
        c_depth = np.minimum(int(np.round(Nx / 66.666)), 3)

    x = np.linspace(np.pi, 2 * np.pi, c_width)
    # Create channel cross section
    cross_sectionY = c_depth * np.sin(x)


    limy = int(np.round(Ny - endspace / dy))
    for i in range(limy):
        for j in range(c_width):
            X[int(indicesY[i]), int(indicesX[i] - c_width + j)] = cross_sectionY[j]
    lowest = np.min(X)
    for i in range(limy, Ny):
        X[int(indicesY[i]), :] = lowest

    return X


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

    return X.transpose(), cross_sectionY

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
    vec = vec * np.sqrt(3) / 2 * dx * np.tan(angle)
    if mat is None:
        vec = -np.array([vec, ] * Nx).transpose() + np.amax(vec)
    else:
        assert type(mat) == np.ndarray
        vec = mat + vec[:, None]
    return vec

def gen_sloped_plane_batch(Ny: int, Nx:int, dx: float, angles: list):
    '''

    :param Ny:
    :param Nx:
    :param dx:
    :param angles:
    :return: List of sloped planes (Ny x Nx matrix) with angles as specified by the list of angles
    '''
    result = []
    for angle in angles:
        result.append(gen_sloped_plane(Ny, Nx, dx, angle))
    return result

def two_norm(a: np.ndarray, b: np.ndarray):
    return np.sqrt(np.sum(np.power(a-b, 2)))

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def find_index_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calc_settling_speed(D_sg: np.ndarray, rho_a, rho_j,g,nu, method='VanRijn'):
    '''

    :param D_sg: Diameter of particles. np.array(Nj)
    :param rho_a: Ambient fluid density
    :param rho_j: Particle density [kg/m^3]
    :param g: Gravitational acceleration
    :param nu: Kinematic viscosity
    :return: Settling speed of particles
    '''
    v_s = []
    method = method.lower()
    if method == 'vanrijn':
        for i in range(D_sg.shape[0]):
            if(D_sg[i]<= 100e-06):
                v_s.append(1/18*(rho_j[i]/rho_a-1)*g*D_sg[i]**2/nu)
            elif(D_sg[i]<= 1000e-06):
                v_s.append(10*nu/D_sg[i]*(np.sqrt(1+0.01*((rho_j[i]/rho_a-1)*g*D_sg[i]**3)/nu**2)-1))
            else:
                v_s.append(1.1*np.sqrt((rho_j[i]/rho_a-1)*g*D_sg[i]))
    elif method == 'soulsby':
        for i in range(D_sg.shape[0]):
            dimless_d = (g * (rho_j[i]/rho_a - 1)/ nu**2) ** (1/3) * D_sg[i]
            v_s.append(nu/D_sg[i] * ( np.sqrt(10.36**2 + 1.049 * dimless_d ** 3) - 10.36))
    else:
        raise KeyError('Method {0} is not a valid option. Options are vanrijn, soulsby'.format(method))
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
    ... np.ndarray([[ 9.81,  9.81,  9.81],
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

def is_square(apositiveint):
    """
    This function checks whether an int is a square number.
    :param apositiveint: integer
    :return: boolean value
    """
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

def factors(n):
    '''Returns a list of factors, that when multiplied gives n.'''
    s = set(reduce(list.__add__,
                   ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))
    l = []
    for f in s:
        l.append(tuple([f, int(n / f)]))

    return l


def largest_factor(n):
    '''Finds the two factors closest to the sqrt(n) that when multiplied gives n.'''
    f = factors(n)
    s = np.sqrt(n)
    final = tuple([int(n/2),2])
    for t in f:
        if t[0] >= s and t[1] <= s and t[0] < final[0] and t[1] > final[1]:
            final = tuple([int(t[0]),int(t[1])])
    if final[0] < final[1]:
        final = tuple([final[1],final[0]])
    return final


def dump(obj, nested_level=0, output=stdout):
    """ This function creates a formatted output for a dict."""
    spacing = '   '
    if isinstance(obj, dict):
        print('%s{' % ((nested_level) * spacing), file=output)
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print('%s%s:' % ((nested_level + 1) * spacing, k), file=output)
                dump(v, nested_level + 1, output)
            else:
                print('%s%s: %s' % ((nested_level + 1) * spacing, k, v), file=output)
        print('%s}' % (nested_level * spacing), file=output)
    elif isinstance(obj, list):
        print('%s[' % ((nested_level) * spacing), file=output)
        for v in obj:
            if hasattr(v, '__iter__'):
                dump(v, nested_level + 1, output)
            else:
                print('%s%s' % ((nested_level + 1) * spacing, v), file=output)
        print('%s]' % ((nested_level) * spacing), file=output)
    else:
        print('%s%s' % (nested_level * spacing, obj), file=output)