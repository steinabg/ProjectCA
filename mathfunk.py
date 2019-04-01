import numpy as np
import warnings

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

def average_speed_hexagon(U_k): # Testet: 18.10.18
    '''
    Calculates the length of the velocity vector in a hexagonal cell. 
    
    :type U_k: (Ny x Nx x 6) array
    :param U_k: Matrix containing velocities in all six directions.\
    First slice of U_k is the velocity towards the NW-edge of the hexagon.\
    The following slices are 
    
    Example:


    >>> import numpy as np
    >>> Ny=Nx=3
    >>> U_k = np.zeros((Ny,Nx,6))
    >>> U_k[1,1,0] = 2
    >>> U_k[1,1,1] = 1
    >>> U_k[1,1,2] = 1
    >>> average_speed_hexagon(U_k)
    ... 3
    ... array([[ 0.        ,  0.        ,  0.        ],
    ...        [ 0.        ,  2.64575131,  0.        ],
    ...        [ 0.        ,  0.        ,  0.        ]])

    '''
#     print( U_k[0,:,0].size )
#     Ny= Nx = U_k[0,:,0].size
#     v = np.zeros((Ny,Nx,3))
#     v[:,:,0] = U_k[:,:,0] - U_k[:,:,3]
#     v[:,:,1] = U_k[:,:,1] - U_k[:,:,4]
#     v[:,:,2] = U_k[:,:,2] - U_k[:,:,5]

    # mean = np.sum(U_k,axis=2)/6 # Likt i alle retninger
    num = np.count_nonzero(U_k,axis=2)
    mean = np.nan_to_num(np.sum(U_k, axis=2) / num)  # Fordelt på antall celler hvor strømmen går
    # mean = np.sqrt( (0.5*(v[:,:,1]-v[:,:,0])+v[:,:,2])**2 + 3/4*(v[:,:,1]+v[:,:,0])**2 )

    return mean
        
def calc_rho_c(Nj, Q_cj, rho_j, rho_a): # out: current density rho_c matrix (all cells)
    '''
    This function calculates the current density rho_c, for all Ny x Nx cells.\
    If a cell has no sediment current, the current density is just the ambient\
    current. Otherwise, the current density is a weighted average of the\
    densities present in the cell. The weights are the volume concentration\
    of each sediment.
    
    :type Nj: int
    :param Nj: Number of sediment layers used in simulation.
    
    :type Q_cj: numpy.ndarray((Ny,Nx,Nj))
    :param Q_cj: Concentration of j'th sediment layer. Unit = 1. Fraction, i.e.\
    Q_cj :math \in \[0,1\]
    
    :type rho_j: numpy.ndarray((Nj))
    :param rho_j: $rho_j[j] =$ Density of j'th sediment layer. Unit = kg/m^3
    
    :type rho_a: float
    :param rho_a: Density of ambient fluid. Unit = kg/m^3
    
    Examples:
    
    >>> import numpy as np
    >>> Nj =2
    >>> Nx=Ny=3
    >>> Q_cj = np.zeros((Ny,Nx,Nj))
    >>> Q_cj[1,1,0] = 1 # 100 % concentration of sediment no. 0.
    >>> Q_cj[1,0,1] = 1 # 100 % concentration of sediment no. 1.
    >>> Q_cj[1,2,:] = 0.5 # 50/50 mix of sediment no. 0 and 1.
    >>> Q_cj[0,0,:] = 1/3 # Equal mix between two sediments and ambient density.
    >>> rho_j = np.array([10,5])
    >>> rho_a = 3
    >>> calc_rho_c(Nj,Q_cj,rho_j,rho_a)
    ... array([[  6. ,   3. ,   3. ],
    ...        [  5. ,  10. ,   7.5],
    ...        [  3. ,   3. ,   3. ]])
    
    '''
    sum = 0
    for j in range(Nj):
        sum += rho_j[j]*Q_cj[:,:,j]
    return rho_a*(1-np.sum(Q_cj,axis=2))+sum

def calc_potEnergy(Q_th, g_prime, rho_c, A): # out: 
    '''
    This fuction calculates and returns the potential energy of the \
    turbidity current for cells in an (Nx by Ny) grid. Each cell has\
    an area A, height Q_th, and some density rho_c.
    
    :type Q_th: numpy.ndarray(Ny,Nx)
    :param Q_th: Turbidity current thickness. Unit = m
    
    :type g_prime: numpy.ndarray(Ny,Nx)
    :param g_prime: Reduced gravity for all cells. Unit = m/s^2
    
    :type rho_c: numpy.ndarray(Ny,Nx)
    :param rho_c: Entry [i,j] is the current density in cell [i,j]. Unit = kg/m^3
    
    :type A: float
    :param A: Area of hexagonal grid cell. Unit = m^2
    
    '''
    return rho_c*g_prime*A/2*(Q_th)**2

def calc_hexagon_area(apothem):
    return 2*np.sqrt(3)*(apothem/2)**2 # Area of hexagon = 2sqrt(3)*apothem

def calc_neighborDiff(input1, input2):
    '''
    This function calculates the difference between the value 'input' in\
    a center cell and the value of 'input' for the neighbors.\
    The result is stored in 'result' and result[i,j,0] is the difference\
    between input[i,j] and the value of input in the NW neighbor.
    The slices of 'result' stores the difference between a cell and its\
    NW, NE, E, SE, SW, W neighbors in that order:

    result[:,:,1] : NE neighbor\
    result[:,:,2] : E neighbor\
    etc...

    '''
    size = input1.shape
    result = np.zeros((size[0]-2,size[1]-2,6))
    result[:,:,0] = input1[1:-1,1:-1] - input2[0:size[1]-2, 1:size[0]-1]
    result[:,:,1] = input1[1:-1,1:-1] - input2[0:size[1]-2, 2:size[0]  ]
    result[:,:,2] = input1[1:-1,1:-1] - input2[1:size[1]-1, 2:size[0]  ]
    result[:,:,3] = input1[1:-1,1:-1] - input2[2:size[1]  , 1:size[0]-1]
    result[:,:,4] = input1[1:-1,1:-1] - input2[2:size[1]  , 0:size[0]-2]
    result[:,:,5] = input1[1:-1,1:-1] - input2[1:size[1]-1, 0:size[0]-2]
    return result



