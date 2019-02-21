

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('Qt5Agg')
import warnings

def generate_rupert_inlet_bathymetry(Ny=200,Nx=200,channel_amplitude=None, channel_width=None, channeldepth=None):
    # Nx = Ny = 200
    X = np.zeros((Ny,Nx))

    y_offset = np.round(Ny/13.333)

    if channel_amplitude is None:
        channel_amplitude = np.round(Ny/6.667)
    
    indicesX = np.arange(np.round(Nx/3),Nx+1)
    indicesY = int(np.round(Ny/2)+y_offset) + np.round(channel_amplitude * np.sin(np.linspace(np.pi,5*np.pi/2,num=int(Nx-indicesX[0]))))
    
    if channel_width is None:
        channel_width = int(np.round(Ny/6.667))
    if channeldepth is None:
        channeldepth = int(np.round(Ny/66.666))
    cross_sectionY = channeldepth * np.sin(np.linspace(np.pi,2*np.pi,channel_width))
    for i in range(len(indicesY)):
        for j in range(channel_width):
            X[int(indicesY[i]-channel_width+j), int(indicesX[i])] = cross_sectionY[j]

    for j in range(channel_width):
        X[int(np.round(Ny/2)+y_offset-channel_width+j), np.arange(1,np.round(Nx/3)+1, dtype=np.int)] = cross_sectionY[j]

    cross_sectionX = np.tan(np.deg2rad(-5))*np.arange(1,Nx+1)

    for i in range(Nx):
        X[:,i] = X[:,i] + cross_sectionX[i]

    X += 30
    x = np.arange(channel_width)
    angle = [np.rad2deg(np.arctan2(cross_sectionY[i], x[i])) for i in range(1, len(cross_sectionY))]
    angle = np.abs(angle)
    if np.max(angle) > 1:
        warnings.warn("Bathymetry: Channel cross-section steepness exceeds angle of repose!"
                      " This will cause an avalanche.")
    return X, cross_sectionY, channel_width

X, cross_sectionY, channel_width = generate_rupert_inlet_bathymetry(Ny=50,Nx=50)
Y, YY = np.meshgrid(np.arange(X.shape[1]),np.arange(X.shape[0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.plot_surface(Y,YY,X)
plt.show()

fig = plt.figure()
x = np.arange(channel_width)
plt.plot(x,cross_sectionY,'-*')
plt.show()

angle = [np.rad2deg(np.arctan2(cross_sectionY[i],x[i])) for i in range(1,len(cross_sectionY))]
angle