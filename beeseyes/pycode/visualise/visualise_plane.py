import numpy as np
from optics.geom_helpers import tuple3_to_np

def visualise_plane(ax3d, plane, color):
    # visulaise the plane in 3d
    NU = 10
    NV = 10
    [pu,pv] = np.meshgrid(np.linspace(0, 1, NU),np.linspace(0, 1, NV))
    pu = pu.ravel()
    pv = pv.ravel()
    U = tuple3_to_np(plane.U)
    V = tuple3_to_np(plane.V)
    C0 = tuple3_to_np(plane.C0)
    ax3d.scatter(
     pu*U[:,0]+pv*V[:,0] + C0[:,0],
     pu*U[:,1]+pv*V[:,1] + C0[:,1],
     pu*U[:,2]+pv*V[:,2] + C0[:,2],
     marker='.', color=color)
