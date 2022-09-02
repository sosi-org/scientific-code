
import numpy as np
import matplotlib.pyplot as plt

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[:,0]
    y       = xyz[:,1]
    z       = xyz[:,2]
    r       =  np.sqrt(x*x + y*y + z*z)
    #theta   =  acos(z/r)*180/ pi #to degrees
    #phi     =  atan2(y,x)*180/ pi
    theta   =  np.arccos(z/r)  # radians
    phi     =  np.arctan2(y,x)
    print('theta' , theta.shape)
    print('phi' ,phi.shape)
    return [r, theta ,phi]

def transform_thetapi(xyz):
    [_, theta, phi] = asSpherical(xyz)
    # (phi,theta) = (theta,phi) # swap (theta, phi)
    thetaphi = np.concatenate(((theta)[:,None], phi[:,None]), axis=1)
    return thetaphi, (r'$\theta$', r'$\phi$')


# I think: It is visualising on a (2d) sphere using polar coords
def visualise_map_spherical_to_planar(center_points, uv_rgba=None, transform2planar=transform_thetapi):
    #traansform_planar, map3to2
    #traansform_planar = center_points
    planar2d, axeslabels = transform2planar(center_points)
    print('planar2d', planar2d)

    fig = plt.figure()
    ax2d = fig.add_axes([0,0,1,1])
    def transf2d(p2d) -> tuple[float,float]:
        return p2d[:,0] * 180/np.pi * np.cos(p2d[:,1]), p2d[:,1] * 180/np.pi

    points = transf2d(planar2d)
    ax2d.scatter(*points, facecolors=uv_rgba, marker='.')
    #ax2d.scatter(*transf2d(planar2d), marker='.')

    xr = [-np.pi, np.pi]
    yr = [-np.pi, np.pi]
    xa = np.arange(xr[0], xr[1], np.pi/5 - 0.0001)
    ya = np.arange(xr[0], xr[1], np.pi/5 - 0.0001)

    for i in range(len(ya)):
        y0 = ya[i]
        c = np.arange(xr[0], xr[1], 0.01)
        horiz = np.concatenate((c[:,None], (c*0 + y0)[:,None]), axis=1)
        ax2d.plot(*transf2d(horiz),'--', linewidth=0.2, color='k')
        # grid_row =

    for i in range(len(xa)):
        x0 = xa[i]
        c = np.arange(yr[0], yr[1], 0.01)
        horiz = np.concatenate(((c*0+ x0)[:,None], c[:,None]), axis=1)
        ax2d.plot(*transf2d(horiz),'--', linewidth=0.2, color='k')

    '''
    ax2d.axhline(y=0, color='k')
    ax2d.axvline(x=0, color='k')
    '''

    ax2d.spines['left'].set_position('zero')
    #ax2d.spines['right'].set_color('none')
    ax2d.yaxis.tick_left()
    ax2d.spines['bottom'].set_position('zero')
    #ax2d.spines['top'].set_color('none')
    ax2d.xaxis.tick_bottom()

    #ax2d.axis([0,6,0,6])

    #ax2d.set_xlim(*array_minmax(points[0]))
    #ax2d.set_ylim(*array_minmax(points[1]))
    ax2d.set_xlim(-180, +180)
    ax2d.set_ylim(-180, +180)
    ax2d.set_xlabel(axeslabels[0], fontsize=15, verticalalignment='bottom', x=0.1)
    ax2d.set_ylabel(axeslabels[1], fontsize=15, verticalalignment='bottom', y=0.9)
    # ax2d.tight_layout() # Doesn't work
    return ax2d
