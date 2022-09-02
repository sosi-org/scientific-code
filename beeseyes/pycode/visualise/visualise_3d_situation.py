import numpy as np

import matplotlib.pyplot as plt

from .ax3dc import ax3dCreate

from sceneconfig import PRODUCTION_FIGS

from .visualise_quiver import visualise_quiver
from .visualise_plane import visualise_plane

def set_axis_label(ax3d, xyz_index, label_text):
  if xyz_index == 0:
    ax3d.set_xlabel(label_text)
  elif xyz_index == 1:
    ax3d.set_ylabel(label_text)
  elif xyz_index == 2:
    ax3d.set_zlabel(label_text)
  else:
    raise

def visualise_3d_situation(corner_points, normals_at_corners, ommatidia_few_corners, ommatidia_few_corners_normals, center_points, normals_at_center_points, beeHead, planes):
    '''
    3D visualise texture plane (grid)
    and eye
    '''
    p0 = beeHead.pos
    print('p0.shape',p0.shape)

    def rot(vectos):
        return np.dot(beeHead.R, vectos.T).T

    # Eye
    #ax3d = ax3dCreate(SZ=28.8)
    ax3d = ax3dCreate(SZ=None)
    # rename `corner_points` to `*corners`
    visualise_quiver(ax3d, rot(corner_points) + p0, rot(normals_at_corners), color='r')  # corners
    visualise_quiver(ax3d, rot(center_points) + p0, rot(normals_at_center_points), color='b') # centers

    general_direction = np.mean(center_points, 0)[None,:]
    visualise_quiver(ax3d, rot(general_direction) + p0, rot(general_direction), color='k') # centers

    # All planes
    if True:
      for i in range(len(planes)):
        visualise_plane(ax3d, planes[i], color='tab:pink')


    # adding Visualisation of few points in 3D
    assert ommatidia_few_corners_normals.shape == ommatidia_few_corners.shape
    visualise_quiver(ax3d, rot(ommatidia_few_corners) + p0, ommatidia_few_corners_normals * 0.01, color='m')
    #plt.show()
    #exit()
    return ax3d




def visualise_3d_situation_eye(selected_center_points, regions_rgb, beeHead, title, ax3d_reuse=None, set_lims=True, flip_axes=False):
    assert selected_center_points.shape[0] == regions_rgb.shape[0]
    assert selected_center_points.shape[1] == 3
    assert regions_rgb.shape[1] == 4
    # todo: rename `regions_rgb` to `regions_rgba`

    # shed the NaNs
    isnan1 = np.isnan(selected_center_points[:,0])
    isnan2 = np.isnan(regions_rgb[:,0])
    isnonan = np.logical_not(np.logical_or(isnan1, isnan2))


    selected_center_points = selected_center_points[isnonan, :]
    regions_rgb = regions_rgb[isnonan, :]
    assert selected_center_points.shape[0] > 0, 'at least one non-NaN point'


    if beeHead is not None:
      p0 = beeHead.pos

      def rot(vectors):
          return np.dot(beeHead.R, vectors.T).T

      X = rot(selected_center_points) + p0

    else:
      X = selected_center_points

    # A permutation: Which actual (in coords) should dbe shown on axes3d's firast axis?
    _X, _Y, _Z = 0, 1, 2
    if flip_axes:
        assert ax3d_reuse is None
        _X, _Y, _Z = 0, 2, 1 # Show your Ys oon axes3d Z, Zs on Y, Xs on X
    ax3d = None
    if ax3d_reuse is None:
        #ax3d = ax3dCreate(SZ=28.8)   # redundant setlim
        ax3d = ax3dCreate(SZ=None)
    else:
        ax3d = ax3d_reuse

    #print()
    #print(X.shape)
    #print(regions_rgb.shape)
    #print('^^')
    h = \
    ax3d.scatter(X[:,_X], X[:,_Y], X[:,_Z], facecolors=regions_rgb,
                 #marker='.',
                 #s=(30.0*3) ** 2
                 #s=None
                 #s=(2.0) ** 2
                 s=(4.0) ** 2
                 )
    #print('ok')

    if PRODUCTION_FIGS:
      # ax3d.set_aspect("equal")
      #h.s=20
      #h.markersize = 20
      #h.s=100.0
      pass

    if set_lims:
      assert  ax3d_reuse is None, 'must set `set_lims=False` when `ax3d_reuse` is used'

    if ax3d_reuse is None:
        if set_lims:
            ax3d.set_xlim(*array_minmax(X[:,_X]))
            ax3d.set_ylim(*array_minmax(X[:,_Y]))
            ax3d.set_zlim(*array_minmax(X[:,_Z]))
        set_axis_label(ax3d, _X, 'X')
        set_axis_label(ax3d, _Y, 'Y')
        set_axis_label(ax3d, _Z, 'Z')
        ax3d.set_title(title)

    return ax3d
