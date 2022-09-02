from typing import Tuple
from typing import Tuple

import numpy as np
# todo: remove
import matplotlib.pyplot as plt

import math

from cte import _PIXELS, _CM
from cte import DIM3
from cte import UNIT_LEN_CM, UNIT_LEN_MM, UNIT_LEN_CM
from cte import CAST_CLIP_FULL, CAST_CLIP_NONE, CAST_CLIP_FRONT

import my_folders
import image_loader



from bee_eye_data import ommatidia_polygons, ommatidia_polygons_fast_representation
# from bee_eye_data import ax3dCreate, visualise_quiver
from bee_eye_data import make_midpoints, make_deviations, my_index


from visualise.visualise_3d_situation import visualise_3d_situation_eye
import sceneconfig


def focus_3d(ax3d, focus, zoom_size):
  ax3d.set_zlim(focus[2]-zoom_size, focus[2]+zoom_size)
  ax3d.set_xlim(focus[0]-zoom_size, focus[0]+zoom_size)
  ax3d.set_ylim(focus[1]-zoom_size, focus[1]+zoom_size)
  ax3d.set_xlabel('x')
  ax3d.set_ylabel('y')
  ax3d.set_zlabel('z')



"""
Coordinate system:
  wall: right=(1,0,0) , up=(0,1,0)
  towards the wall: (0,0,1)
"""


from optics.plane import Plane
from optics.bee_look import BeeHead
# import old_demo # old_demo moved out of this file


def make_whichfacets(sv_vertices, sv_regions, areas, SD_THRESHOLD, AREA_THRESHOLD, MAX_SIDES):
    ommatidia_polygons1, regions_side_count = \
       ommatidia_polygons_fast_representation(sv_vertices, sv_regions, maxsides=MAX_SIDES, default=np.NaN)

    #print('ommatidia_polygons1\n', ommatidia_polygons1)
    midpoints = make_midpoints(ommatidia_polygons1, regions_side_count)
    #print('midpoints', midpoints.shape)
    #print('midpoints\n', midpoints)
    polyg_d = make_deviations(ommatidia_polygons1, midpoints, regions_side_count)
    #print('polyg_d', polyg_d.shape) # (2, 14, 3)
    #print('polyg_d', polyg_d)
    #print('polyg_d^2', polyg_d * polyg_d)
    counts = regions_side_count.astype(polyg_d.dtype)[:,None]
    #print('counts.shape', counts.shape)
    #print('counts', counts)
    # nanxx = np.nansum
    sum_rows = np.nansum(polyg_d * polyg_d, axis=1) / counts  # (n,)
    var_s = np.nansum(sum_rows, axis=1)
    #print('>>> var_s', var_s.shape)
    #print('=========')
    sd_s = np.power(var_s, 0.5)
    #print('sd_s', var_s.shape)
    print(sd_s)

    '''
    plt.figure()
    plt.title('STD of ommatidia corners')
    plt.hist(sd_s, bins=np.arange(0, 1.1, 0.01))
    plt.yscale('log')
    plt.show()
    '''
    plt.figure()
    plt.title('STD of ommatidia corners')
    hist1, bin_edges = np.histogram(sd_s, bins=np.arange(0, 1.1, 0.01))
    hist1cumsum = np.cumsum(hist1)
    plt.plot(bin_edges[:-1], hist1cumsum)
    plt.plot(bin_edges[:-1], hist1)
    plt.yscale('log')
    #plt.show()

    # areas
    '''
    plt.figure()
    plt.title('Areas of ommatidia_polygons (from SphericalVoronoi)')
    plt.hist(areas, bins=np.arange(0,0.2,0.01 / 10.0))
    plt.yscale('log')
    plt.show()
    '''

    #AREA_THRESHOLD = 0.050
    #AREA_THRESHOLD = 0.01/2/4
    w_areas = areas < AREA_THRESHOLD
    sv_regions_sel = my_index(sv_regions, w_areas)   # sv_regions[w_areas]
    '''
    unionvi = np.array(list(set().union(*sv_regions_sel)))
    sv.vertices = sv.vertices[unionvi, :]
    normals_at_corners = normals_at_corners[unionvi, :]
    '''

    # SD_THRESHOLD = 0.2
    which_facets_sd = sd_s < SD_THRESHOLD
    which_facets = np.logical_and(which_facets_sd, w_areas)

    which_indices = np.arange(0,which_facets.shape[0])[which_facets]
    print('which_indices=', which_indices)


    return which_facets

def concat_lists(sv_regions_sel):
    c = []
    for i in range(len(sv_regions_sel)):
        c.extend(sv_regions_sel[i])
    return c


#def select_regions(sv_regions, areas, corner_points, MAX_SIDES):
def select_regions(sv_regions, which_facets):

    #which_facets = make_whichfacets(corner_points, sv_regions, areas, SD_THRESHOLD=SD_THRESHOLD, AREA_THRESHOLD=AREA_THRESHOLD, MAX_SIDES=MAX_SIDES)

    n = which_facets.shape[0]
    assert which_facets.shape[0] == len(sv_regions)
    selected_regions = []
    for i in range(n):
        if which_facets[i]:
            selected_regions.append(sv_regions[i])

    #selected_center_points = select_centers(which_facets, center_points)
    return selected_regions

# dont use this
def select_centers(which_facets, center_points):

    # which_facets = make_whichfacets(corner_points, sv_regions, areas, SD_THRESHOLD=SD_THRESHOLD, AREA_THRESHOLD=AREA_THRESHOLD, MAX_SIDES=MAX_SIDES)

    n = which_facets.shape[0]
    selected_center_points_list = []
    for i in range(n):
        if which_facets[i]:
            selected_center_points_list.append(center_points[i])

    # todo: faster
    m = len(selected_center_points_list)
    ndims = center_points.shape[1]
    selected_center_points = np.zeros((m, ndims), dtype=center_points.dtype)
    for i in range(m):
        selected_center_points[i,:] = center_points[i]

    return selected_center_points


# pick a few only-for debugging purpose
DEBUG_FEW = True

def pick_subset_of_vectors(sv_regions, areas, corner_points, normals_at_corners):
    # Future plan:
    # 1. based on `which_facets`, choose a subset of `sv_regions`.
    # 2. Then do the `ommatidia_polygons_fast_representation()` for those regions. (pads with NaN)
    # 3. (optional: not now) Avoid re-computing corners that are not used there? How? no. Just remove them corners not used? Then it will need re-indexing of the "corner" vertices.
    # 4. Flatten (those padded with NaN).
    # 5. Remove NaNs => re-index again?! (ALl can be done here)
    # Ok foret about this faast thing. It is in-fact incomplete.

    # see temp6.py
    #  selects some faces and returns them as an array of (point,normal)s
    # Two types of selection:
    # 1. select a few for debug (not in production)
    # 2. select aa subset because of invalid regions

    MAX_SIDES = 14 # 6 #14 # 6
    if DEBUG_FEW:
        few_face_indices = [0,1]
        #print('**', len(sv_regions))
        #few_face_indices = [300, 301] #[500, 501]
        few_regions = [sv_regions[face] for face in few_face_indices]

        # original points. Limit only for rdebugging purposes
        ommatidia_selected_polygons1, regions_side_count = \
            ommatidia_polygons_fast_representation(corner_points, few_regions, maxsides=MAX_SIDES)
        ommatidia_polygons = ommatidia_selected_polygons1
    else:
        ommatidia_originl_polygons1, regions_side_count = \
            ommatidia_polygons_fast_representation(corner_points, sv_regions, maxsides=MAX_SIDES)
        ommatidia_polygons = ommatidia_originl_polygons1

    print('\n'*24 , '-----'*10)
    # sv_regions < -> few_regions
    # which_facets = make_whichfacets(corner_points, sv_regions, areas, SD_THRESHOLD=SD_THRESHOLD, AREA_THRESHOLD=AREA_THRESHOLD, MAX_SIDES=MAX_SIDES)

    which_corners_in_few_faces = np.array([*sv_regions[0], *sv_regions[1]])

    ommatidia_few_corners = ommatidia_selected_polygons1.reshape(-1,3)
    ommatidia_few_corners = ommatidia_few_corners[np.logical_not(np.isnan(ommatidia_few_corners[:,0])), :]
    assert which_corners_in_few_faces.shape[0] == ommatidia_few_corners.shape[0]
    ommatidia_few_corners_normals = normals_at_corners[which_corners_in_few_faces, :]

    # pair: which_corners_in_few_faces, ommatidia_few_corners_normals
    #  where:
    assert which_corners_in_few_faces.shape[0] == ommatidia_few_corners.shape[0]
    assert which_corners_in_few_faces.shape[0] == ommatidia_few_corners_normals.shape[0]

    #return ommatidia_unique_corners, ommatidia_few_corners_normals
    return ommatidia_few_corners, ommatidia_few_corners_normals

def aaaaa():
    #ommatidia_selected_polygons1, regions_side_count = \
    #   ommatidia_polygons()
    # (3250, MAX_SIDES, 3)

    # rename: corner_points -> corner_vertices -> corner_points
    corner_points, sv_regions, normals_at_center_points, normals_at_corners, center_points, areas = ommatidia_polygons()
    assert corner_points.shape == normals_at_corners.shape

    print('\n'*24 , '-----'*10)

    ommatidia_few_corners, ommatidia_few_corners_normals = pick_subset_of_vectors(sv_regions, areas, corner_points, normals_at_corners)

    # (6496, 3) (6496, 3)
    print(corner_points.shape, normals_at_corners.shape)


    which_facets = make_whichfacets(corner_points, sv_regions, areas, SD_THRESHOLD=sceneconfig.SD_THRESHOLD, AREA_THRESHOLD=sceneconfig.AREA_THRESHOLD, MAX_SIDES=14)

    #selected_regions = select_regions(sv_regions, areas, corner_points, MAX_SIDES=14)
    selected_regions =  select_regions(sv_regions, which_facets)
    selected_center_points = select_centers(which_facets, center_points)
    # `selected_regions` is linked to [indices of] `(corner_points, normals_at_corners)`

    all_centers = center_points
    all_normals_at_center_points = normals_at_center_points

    return (corner_points, normals_at_corners), (center_points, normals_at_center_points), (ommatidia_few_corners_normals, ommatidia_few_corners), selected_regions, selected_center_points, which_facets, (all_centers,all_normals_at_center_points)


def array_minmax(x):
    mn = np.min(x)
    mx = np.max(x)
    md = (mn + mx)/2.0
    (mn,mx) = ((mn-md)*1.2+md, (mx-md)*1.2+md)
    return (mn, mx)



def visualise_uv(u,v, u_few, v_few, texture, uv_rgba=None, title=None, fig=None):
    # (u,v) visualisation on plane (pixels)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    tt = texture
    # tt = np.transpose(texture, axes=(1,0,2))
    plt.imshow(tt, extent=(0.0,1.0,0.0,1.0), alpha=0.6) #, origin='lower')
    #plt.plot(u, v, '.', facecolors=uv_rgba)
    plt.scatter(v, 1-u, marker='.', facecolors=uv_rgba)
    if (u_few is not None) and (v_few is not None):
       plt.plot(v_few, 1-u_few, 'o', color='r')
    #plt.plot(u6,v6, 'r.')
    plt.xlabel('u')
    plt.ylabel('v')
    '''
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    if title is not None:
        ax.set_title(title)
    '''

    # https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
    return ax

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

# visualise_uv_map

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

import sampling

def histogram_of_sides(regions):
   sides_l = []
   for region in regions:
      n = len(region)
      sides_l.append(n)

   plt.figure()
   plt.title('Number of sides: Delanuey')
   plt.hist(np.array(sides_l), bins=np.arange(2-1,11+1, 0.2))
   plt.yscale('log')

'''histogram etc'''
def trajectory_stats(bee_path):

    print(array_minmax(bee_path[:,0]))
    print(array_minmax(bee_path[:,1]))
    print(array_minmax(bee_path[:,2]))

    plt.figure()
    ax2d = plt.figure().add_subplot(111)
    plt.title('Bee locations')
    for dim in [0,1,2]:
       vals = bee_path[:, dim]
       hist1, bin_edges = np.histogram(vals, bins=np.arange(-500, 1500, 5.0))
       ph, = ax2d.plot((bin_edges[:-1] + bin_edges[1:])/2.0, hist1)
       ph.set_label(['X (RWxSmoothed)','Y (RWySmoothed)','Z (RWzSmoothed)'][dim])
    ax2d.legend()
    plt.yscale('log')
    ax2d.set_ylabel('Freq.')

    # plt.show()

    '''
    (7.079130000000021, 573.19317)
    (42.99566999999999, 371.98803)
    (-21.25277299999999, 272.127343)
    '''


# move thisd into visualise_3d_situation_eye
def rgb_to_rgba(regions_rgb):
    one = np.ones((regions_rgb.shape[0],1), dtype=float)
    regions_rgba = np.concatenate( (regions_rgb / 255.0, one), axis=1)

    _ALPHA = 3
    nans = np.isnan(regions_rgb[:,0])
    regions_rgba[nans, 0:2] = 0.0
    regions_rgba[:, _ALPHA] = 1.0
    regions_rgba[nans, _ALPHA] = 0.0
    #regions_rgba = alpha1(regions_rgba)
    return regions_rgba

def alpha1(rgba, alpha_value=1.0):
    ''' note: inplace '''
    _ALPHA = 3
    nans = np.isnan(rgba[:,0])
    rgba[nans, 0:2] = 0.0
    rgba[:, _ALPHA] = alpha_value
    rgba[nans, _ALPHA] = 0.0
    return rgba

def two_eyes(all_centers, all_normals_at_center_points, gap_x, left=True, right=True):
    FLIP_XZ = np.diag([-1.0,1,-1])
    def mult(a,b):
      return np.dot(a, b.T).T

    X1 = np.array([1.0, 0.0, 0.0])
    gap_d = X1[None,:] * (-gap_x)
    centers_left = mult(FLIP_XZ, all_centers) + gap_d
    all_normals_left =  mult(FLIP_XZ, all_normals_at_center_points)

    FLIP_X = np.diag([-1.0,1,1])
    centers_right = mult(FLIP_X, centers_left)
    all_normals_right =  mult(FLIP_X, all_normals_left)

    ptupl = tuple()
    ntupl = tuple()
    if left:
      ptupl = ptupl + tuple([centers_left])
      ntupl = ntupl + tuple([all_normals_left])
    if right:
      ptupl = ptupl + tuple([centers_right])
      ntupl = ntupl + tuple([all_normals_right])
    all_centers = np.concatenate(ptupl, axis=0)
    all_normals = np.concatenate(ntupl, axis=0)
    return all_centers, all_normals

from optics.get_actual_ommatidium_rays import getActualOmmatidiumRays

def project_colors(all_centers, all_normals, beeHead, plane, plane_texture, clip):

    O,D = getActualOmmatidiumRays(
      all_centers, all_normals,
      beeHead.R, beeHead.pos, beeHead.eye_size_cm)

    print('@@beeHead.eye_size_cm', beeHead.eye_size_cm)

    #for i in range(len(planes)):
    (u,v),casted_points = plane.raycastRaysOnPlane(
      O,D,
      clip=clip,
      return_casted_points=True)

    uv = np.concatenate((u[:,None], v[:,None]), axis=1)

    uv_rgb, *_ = sampling.sample_colors(uv, None, plane_texture)
    uv_rgba = rgb_to_rgba(uv_rgb)
    #uv_rgba = alpha1(uv_rgba*0)
    #_rgba = uv_rgba


    if False:
        # shed the NaNs
        isnan1 = np.isnan(uv[:,0])
        isnan2 = np.isnan(uv_rgba[:,0])
        isnonan = np.logical_not(np.logical_or(isnan1, isnan2))
        all_centers = all_centers[isnonan, :]
        O=O[isnonan, :]
        D=D[isnonan, :]
        uv_rgba = uv_rgba[isnonan, :]
        casted_points = casted_points[isnonan, :]
        assert O.shape[0] > 0, 'at least one non-NaN point'
        assert uv_rgba.shape[0] > 0, 'at least one non-NaN point'
        assert casted_points.shape[0] > 0, 'at least one non-NaN point'

    #return _cpoints, _rgba, casted_points
    #return all_centers, uv_rgba, casted_points
    return O, uv_rgba, casted_points, uv

from sceneconfig import PRODUCTION_FIGS

def anim_frame(
      textures, planes,
      bee_head_pos, bee_direction, eye_sphere_size_cm, clip,
      corner_points, normals_at_corners,
      ommatidia_few_corners, ommatidia_few_corners_normals,
      selected_regions, which_facets,
      all_pair,
      # points to project: no!
      selected_center_points, selected_center_points_normal,
      #center_points, center_points_normal,

      whether_visualise_eye_3d, whether_visualise_uv_samples, whether_visualise_uv_scatter,
      animation_fig=None,
      text_description=''
):

    #head_transformation = -M*10 *1000 # -M*10 is adhoc
    #beeHead = BeeHead(head_transformation)
    beeHead = BeeHead()

    beeHead.set_eye_position(bee_head_pos)
    beeHead.set_direction(bee_direction, eye_sphere_size_cm)

    # eye_size_cm = 1.0 * UNIT_LEN_CM

    # xx = corner_points
    # xx.shape is 6496, 3
    # Points are on a unit sphere
    # print( np.linalg.norm(xx, axis=1) )  -> [1,1,1,...]

    print('beeHead.R', beeHead.R)

    # CORNER-based
    CORNER_BASED = True

    if CORNER_BASED:
        points_to_cast = corner_points #* beeHead.eye_size_cm
        normals_to_cast = normals_at_corners
        # corners, normals_to_cast (6496, 3) (6496, 3)
        print('corners, normals_to_cast', points_to_cast.shape, normals_to_cast.shape)
    else:
        #points_to_cast = selected_center_points
        #normals_to_cast = selected_center_points_normals
        points_to_cast = center_points
        normals_to_cast = center_points_normal



    #selected_center_points_actual, __selected_centerpoints_normals_actual = getActualOmmatidiumRays(selected_center_points, selected_center_points*0,
    #  beeHead.R, beeHead.pos, beeHead.eye_size_cm)

    O,D = getActualOmmatidiumRays(points_to_cast, normals_to_cast,
      beeHead.R, beeHead.pos, beeHead.eye_size_cm)

    print('@@beeHead.eye_size_cm', beeHead.eye_size_cm)

    print('@planes[0]', planes[0])
    #for i in range(len(planes)):
    (u,v),casted_points = planes[0].raycastRaysOnPlane(
      O,D,
      clip=clip,
      return_casted_points=True)

    actual_eyepoints = O
    print('MEAN & STD', np.mean(O,axis=0), '   STD=', np.std(O,axis=0))
    assert u.shape ==(points_to_cast.shape[0],)
    uv = np.concatenate((u[:,None], v[:,None]), axis=1)

    assert len(planes) == 1, "A single plane (single side) is supported. Coming soon."
    if len(textures) == 1:
      pass
    else:
      print ("Warning: A single texture (single side) is supported. Using the first one only. Coming soon.")


    # or selected_center_points
    O_few,D_few = getActualOmmatidiumRays(
       ommatidia_few_corners, ommatidia_few_corners_normals,
       beeHead.R, beeHead.pos, beeHead.eye_size_cm)
    # A few other points too
    (u_few,v_few) = planes[0].raycastRaysOnPlane(
       O_few,D_few,
       clip=CAST_CLIP_NONE)


    if whether_visualise_uv_scatter:
        visualise_uv(u,v, u_few, v_few, textures[0], fig=None)


    # selected_center_points = select_centers(which_facets, center_points)

    #selected_uv = select_centers(which_facets, uv)

    # `selected_regions` is ..., which match `which_facets`

    #if CORNER_BASED:
    nfs = np.sum(which_facets)
    #assert selected_uv.shape[0] == nfs
    assert len(selected_regions) == nfs
    #regions_rgb = sampling.sample_colors(selected_uv, selected_regions, textures[0])
    # selected_regions describ polygons over uv (i.e. the result will be of different indexing)
    regions_rgb, uvm_debug = sampling.sample_colors(uv, selected_regions, textures[0])

    nf =len(selected_regions)
    #assert u.shape[0] == nf
    assert regions_rgb.shape[0] == nf
    assert len(selected_regions) == nf
    assert nf == nfs

    assert selected_center_points.shape[0] == nf
    assert regions_rgb.shape[0] == nf

    print('non-nan', np.sum(np.logical_not(np.isnan(regions_rgb)), axis=0)) # [14,14,14]

    regions_rgba = rgb_to_rgba(regions_rgb)

    if whether_visualise_eye_3d:
        #actual_eyepoints
        # one center_point for each region. todo: re-index center_point-s based on selected regions
        # todo: remove
        ax3 = visualise_3d_situation_eye(selected_center_points, regions_rgba, beeHead, 'eye only (1) (todo: remove)', set_lims=False)

        #plt.show()
        # (all_centers, all_normals_at_center_points)
        if all_pair is not None:
            global FOCUS

            (all_centers, all_normals_at_center_points) = all_pair
            #towards neggaaative Z
            all_centers_, all_normals_ = two_eyes(all_centers, all_normals_at_center_points, gap_x=0.4, right=False)

            print('beeHead', beeHead.pos, beeHead.eye_size_cm)
            print(beeHead.pos)
            print()
            (_cpoints, _rgba, _casted_points, _uv) = project_colors(all_centers_, all_normals_, beeHead, planes[0], textures[0], clip=CAST_CLIP_FULL)
            #(_cpoints, _rgba, _casted_points, _uv) = project_colors(points_to_cast, normals_to_cast, beeHead, planes[0], textures[0], clip=CAST_CLIP_NONE)

            ax3_ = visualise_3d_situation_eye(_cpoints, alpha1(_rgba*1),  None, 'left eye', set_lims=False, ax3d_reuse=None, flip_axes=False)
            focus_3d(ax3_, *FOCUS)
            ax3_.view_init(elev=90, azim=91)

            if PRODUCTION_FIGS:
              ax3_.dist = 10.0
              ax3_.grid(False)
              ax3_.set_xticks([])
              ax3_.set_yticks([])
              ax3_.set_zticks([])
              ax3_.set_zticks([])
              #plt.box(False)
              plt.axis('off')

            plt.show()

            def myrand(points):
              std_scalar = np.linalg.norm(np.std(points, axis=0))
              noise = np.random.normal(size=points.shape) * std_scalar
              return points + noise

            ax3 = plt.figure() .add_subplot(projection='3d', autoscale_on=True)
            _ = visualise_3d_situation_eye(_cpoints, _rgba,  None, 'eye2', set_lims=False, ax3d_reuse=ax3)

            _ = visualise_3d_situation_eye(_cpoints + myrand(_cpoints)*0, alpha1(_rgba*1),  None, 'eye3', set_lims=False, ax3d_reuse=ax3)


            _ = visualise_3d_situation_eye(_casted_points, _rgba, None, 'eye4', set_lims=False, ax3d_reuse=ax3)
            #_ = visualise_3d_situation_eye(_cpoints, alpha1(_rgba*0),  None, 'eye4*', set_lims=False, ax3d_reuse=ax3)

            #focus_3d(ax3, [45,45,-2], 15.0)
            #focus_3d(ax3, [0, 0, 0], 15.0*3)
            #focus_3d(ax3, [0, 0, 0], 15.)
            focus_3d(ax3, *FOCUS)

            #plt.show()


            ax2 = \
              visualise_uv(_uv[:,0], _uv[:,1], None, None, textures[0], uv_rgba=_rgba) #, fig=animation_fig)
            ax2.set_xlim(0,1.0)
            ax2.set_ylim(0,1.0)
            ax2.set_aspect('auto')
            ax2.set_title('sampled pixels: eye pair')
            #ax2.text(-0.1, 1.1, text_description, size=10, color='k', backgroundcolor='w')







            print('okkkkkk')
            #exit()
        # in progress
        #print('----first')
        #print(selected_center_points.shape, casted_points.shape)  # (3250, 3) (6496, 3)
        #print(regions_rgba.shape)
        # fails:
        #visualise_3d_situation_eye(casted_points, regions_rgba, beeHead, 'eye9', ax3d_reuse=ax3)
        plt.show()
        #exit()

        uv_rgb, *_ = sampling.sample_colors(uv, None, textures[0])
        uv_rgba = rgb_to_rgba(uv_rgb)
        #print('----second')
        OFFSET = np.array([2,0,0])[None,:]*0

        # todo: reomove
        ax3 = visualise_3d_situation_eye(O+OFFSET, alpha1(uv_rgba*0), None, 'eye5', set_lims=False, ax3d_reuse=None)
        visualise_3d_situation_eye(casted_points, uv_rgba, None, 'eye6', ax3d_reuse=ax3,set_lims=False)

        if False:
            #ax3.set_xlim(-100,100)
            ax3.set_xlim(0,80)
            ax3.set_ylim(0,80)
            #ax3.set_zlim(-100,100)

            #ZS = 5.0
            ZS = 15.0
            ax3.set_zlim(-2-ZS, -2+ZS)
            ax3.set_xlim(45-ZS, 45+ZS)
            ax3.set_ylim(45-ZS, 45+ZS)

        focus_3d(ax3, *FOCUS)

        #print('second ok')

        # see /Users/a9858770/cs/scientific-code/beeseyes/p3/lib/python3.9/site-packages/matplotlib/axes/_axes.py : 4496

        # reminder: selected_center_points = select_centers(which_facets, center_points)
        plt.show()


    if whether_visualise_uv_samples:
        ax2 = \
          visualise_uv(uvm_debug[:,0], uvm_debug[:,1], None, None, textures[0], uv_rgba=regions_rgba, fig=animation_fig)

        ax2.set_xlim(0,1.0)
        ax2.set_ylim(0,1.0)
        ax2.set_aspect('auto')
        ax2.set_title('sampled pixels*')

        ax2.text(-0.1, 1.1, text_description, size=10, color='k', backgroundcolor='w')

        anim_frame_artist = ax2

    return (beeHead, regions_rgba, anim_frame_artist)

from trajectory import trajectory_provider

# uses trajectory

def cast_and_visualise(
      corner_points, normals_at_corners,
      center_points, normals_at_center_points,
      ommatidia_few_corners_normals, ommatidia_few_corners,
      selected_regions, selected_center_points,
      which_facets,
      all_pair):

    FLIP_XZ = np.diag([-1.0,1,-1])
    def mult(a,b, c):
      return np.dot(a, b.T).T  + c #+ c[None,:]
    corner_points = mult(FLIP_XZ, corner_points, 0)
    normals_at_corners = mult(FLIP_XZ, normals_at_corners, 0)
    center_points = mult(FLIP_XZ, center_points, 0)
    normals_at_center_points = mult(FLIP_XZ, normals_at_center_points, 0)
    ommatidia_few_corners_normals = mult(FLIP_XZ, ommatidia_few_corners_normals, 0)
    ommatidia_few_corners = mult(FLIP_XZ, ommatidia_few_corners, 0)
    selected_center_points = mult(FLIP_XZ, selected_center_points, 0)

    if False:
      # PLAIN plot raw eye data
      ax3 = visualise_3d_situation_eye(center_points, center_points*0.0, None, 'simple eye')
      # **
      #ax3 = visualise_3d_situation_eye(corner_points, corner_points*0.0, None, 'simple eye')

      if False:
            assert X.shape == N.shape
            assert X.shape[0] == N.shape[0]

            qv = ax3d.quiver( \
              X[:,0], X[:,1], X[:,2], \
              N[:,0],N[:,1],N[:,2], \
              pivot='tail', length=0.1/10, normalize=True, color=color
              )

      # 0000
      plt.show()
      exit()


    #(M, bee_head_pos, bee_direction) = trajectory_provider()
    #(M, bee_path, bee_directions, frame_times) = trajectory_provider()
    (bee_path, bee_directions, frame_times) = trajectory_provider(sceneconfig.POSITIONS_XLS)


    #######################################
    # Part one: Figure one: non-animated
    # Custom mmanual bee position
    #######################################

    # pick a single frame for first sample
    frame_index = 100
    #bee_head_pos = bee_path[frame_index][None,:]
    #bee_head_pos = bee_path[None, frame_index,:]
    bee_head_pos = bee_path[frame_index,:]
    bee_direction = bee_directions[frame_index]
    frame_time = frame_times[frame_index]

    print('frame_time', frame_time)
    print('bee_direction', bee_direction)
    print('bee_head_pos', bee_head_pos)
    #return (M, bee_head_pos, bee_direction)

    # IN_PROGRESS:
    # requires some refacgtoring/parametrising the scale of the experiment. 

    # left eye
    # lean right

    # Y: -1=nose-up +1=nose-down
    # X: +1=left (of the honey bee)
    # Z: -1=front (oops)

    #0.7071067811865475 = sin(45)
    #0.49999999999999994 = sin(30)

    # Currently doesnt work. Using hard-coded values:
    #RELATIVE_SCALE = 20.0 * UNIT_LEN_CM
    #bee_direction = np.array([-20.0/2,  -20.0/2*0,  1.0])
    #bee_head_pos = np.array([RELATIVE_SCALE*0.2*5,  RELATIVE_SCALE*0.2*5, RELATIVE_SCALE*0.2])

    # RELATIVE_SCALE = 20.0 * UNIT_LEN_CM
    #bee_direction = np.array([0.5,  0,  0.86])
    #bee_head_pos = np.array([RELATIVE_SCALE*0.2*11,  RELATIVE_SCALE*0.2*5, RELATIVE_SCALE*0.2])

    #RELATIVE_SCALE = 20.0 * UNIT_LEN_CM
    #bee_direction = np.array([0.5,  0,  0.86])
    #bee_head_pos = np.array([RELATIVE_SCALE*0.2*11,  RELATIVE_SCALE*0.2*5, RELATIVE_SCALE*0.01])
    #bee_head_pos = np.array([44.0*UNIT_LEN_CM,  20.0*UNIT_LEN_CM, 0.2*UNIT_LEN_CM])

    # picture physical size (80.8, 63.40) (cm x cm)
    #bee_head_pos = np.array([40.0*UNIT_LEN_CM,  31.0*UNIT_LEN_CM, 1*UNIT_LEN_CM])
    #bee_head_pos = np.array([50.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -10*UNIT_LEN_CM])


    #bee_direction = np.array([0.5,  0,  -0.86])   #looks towards positive-Z, left eye
    ##bee_head_pos = np.array([50.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -1*UNIT_LEN_CM])
    #bee_head_pos = np.array([45.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -1*UNIT_LEN_CM])
    #eye_sphere_size_cm = 2.0 * UNIT_LEN_MM * 0.001


    #bee_direction = np.array([-0.5,  0,  -0.86])   #looks towards positive-Z, left eye
    #bee_head_pos = np.array([45.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -1*UNIT_LEN_CM])
    #eye_sphere_size_cm = 2.0 * UNIT_LEN_MM * 0.001

    # after flip-ing the eye 180:

    # ###########
    # Works well. Just needs a minor rotaation
    # ##########
    #bee_direction = np.array([-0.5,  0,  0.86])   #looks towards positive-Z, left eye
    #bee_head_pos = np.array([45.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -2*UNIT_LEN_CM])
    #eye_sphere_size_cm = 2.0 * UNIT_LEN_MM * 10/10 # * 0.1*400
    #clip=CAST_CLIP_FULL

    # Works well. Just needs rotation:
    #bee_direction = np.array([-0.05,  -0.5,  0.86])   #looks towards positive-Z, left eye
    # bee_head_pos = np.array([45.0*UNIT_LEN_CM,  (45.0+0)*UNIT_LEN_CM, -2*UNIT_LEN_CM])
    #bee_head_pos = np.array([41.0,  41.0, -2]) * UNIT_LEN_CM
    #bee_head_pos = np.array([41.0,  41.0, -5]) * UNIT_LEN_CM
    #bee_head_pos = np.array([2.5,  2.5, -2]) * UNIT_LEN_CM
    #bee_head_pos = np.array([0.0,  5.0, -1.5]) * UNIT_LEN_CM
    #bee_head_pos = np.array([2.0,  3.0, -1.5]) * UNIT_LEN_CM
    #bee_direction = np.array([-0.05,  -0.5,  0.86])   #looks towards positive-Z, left eye
    global FOCUS
    #FOCUS = ([0, 0, 0], 15.)
    #FOCUS = ([0, 0, 0], 5.)
    #clip=CAST_CLIP_NONE
    #clip=CAST_CLIP_FULL

    from sceneconfig import TEXTURES_FILES, TEXTURE_DPI_INFO

    # After meeting with Hadi (14 Sept)
    bee_direction = np.array([-0.01,  -0.01,  1.0])
    #  ... 5cm
    #bee_head_pos = np.array([2.5,  2.5, -1.5]) * UNIT_LEN_CM
    #  (4.46 x 3.5) (cm)
    bee_head_pos = np.array([2.2,  1.75, -1.5]) * UNIT_LEN_CM
    #bee_head_pos = np.array([2.2-1.5,  1.75,  -1.0]) * UNIT_LEN_CM # chosen, but not looking good?
    # +DOWN, +RIGHT
    #FOCUS = (bee_head_pos, 5.)
    FOCUS = (bee_head_pos, 0.2)
    eye_sphere_size_cm = 2.0 * UNIT_LEN_MM
    clip=CAST_CLIP_FULL


    print('eye_sphere_size_cm', eye_sphere_size_cm)

    frame_time = 0.0

    assert len(TEXTURES_FILES) == 2, '''Currently, textures for only only two sides supported for now. Coming soon.'''

    # 2D Visualisation of (u,v) on textures
    #texture1, physical_size1, dpi1 = image_loader.load_image_withsize(TEXTURES_FILES[0], sample_px=200, sample_cm=10.0)
    texture1, physical_size1, dpi1 = image_loader.load_image_withsize(TEXTURES_FILES[0], sample_px=TEXTURE_DPI_INFO[_PIXELS], sample_cm=TEXTURE_DPI_INFO[_CM])
    texture2, physical_size2, dpi2 = image_loader.load_image_withsize(TEXTURES_FILES[1], dpi=dpi1)
    assert dpi1 == dpi2

    #RELATIVE_SCALE_OF_EXPERIMENT = 30
    #RELATIVE_SCALE = 30
    print('physical_size1', physical_size1) # (35.050000000000004, 58.1)

    textures = [texture1, texture2]

    #plane2 = Plane(*physical_size2)
    #plane1 = Plane(*physical_size1)
    #plane1 = Plane(*physical_size1, C0_pos=(0,0,0))
    # Blue flower:
    #plane1 = Plane(*physical_size1, C0_pos=(-17.5,-26.0,0))  # units in CM
    plane1 = Plane(*physical_size1, C0_pos=(0.0, 0.0, 0))  # units in CM
    #planes = [plane1, plane2]
    planes = [plane1,]

    selected_center_points_normal = None
    beeHead, regions_rgba,_ = \
    anim_frame(
        textures, planes,
        bee_head_pos, bee_direction, eye_sphere_size_cm, clip,
        corner_points, normals_at_corners,
        ommatidia_few_corners, ommatidia_few_corners_normals,
        selected_regions, which_facets,
        #
        all_pair,
        #
        selected_center_points, selected_center_points_normal,
        #center_points, normals_at_center_points,
        whether_visualise_eye_3d=True, whether_visualise_uv_samples=True,
        whether_visualise_uv_scatter=True,
        text_description='text'
    )
    # show first figure, using custom Bee position
    plt.show()
    #exit()

    #######################################
    # Part two: Figure two: animated, etc
    #######################################

    print('bee_path>>', bee_path)
    # Visualisations

    from visualise.visualise_3d_situation import visualise_3d_situation

    # 3D Visualisation of environment
    ax3d = \
    visualise_3d_situation(corner_points, normals_at_corners, ommatidia_few_corners, ommatidia_few_corners_normals, center_points, normals_at_center_points, beeHead, planes)
    ax3d.plot3D(bee_path[:,0], bee_path[:,1], bee_path[:,2], alpha=0.1, linewidth=0.3, marker='.')
    ax3d.set_xlim(*array_minmax(bee_path[:,0]))
    ax3d.set_ylim(*array_minmax(bee_path[:,1]))
    ax3d.set_zlim(*array_minmax(bee_path[:,2]))

    trajectory_stats(bee_path)


    # visualise_3d_situation_eye(center_points, regions_rgba, beeHead, 'sferikal ')

    ## corner_points linked to uv
    #assert corner_points.shape[0] == uv.shape[0]
    #visualise_uv_map(corner_points, regions_rgba)
    # visualise_map_spherical_to_planar(corner_points, regions_rgba)

    # center_points linked to regions_rgba
    #assert corner_points.shape[0] == uv.shape[0]
    #print('assert corner_points.shape[0] == regions_rgba.shape[0]', corner_points.shape, regions_rgba.shape)
    #assert corner_points.shape[0] == regions_rgba.shape[0]
    #visualise_map_spherical_to_planar(center_points, regions_rgba)
    # No, we don't have the projection of the center points (in `uv` and  `regions_rgba`). They aare made from coorner points.
    #

    #print('corner_points.shape ?== regions_rgba.shape', corner_points.shape, regions_rgba.shape)
    #assert corner_points.shape[0] == regions_rgba.shape[0]
    #visualise_map_spherical_to_planar(corner_points, regions_rgba)

    print('selected_center_points.shape ?== regions_rgba.shape', corner_points.shape, regions_rgba.shape)
    assert selected_center_points.shape[0] == regions_rgba.shape[0]
    ax2d = \
    visualise_map_spherical_to_planar(selected_center_points - 0, regions_rgba)
    ax2d.set_title('carto-')

    histogram_of_sides(selected_regions)

    #return # skip animation creation
    #NUM_FRAMES = 20*3
    NUM_FRAMES = 1
    # NUM_FRAMES = 5
    #NUM_FRAMES = 20
    #NUM_FRAMES = 20

    # Animation
    from matplotlib.animation import FuncAnimation
    anim_fig = plt.figure()
    #def anim_init():
    #    pass
    def animate_frame(animation_frame_index):
        #for frame_index in range(2): # range(len(bee_path)):
        frame_index = animation_frame_index + 100+60

        bee_head_pos = bee_path[frame_index]
        bee_direction = bee_directions[frame_index]
        frame_time = frame_times[frame_index, 0]

        print('ok')
        print(bee_head_pos.shape) # (3,)
        print(bee_head_pos)
        print(bee_direction.shape) # (3,)
        print(bee_direction)

        eye_sphere_size_cm = 1.0 * UNIT_LEN_CM

        clip=CAST_CLIP_FULL

        #(all_centers,all_normals_at_center_points) = (None,None)
        all_pair = None

        text_description = f'time: {round(frame_time,2)} (s)   frame:{animation_frame_index}'
        anim_fig.clear() # Much faster
        (beeHead, regions_rgba, anim_frame_artist) = \
        anim_frame(
            textures, planes,
            bee_head_pos, bee_direction, eye_sphere_size_cm, clip,
            corner_points, normals_at_corners,
            ommatidia_few_corners, ommatidia_few_corners_normals,
            selected_regions, which_facets,
            all_pair, #(all_centers,all_normals_at_center_points),
            selected_center_points, selected_center_points_normal,
            #center_points, normals_at_center_points,
            whether_visualise_eye_3d=False, whether_visualise_uv_samples=True,
            whether_visualise_uv_scatter=False,
            animation_fig=anim_fig,
            text_description=text_description
        )
        print('frame done:', animation_frame_index, 'traj frame:', frame_index)
        return (anim_frame_artist,)

    # plt.rcParams['animation.ffmpeg_path'] = u'/usr/local/bin/ffmpeg'
    anim = FuncAnimation(anim_fig, animate_frame, #init_func=anim_init,
                               frames=NUM_FRAMES, interval=100 #, blit=True
                               , blit=False
                               #, progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
                               )
    #anim.save(my_folders.get_output_path('beeAnim.mpeg'))
    anim.save(my_folders.get_output_path('beeAnim.gif'), writer='imagemagick')
    # ffmpeg, codec=s, writer='imagemagick' for gif
    # Also see: Animation.to_html5_video
    print('saved animation', flush=True)


def main2():
   (corner_points, normals_at_corners), (center_points, normals_at_center_points), (ommatidia_few_corners_normals, ommatidia_few_corners), selected_regions, selected_center_points, which_facets, (all_centers,all_normals_at_center_points) = aaaaa()
   cast_and_visualise(corner_points, normals_at_corners, center_points, normals_at_center_points, ommatidia_few_corners_normals, ommatidia_few_corners, selected_regions, selected_center_points, which_facets, (all_centers,all_normals_at_center_points))

main2()

# old_demo refactored out of this
#old_demo.old_demo()

plt.show()
