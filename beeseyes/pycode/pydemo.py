from typing import Tuple
from typing import Tuple
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
import math

from bee_eye_data import ommatidia_polygons, ommatidia_polygons_fast_representation
from bee_eye_data import ax3dCreate, visualise_all
from bee_eye_data import make_midpoints, make_deviations, my_index

HEX6 = 6
DIM3 = 3

#0.00125 , 0.01/2/4
AREA_THRESHOLD = 0.02*10

#0.2
SD_THRESHOLD=0.03

AREA_THRESHOLD = 0.02*10 *100000+1000
SD_THRESHOLD=0.03 * 10000+1000



# https://en.wikipedia.org/wiki/Blue_flower#/media/File:Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpg
# https://en.wikipedia.org/wiki/Blue_flower
BLUE_FLOWER = "../art/256px-Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpeg"

#BLUE_FLOWER = '/Users/a9858770/Documents/xx/3bebe3b139b7e0e01573faabb4c92934.jpeg'
#BLUE_FLOWER = '/Users/a9858770/Documents/bee-walt-Spike_art.PNG.png'

def eye_centers(n):
  nx = int(np.sqrt(n)+1) +2
  ny = int(np.sqrt(n)) + 2

  xa = np.linspace(0,nx-1,nx)
  ya = np.linspace(0,ny-1,ny)
  [X,Y] = np.meshgrid(xa,ya)
  X[0::2] += 0.5 # shift
  xc = X.ravel()[:,None]
  yc = Y.ravel()[:,None]
  print(xc.shape, yc.shape) # (56, 1) (56, 1)
  c_xyi = np.concatenate((xc, yc), axis=1)
  eta = (np.random.rand(*c_xyi.shape)-0.5) * 0.2
  c_xy  = c_xyi + eta # + m_odd * 0.2

  c_xy = c_xy / float(nx)

  # indices: (0:nx, 0:nx, ...) x ny
  return c_xy, (nx,ny)

''' By james-porter, https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci '''

def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.vertices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])

            # this is a one liner for if a simplex contains the point we`re interested in,
            # extend the neighbors list by appending all the *other* point indices in the simplex

    # now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(set(neighbors))

'''
  ret[vert_index] = array of len 6 of vert_index.
  Padded with NO_VERT=-1

  Not very fast
'''
def all_hexa_verts(points_xy):
    n = points_xy.shape[0]
    tri = Delaunay(points_xy)
    NO_VERT = -1
    hexa_verts = np.zeros((n,6), dtype=int)

    for pindex in range(n):
       # find 6 neighbours
       nl = find_neighbors(pindex, tri)
       nla = np.array(nl)
       padlen = 6 - len(nl)
       if padlen > 0:
          nla = np.pad(nla, (0,padlen,), 'constant', constant_values=(NO_VERT, NO_VERT,))
          nla = nla[:6] * 0 + NO_VERT
       if padlen < 0:
          nla = nla[:6] * 0 + NO_VERT

       #print(nla)
       hexa_verts[pindex, :] = nla
    #print(hexa_verts)
    return hexa_verts

def eye_attribs_demo(points_xy):
    n = points_xy.shape[0]
    zc = np.zeros((n, 1))
    cxyz = np.concatenate((points_xy, zc), axis=1)
    z0 = np.zeros((n, 1))
    normals = np.concatenate((z0+0.0, z0+0.0, z0 + 1.0 ), axis=1)

    vor = Voronoi(points_xy)
    print(vor.vertices)
    print(vor.vertices.shape)
    # return vor.vertices

    tri = Delaunay(points_xy)
    pindex = 35
    nl = find_neighbors(pindex, tri)
    print('find_neighbors', nl)

    plt.plot(points_xy[nl,0], points_xy[nl,1], 'g*')
    plt.plot(points_xy[pindex,0], points_xy[pindex,1], 'go')

    all_hexa_verts(points_xy)
    return vor.vertices

def show_hexagons(points_xy):
    #def hexagons(points_xy):
    vor = Voronoi(points_xy)
    if True:
       fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                    line_width=2, line_alpha=0.6, point_size=2)


    tri = Delaunay(points_xy)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

    # the dual (of voronoi)
    if True:
       plt.triplot(points_xy[:,0], points_xy[:,1], tri.simplices)
       plt.plot(points_xy[:,0], points_xy[:,1], 'o')

    #print('simpices', tri.simplices)

    #print('tri', dir(tri))
    # # print(tri.vertex_to_simplex)
    #print()
    #print(tri.vertex_neighbor_vertices)  # (indptr, indices)


def eye_attribs(points_xy, z_offset):
    n = points_xy.shape[0]
    zc = np.zeros((n, 1))
    cxyz = np.concatenate((points_xy, zc), axis=1)
    z0 = np.zeros((n, 1))
    normals = np.concatenate((z0+0.0, z0+0.0, z0 + z_offset ), axis=1)
    eta = (np.random.rand(*normals.shape)-0.5) * 0.003
    normals = normals + eta

    return cxyz, normals

def one_hexagonal_rays(pindex, hexa_verts_table, points_xyz, normals_xyz):
    # ray shoot: origin, dir
    # rays =
    #rays_origins = np.zeros((HEX6, 3))
    #rays_dirs = np.zeros((HEX6, 3))


    indices = hexa_verts_table[pindex, :]
    rays_origins = points_xyz[indices, :]
    # centre = points_xyz[pindex, :] # not used
    # deliberately incorrect:
    centre_normal = normals_xyz[pindex, :][None, :]
    rays_normals = (normals_xyz[indices, :] + centre_normal ) * 0.5

    #for i in range(HEX6):
    #   rays_normals = rays_normals
    #points_xyz =

    rays_dirs = rays_normals
    # return rays
    return (rays_origins, rays_dirs)

'''
     Based on `derive_formulas.py`
'''
def ray_cast(
              U : tuple[float, float, float],
              V : tuple[float, float, float],
              C0 :tuple[float, float, float],
              D,
              O,
              clip:bool=True
            ):
   n = D.shape[0]
   assert D.shape == (n,3)
   assert O.shape == (n,3)
   assert len(C0) == 3
   assert len(U) == 3
   assert len(V) == 3

   # vectorised
   dx,dy,dz = D[:,0], D[:,1], D[:,2]
   ox,oy,oz = O[:,0], O[:,1], O[:,2]

   # simple value
   ux,uy,uz = float(U[0]),  float(U[1]), float(U[2])
   vx,vy,vz = float(V[0]),  float(V[1]), float(V[2])
   x0,y0,z0 = float(C0[0]), float(C0[1]), float(C0[2])

   # solution: u,v
   denom =  dx*uy*vz - dx*uz*vy - dy*ux*vz + dy*uz*vx + dz*ux*vy - dz*uy*vx
   a = (dx*oy*vz - dx*oz*vy + dx*vy*z0 - dx*vz*y0 - dy*ox*vz + dy*oz*vx - dy*vx*z0 + dy*vz*x0 + dz*ox*vy - dz*oy*vx + dz*vx*y0 - dz*vy*x0) / denom
   b = (-dx*oy*uz + dx*oz*uy - dx*uy*z0 + dx*uz*y0 + dy*ox*uz - dy*oz*ux + dy*ux*z0 - dy*uz*x0 - dz*ox*uy + dz*oy*ux - dz*ux*y0 + dz*uy*x0) / denom
   t = (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx + ux*vy*z0 - ux*vz*y0 - uy*vx*z0 + uy*vz*x0 + uz*vx*y0 - uz*vy*x0) / denom

   if (clip):
       w = np.logical_and(t > 0,   a > -1.9)
       w = np.logical_and(w,   b > -1.9)
       w = np.logical_and(w,   a < 2.9)
       w = np.logical_and(w,   b < 2.9)
       #a = a[w]
       #b = b[w]
       #t = t[w]
       not_w = np.logical_not(w)
       a[not_w] = np.NaN
       b[not_w] = np.NaN
       t[not_w] = np.NaN

   # todo: remove negative `t`
   #print(a)
   #print(b)
   return (a,b)


def tuple3_to_np(pxyz):
   x,y,z = pxyz
   return np.array([x,y,z], dtype=float)[None, :]

def normalise_np(pxyz):
   return pxyz * 1.0 / (np.linalg.norm(pxyz, axis=0, keepdims=True) + 0.0000001)

def rotation_matrix(bee):
   # right hand
   U = tuple3_to_np(bee['u'])
   # top
   V = tuple3_to_np(bee['v'])
   U = normalise_np(U)
   # straight ahead: into the plane
   W = -np.cross(U, V)
   W = normalise_np(W)
   # again re-create the "top" direction
   V = np.cross(U, W)
   V = normalise_np(V)
   # return np.eye(3)
   # [U,V,W] concat as rows
   return np.concatenate((U,V,W), axis=0)

def visualise_plane(ax3d, plane):
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
     marker='.', color='g')

def visuaise_3d(rays_origins, rays_dirs, points_xyz, plane):
    SZ=8.0*1.2 * 3
    ax3d = plt.figure().add_subplot(projection='3d', autoscale_on=False,
       xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0))
    # ax3d = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    #ax3d.set_aspect('equal')
    #ax3d.set_aspect(1)

    # only a single hex being casted
    qv = ax3d.quiver( \
     rays_origins[:,0],rays_origins[:,1],rays_origins[:,2], \
     rays_dirs[:,0],rays_dirs[:,1],rays_dirs[:,2], \
     pivot='tail', length=1.0, normalize=True, color='r'
    )
    '''
    ax3.quiverkey(qv, 0.9, 0.9, 1, r'$xxxx$', labelpos='E',
               coordinates='figure')
    '''

    # All ommatidia
    ax3d.scatter(points_xyz[:,0],points_xyz[:,1],points_xyz[:,2], marker='.')

    visualise_plane(ax3d, plane)


import scipy.misc
import imageio
def load_image(image_path):
   img = imageio.imread(image_path)
   pict_array2d = np.asarray(img)
   #pict_array2d = numpy.array(Image.fromarray(pict_array2d).resize())
   #pict_array2d = scipy.misc.imresize(pict_array2d, FIXED_SIZE[0:2])
   return pict_array2d

'''
Samples the area in the image (pixels)
inside the given 6-points (hexagon)
that is casted on the image.
The image coords (and u,v) are within [0,1]x[0,1]
6-point area/pixel sampling
'''
def sample_hex(u6,v6, texture):
    (w,h,rgb3) = texture.shape # (192, 256, 3)
    print(u6*w)
    print(v6*h)
    pass

def demo_lattice_eyes(EYE_SIZE):
    #N = 7 * 7
    #NW = 8
    #N = 5000
    #NW = 70
    N = 500

    points_2d,_ = eye_centers(N)
    #v, _ = eye_attribs_demo(points_2d, 1.0)
    show_hexagons(points_2d)
    #show_hexagons(v)

    hexa_verts_table = all_hexa_verts(points_2d)

    '''
    v, _ = eye_attribs_demo(points_2d, 1.0)
    '''

    (points_xyz, normals_xyz) = eye_attribs(points_2d, 1.0)
    pindex = 35
    rays_origins, rays_dirs = one_hexagonal_rays(pindex, hexa_verts_table, points_xyz, normals_xyz)

    rays_origins_e = rays_origins * EYE_SIZE
    eye_points = points_xyz * EYE_SIZE \
      + 0 * tuple3_to_np((15.0,15.0,-10.0))  # in cm

    return eye_points, normals_xyz, rays_origins_e, rays_dirs



def raycastOmmatidium(eye_points, rays_dirs, bee_R, bee_pos, plane, clip:bool=True):
   n = rays_dirs.shape[0]
   assert eye_points.shape == (n, DIM3)
   assert rays_dirs.shape == (n, DIM3)
   assert bee_pos.shape == (1,DIM3)
   assert bee_R.shape == (DIM3, DIM3)

   O = np.dot(bee_R, eye_points.T).T + bee_pos
   D = np.dot(bee_R, rays_dirs.T).T
   (u,v) = ray_cast(plane.U,plane.V,plane.C0, D,O, clip=clip)
   return O, D, (u,v)

class Plane:
    def __init__(self):
       # centimeters
       # todo: rename U -> A, a->uz
       plane = {
          'U': (30,0,0),
          'V': (0,30,0),
          'C0': (0,0,0),
       }
       '''
           x = u * U + v * V + C0
           x = t * N + P0 + Bee.pos
           (t,u,v)

               0 < u < 1
               0 < v < 1
               t > 0

           [plane1, plane2, ]
           Planes:
             * decision wall:  50x40
             * side wall:  50x100
             * floor & ceiling:  ...
       '''
       self.U = plane['U']
       self.V = plane['V']
       self.C0 = plane['C0']



class BeeHead:
    def __init__(self):
        bee = {
          'pos': (15.0, 15.0, -10.0/3),
          #'u': (15.0, 15.0, -10.0),
          'u': (1, 0.1,-0.2), # Bee's right hand  (1,0,-0.2)
          'v': (0,1,0),  # Bee's top
        }
        # bee_R = rotation_matrix(bee)
        #bee_R = np.eye(3)

        bee_R = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        print('bee_R.shape', bee_R.shape)

        bee_pos = tuple3_to_np(bee['pos'])
        # self.np = {}

        self.R = bee_R
        self.pos = bee_pos


def main():

    EYE_SIZE = 0.1*10 # cm
    eye_points, normals_xyz, rays_origins_e, rays_dirs \
       = demo_lattice_eyes(EYE_SIZE)


    texture = load_image(BLUE_FLOWER)
    #  (192, 256, 3)

    plane = Plane()

    beeHead = BeeHead()

    O,D,(u,v) = raycastOmmatidium(eye_points, normals_xyz, beeHead.R, beeHead.pos, plane)

    (rays_origins_transformed, rays_dirs_transformed, (u6,v6)) = raycastOmmatidium(rays_origins_e, rays_dirs, beeHead.R, beeHead.pos, plane)
    visuaise_3d(rays_origins_transformed, rays_dirs_transformed, O, plane)


    axes2 = plt.figure()
    plt.imshow(texture, extent=(0.0,1.0,0.0,1.0), alpha=0.6)
    plt.plot(v, 1-u, '.')
    plt.plot(v6, 1-u6, 'r.')

    sample_hex(u6,v6, texture)

    # plt.show()

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


    which_facets = make_whichfacets(corner_points, sv_regions, areas, SD_THRESHOLD=SD_THRESHOLD, AREA_THRESHOLD=AREA_THRESHOLD, MAX_SIDES=14)

    #selected_regions = select_regions(sv_regions, areas, corner_points, MAX_SIDES=14)
    selected_regions =  select_regions(sv_regions, which_facets)
    selected_center_points = select_centers(which_facets, center_points)
    # `selected_regions` is linked to [indices of] `(corner_points, normals_at_corners)`

    return (corner_points, normals_at_corners), (center_points, normals_at_center_points), (ommatidia_few_corners_normals, ommatidia_few_corners), selected_regions, selected_center_points, which_facets

def visualise_3d_situation(corner_points, normals_at_corners, ommatidia_few_corners, ommatidia_few_corners_normals, center_points, normals_at_center_points, beeHead, plane):
    p0 = beeHead.pos
    print('p0.shape',p0.shape)

    def rot(vectos):
        return np.dot(beeHead.R, vectos.T).T

    ax3d = ax3dCreate()
    # rename `corner_points` to `*corners`
    visualise_all(ax3d, rot(corner_points) + p0, rot(normals_at_corners), 'r')  # corners
    visualise_all(ax3d, rot(center_points) + p0, rot(normals_at_center_points), 'b') # centers

    general_direction = np.mean(center_points, 0)[None,:]
    visualise_all(ax3d, rot(general_direction) + p0, rot(general_direction), color='k') # centers
    visualise_plane(ax3d, plane)
    print((normals_at_center_points*0).shape, '<<<<<<<=====')

    # adding Visualisation of few points in 3D
    print('====', ommatidia_few_corners_normals.shape, ommatidia_few_corners.shape)
    assert ommatidia_few_corners_normals.shape == ommatidia_few_corners.shape
    visualise_all(ax3d, rot(ommatidia_few_corners) + p0, ommatidia_few_corners_normals * 0.01, 'm')


def array_minmax(x):
    mn = np.min(x)
    mx = np.max(x)
    md = (mn + mx)/2.0
    (mn,mx) = ((mn-md)*1.2+md, (mx-md)*1.2+md)
    return (mn, mx)

def visualise_3d_situation_eye(selected_center_points, regions_rgb, beeHead, title):
    assert selected_center_points.shape[0] == regions_rgb.shape[0]

    p0 = beeHead.pos

    def rot(vectors):
        return np.dot(beeHead.R, vectors.T).T

    X = rot(selected_center_points) + p0

    ax3d = ax3dCreate()
    ax3d.scatter(X[:,0], X[:,1], X[:,2], facecolors=regions_rgb, marker='.')
    print('XXX ', X)
    print('regions_rgb ', regions_rgb)

    ax3d.set_xlim(*array_minmax(X[:,0]))
    ax3d.set_ylim(*array_minmax(X[:,1]))
    ax3d.set_zlim(*array_minmax(X[:,2]))
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title(title)

    return ax3d

def visualise_uv(u,v, u_few, v_few, texture, uv_rgba=None, title=None):
    # (u,v) visualisation on plane (pixels)
    axes2 = plt.figure()
    ax = axes2.add_subplot(111)
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

def sample_colors(uv, regions, texture):
    print('uv.shape', uv.shape)
    if texture.shape[2] == 4:
        texture = texture[:,:, 0:3]
    nan_rgb = np.zeros((3,)) + np.NaN
    EPS = 0.00000001
    (H,W) = texture.shape[0:2]
    print('W,H', W,H)
    W_ = (W + 1 - EPS)
    H_ = (H + 1 - EPS)
    nf = len(regions)
    uvm_for_debug = np.zeros((nf,2),dtype=float)
    regions_rgb = np.zeros((nf,3),dtype=float)
    for i in range(nf):
        # temporary solution: sample at center only
        #if np.isnan(uv[regions[i], 0]):
        um = np.mean(uv[regions[i], 0])
        vm = np.mean(uv[regions[i], 1])
        uvm_for_debug[i, :] = [um, vm]
        if np.isnan(um) or np.isnan(vm):
            rgb = nan_rgb
        else:
            # sample
            py = math.floor(um * H_)
            px = math.floor(vm * W_)
            if px < 0 or py < 0 or px >= W or py >= H:
                rgb = nan_rgb
            else:
               rgb = texture[py,px]
        regions_rgb[i] = rgb
    return regions_rgb, uvm_for_debug

def cast_and_visualise(corner_points, normals_at_corners, center_points, normals_at_center_points, ommatidia_few_corners_normals, ommatidia_few_corners, selected_regions, selected_center_points, which_facets):
    plane = Plane()
    beeHead = BeeHead()

    # corners, normals_at_corners (6496, 3) (6496, 3)
    print('corners, normals_at_corners', corner_points.shape, normals_at_corners.shape)
    O,D,(u,v) = raycastOmmatidium(corner_points, normals_at_corners, beeHead.R, beeHead.pos, plane )
    print('>>>u,v', u.shape)

    # Visualisations

    # 3D Visualisation of environment
    visualise_3d_situation(corner_points, normals_at_corners, ommatidia_few_corners, ommatidia_few_corners_normals, center_points, normals_at_center_points, beeHead, plane)

    O_few,D_few,(u_few,v_few) = raycastOmmatidium(
       ommatidia_few_corners, ommatidia_few_corners_normals,
       beeHead.R, beeHead.pos, plane,
       clip=False)


    # 2D Visualisation of (u,v) on textures
    texture = load_image(BLUE_FLOWER)
    #  (192, 256, 3)

    visualise_uv(u,v, u_few, v_few, texture)

    # selected_center_points = select_centers(which_facets, center_points)

    uv = np.concatenate((u[:,None], v[:,None]), axis=1)
    #selected_uv = select_centers(which_facets, uv)

    print('u,v', u.shape)
    print('uv', uv.shape)
    print('selected_regions', len(selected_regions))
    nfs = np.sum(which_facets)
    #assert selected_uv.shape[0] == nfs
    assert len(selected_regions) == nfs
    #regions_rgb = sample_colors(selected_uv, selected_regions, texture)
    regions_rgb, uvm_debug = sample_colors(uv, selected_regions, texture)

    nf =len(selected_regions)
    #assert u.shape[0] == nf
    assert regions_rgb.shape[0] == nf
    assert len(selected_regions) == nf

    assert selected_center_points.shape[0] == nf
    assert regions_rgb.shape[0] == nf

    print(regions_rgb)
    print('non-nan', np.sum(np.logical_not(np.isnan(regions_rgb)), axis=0)) # [14,14,14]
    #print('non-nan', np.sum(np.logical_not(np.isnan(regions_rgb)), axis=1))

    nans = np.isnan(regions_rgb[:,0])
    one = np.ones((regions_rgb.shape[0],1), dtype=np.float)
    #print('max', np.nanmax(regions_rgb))
    regions_rgba = np.concatenate( (regions_rgb / 255.0, one), axis=1)
    _ALPHA = 3
    regions_rgba[nans, 0:2] = 0.0
    regions_rgba[:, _ALPHA] = 1.0
    regions_rgba[nans, _ALPHA] = 0.0
    #regions_rgba[:, :] = 0.0
    #regions_rgba[nans, _ALPHA] = 1.0


    # one center_point for each region. todo: re-index center_point-s based on selected regions
    visualise_3d_situation_eye(selected_center_points, regions_rgba, beeHead, 'good ones')

    #uv2 = uv.copy()
    #n1 = np.logical_not(nans[:])
    #print('n1.shape', n1.shape)
    #print('uv2.shape', uv2.shape)
    #uv2 = uv2[n1,:]
    #visualise_uv(uv2[:,0], uv2[:,1], uv2[0:0,0], uv2[0:0,1], texture)
    ax2 = \
      visualise_uv(uvm_debug[:,0], uvm_debug[:,1], None, None, texture, uv_rgba=regions_rgba)

    ax2.set_xlim(0,1.0)
    ax2.set_ylim(0,1.0)
    ax2.set_title('sampled pixels')



    # visualise_3d_situation_eye(center_points, regions_rgba, beeHead, 'sferikal ')

    '''
    #O,D,(u,v) = raycastOmmatidium(eye_points, normals_xyz, beeHead.R, beeHead.pos, plane)
    #def raycastOmmatidium(eye_points, rays_dirs, bee_R, bee_pos, plane):
    # sum(a[i,j,:] * b[k,:,m])
    # (3250, MAX_SIDES, 3), (3,3) -> (3250, MAX_SIDES, 3)
    O =np.dot(ommatidia_polygons1, bee_R.T) + bee_pos[None,None,:]
    # (3250, 3), (3,3) -> (3250, 3)
    D =np.dot(ommatidia_normals, bee_R.T) + bee_pos[None,None,:]
    #O = np.dot(bee_R, eye_points.T).T + bee_pos
    #D = np.dot(bee_R, rays_dirs.T).T
    (u,v) = ray_cast(plane.U,plane.V,plane.C0, D,O)
    return O, D, (u,v)
    '''

    print('.')


def main2():
   (corner_points, normals_at_corners), (center_points, normals_at_center_points), (ommatidia_few_corners_normals, ommatidia_few_corners), selected_regions, selected_center_points, which_facets = aaaaa()
   cast_and_visualise(corner_points, normals_at_corners, center_points, normals_at_center_points, ommatidia_few_corners_normals, ommatidia_few_corners, selected_regions, selected_center_points, which_facets)

main2()

#main()
plt.show()
