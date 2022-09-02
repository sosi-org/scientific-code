import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay

from cte import _PIXELS, _CM
from cte import UNIT_LEN_CM, UNIT_LEN_MM, CAST_CLIP_FULL

from sceneconfig import NEW_FLOWER, TEXTURE_DPI_INFO

from optics.geom_helpers import tuple3_to_np
# todo: move to the appropriate place:
from optics.get_actual_ommatidium_rays import getActualOmmatidiumRays
from optics.plane import Plane
from optics.bee_look import BeeHead
import image_loader

from visualise.visualise_3d import visualise_3d

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

def eye_attribs(points_xy, z_offset):
    n = points_xy.shape[0]
    zc = np.zeros((n, 1))
    cxyz = np.concatenate((points_xy, zc), axis=1)
    z0 = np.zeros((n, 1))
    normals = np.concatenate((z0+0.0, z0+0.0, z0 + z_offset ), axis=1)
    eta = (np.random.rand(*normals.shape)-0.5) * 0.003
    normals = normals + eta

    return cxyz, normals


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


'''
  res[vert_index] = array of len 6 of vert_index.
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



''' By james-porter, https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci 
How to find all neighbors of a given point in a delaunay triangulation using scipy.spatial.Delaunay?
'''

def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.vertices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])

            # this is a one liner for if a simplex contains the point we`re interested in,
            # extend the neighbors list by appending all the *other* point indices in the simplex

    # now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(set(neighbors))

# clip:bool=True

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


def eye_centers(n):
  nx = int(np.sqrt(n)+1) +2
  ny = int(np.sqrt(n)) + 2

  xa = np.linspace(0,nx-1,nx)
  ya = np.linspace(0,ny-1,ny)
  [X,Y] = np.meshgrid(xa,ya)
  X[0::2] += 0.5 # shift
  xc = X.ravel()[:,None]
  yc = Y.ravel()[:,None]
  #print(xc.shape, yc.shape) # (56, 1) (56, 1)
  c_xyi = np.concatenate((xc, yc), axis=1)
  eta = (np.random.rand(*c_xyi.shape)-0.5) * 0.2
  c_xy  = c_xyi + eta # + m_odd * 0.2

  c_xy = c_xy / float(nx)

  # indices: (0:nx, 0:nx, ...) x ny
  return c_xy, (nx,ny)

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

# Uses Voronoi
def old_demo():

    # Figure one is commpletely not aligned (after changing the logic for M)
    #TEXTURE_FILENAME = BLUE_FLOWER

    TEXTURE_FILENAME = NEW_FLOWER

    EYE_SIZE = 0.1*10 # cm
    eye_points, normals_xyz, rays_origins_e, rays_dirs \
       = demo_lattice_eyes(EYE_SIZE)


    # Not tested. Previous form was using load_image(), without the dpi specs
    #texture, physical_size0, dpi0 = image_loader.load_image_withsize(TEXTURE_FILENAME, sample_px=200, sample_cm=10.0)
    texture, physical_size0, dpi0 = image_loader.load_image_withsize(TEXTURE_FILENAME, sample_px=TEXTURE_DPI_INFO[_PIXELS], sample_cm=TEXTURE_DPI_INFO[_CM])
    #todo: rename: sample_px -> ...




    #  (192, 256, 3)


    #RELATIVE_SCALE = 20.0 * UNIT_LEN_CM
    #_bee_head_pos = np.array([RELATIVE_SCALE*0.2*5,  RELATIVE_SCALE*0.2*5, RELATIVE_SCALE*0.2])
    #_bee_direction = np.array([-20.0/2,  -20.0/2*0,  1.0])
    #_eye_sphere_size_cm = 0.3

    #_bee_direction = np.array([-0.5,  0,  -0.86])   #looks towards positive-Z, left eye
    #_bee_head_pos = np.array([45.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -1*UNIT_LEN_CM])
    #_eye_sphere_size_cm = 2.0 * UNIT_LEN_MM * 0.001 * 100

    #_bee_direction = np.array([-0.5,  0,  0.86])   #looks towards positive-Z, left eye
    #_bee_head_pos = np.array([45.0*UNIT_LEN_CM,  45.0*UNIT_LEN_CM, -2*UNIT_LEN_CM])
    #_eye_sphere_size_cm = 2.0 * UNIT_LEN_MM * 10/10 # * 0.1*400

    # searching to find it back
    _bee_direction = np.array([-0.5,  0,  0.86])   #looks towards positive-Z, left eye
    _bee_head_pos = np.array([0.1*45.0*UNIT_LEN_CM,  0.1*45.0*UNIT_LEN_CM, -2*UNIT_LEN_CM])
    _eye_sphere_size_cm = 2.0 * UNIT_LEN_MM * 10/10


    plane = Plane(30.0, 30.0)

    beeHead = BeeHead()
    beeHead.set_eye_position(_bee_head_pos)
    beeHead.set_direction(_bee_direction, eye_size_cm=_eye_sphere_size_cm)

    eye_size_cm = 1.0 * UNIT_LEN_CM


    print('@@normals_xyz', normals_xyz)
    print('@@beeHead', beeHead)
    print('@@beeHead.p', beeHead.pos)

    # why is None? : beeHead.pos

    O,D = getActualOmmatidiumRays(eye_points, normals_xyz, beeHead.R, beeHead.pos, eye_size_cm)
    (u,v) = plane.raycastRaysOnPlane(O,D, clip=CAST_CLIP_FULL)


    rays_origins_transformed, rays_dirs_transformed = getActualOmmatidiumRays(rays_origins_e, rays_dirs, beeHead.R, beeHead.pos, eye_size_cm)
    (u6,v6) = plane.raycastRaysOnPlane(
      rays_origins_transformed, rays_dirs_transformed,
      clip=CAST_CLIP_FULL)
    visualise_3d(rays_origins_transformed, rays_dirs_transformed, O, plane)


    axes2 = plt.figure()
    plt.imshow(texture, extent=(0.0,1.0,0.0,1.0), alpha=0.6)
    plt.plot(v, 1-u, '.')
    plt.plot(v6, 1-u6, 'r.')

    # tests `sample_hex()`
    image_loader.sample_hex(u6,v6, texture)

    plt.show()

# an old demo that tests sample_hex and uses Voronoi

if __name__ == "__main__":
    old_demo()
    plt.show()
