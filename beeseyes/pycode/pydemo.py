from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from bee_eye_data import ommatidia_polygons

HEX6 = 6


# https://en.wikipedia.org/wiki/Blue_flower#/media/File:Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpg
# https://en.wikipedia.org/wiki/Blue_flower
BLUE_FLOWER = "../art/256px-Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpeg"

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
def ray_cast(U,V,C0, D,O):
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
   # t = (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx + ux*vy*z0 - ux*vz*y0 - uy*vx*z0 + uy*vz*x0 + uz*vx*y0 - uz*vy*x0) / denom

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
    [pu,pv] = np.meshgrid(np.linspace(0,1,50),np.linspace(0,1,50))
    pu = pu.ravel()
    pv = pv.ravel()
    U = tuple3_to_np(plane['U'])
    V = tuple3_to_np(plane['V'])
    C0 = tuple3_to_np(plane['C0'])
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

def demo_lattice_eyes():
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

    return points_xyz, normals_xyz, rays_origins, rays_dirs

def raycastOmmatidium(eye_points, rays_dirs, bee_R, bee_pos, plane):
   O = np.dot(bee_R, eye_points.T).T + bee_pos
   D = np.dot(bee_R, rays_dirs.T).T
   (u,v) = ray_cast(plane['U'],plane['V'],plane['C0'], D,O)
   return O, D, (u,v)

class Plane:
    def __init__(self):
        pass


class BeeHead:
    def __init__(self):
        bee = {
          'pos': (15.0, 15.0, -10.0),
          #'u': (15.0, 15.0, -10.0),
          'u': (1, 0.1,-0.2), # Bee's right hand  (1,0,-0.2)
          'v': (0,1,0),  # Bee's top
        }
        bee_R = rotation_matrix(bee)

        bee_pos = tuple3_to_np(bee['pos'])
        # self.np = {}

        self.R = bee_R
        self.pos = bee_pos


def main():

    points_xyz, normals_xyz, rays_origins, rays_dirs \
       = demo_lattice_eyes()

    texture = load_image(BLUE_FLOWER)



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





    beeHead = BeeHead()
    beeHead.pos
    beeHead.R

    # O = points_xyz / 7.0 * 0.1/70  # in cm

    EYE_SIZE = 0.1*10 # cm
    eye_points = points_xyz * EYE_SIZE \
      + 0 * tuple3_to_np((15.0,15.0,-10.0))  # in cm



    O,D,(u,v) = raycastOmmatidium(eye_points, normals_xyz, beeHead.R, beeHead.pos, plane)

    # print(texture.shape, 'sss') #  (192, 256, 3)

    #visuaise_3d(rays_origins, rays_dirs, points_xyz, plane)

    rays_origins_e = rays_origins * EYE_SIZE
    # visuaise_3d(O, D, O, plane)


    (rays_origins_transformed, rays_dirs_transformed, (u6,v6)) = raycastOmmatidium(rays_origins_e, rays_dirs, beeHead.R, beeHead.pos, plane)
    visuaise_3d(rays_origins_transformed, rays_dirs_transformed, O, plane)


    axes2 = plt.figure()
    plt.imshow(texture, extent=(0.0,1.0,0.0,1.0), alpha=0.6)
    #plt.imshow(extent=(0.0,1.0,0.0,1.0), url=BLUE_FLOWER)
    #axes2.hold(True)
    plt.plot(u,v, '.')
    plt.plot(u6,v6, 'r.')

    sample_hex(u6,v6, texture)

    plt.show()

def xxx5():
    ommatidia_polygons1, regions_side_count = \
       ommatidia_polygons()
    # (3250, MAX_SIDES, 3)

    print('.')

xxx5()

main()
