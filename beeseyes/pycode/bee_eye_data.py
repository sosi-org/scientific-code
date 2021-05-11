import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from time import sleep

BEES_DATA_FILE = '../hadi/doi_10.5061_dryad.23rj4pm__v1/DataForPlots.mat'

def test_data():

   mat = scipy.io.loadmat(BEES_DATA_FILE)
   # print(mat)
   print()
   print('keys:')
   print(mat.keys())
   '''
   '__header__',
   '__version__',
   '__globals__',
   'CCT_Thresh',
   'IDWMinRangeFacet',
   'baseHeightStep',
   'bordersFromNearest',
   'densityLim',
   'forceHorizontal',
   'forceVertical',
   'fullBeeData',
   'lensT_Thresh',
   'limLargeDensities',
   'limRetinaDist_Thresh',
   'measurmentIntervals',
   'nearestOnWorld',
   'powerP',
   'powerPonSphere',
   'None',
   'stepInFromBorders',
   'stepInInterval',
   'surfaceFitRngTests',
   'useDirectHexAxes',
   'useReverseNormal',
   '__function_workspace__'
   '''
   print('----------------------------------------------------------------')
   fullBeeData = mat['fullBeeData']
   #print(fullBeeData)

   """
   [
     [
       (
         array(['AM_60185'], dtype='<U8'),
         array(['AM'], dtype='<U2'),
         array(['F'], dtype='<U1'),
         array(['Z:\\Analysis\\Amira\\CE\\60185_AM_F\\60185_AM_surfaces_LabelsForMatlab'], dtype='<U65'),
         array([[(array([[2]], dtype=uint8),
         array([[3]], dtype=uint8),
         array([[4]], dtype=uint8),
         array([[1]], dtype=uint8),
         array([[7]], dtype=uint8),
         array([[8]], dtype=uint8),
         array([[9]], dtype=uint8),
         array([[5]], dtype=uint8)
       )
     ]
   ],
   dtype=[
    ('Lens', 'O'), ('Cones', 'O'), ('Retina', 'O'), ('LaminaC', 'O'), ('LensOuterS', 'O'), ('LensInnerS', 'O'), ('RetinaOuterS', 'O'), ('inside', 'O')
   ]
   ),
   array([[ 1.69048e+00,  8.84850e-02, -1.23002e+00,  6.66520e+03],
         [-2.99166e-01,  2.05415e+00, -2.63387e-01,  9.39341e+02],
         ...

   """

   '''
   mat_dict = {}
   mat_dict.update(mat['fullBeeData'])
   mat.close()
   print(mat_dict)
   '''

   f = scipy.io.loadmat(BEES_DATA_FILE)
   a = f['fullBeeData']
   print('version:', f['__version__'])
   print('header', f['__header__'])
   #print('size', f.size)
   #print('dtype', f.dtype)
   #print('shape', f.shape)
   # f.close()

   print(a[0][0][0]) # ['AM_60185']
   #print(a[0][1]) #  big 
   print('----------------------------------------------------------------')

   #print(a)
   big_tuple = a[0,0]
   # a tuple of arrays

   #normals_xyz
   #SphereIntersect
   #HeadeCenter


   print('len(big_tuple)', len(big_tuple)) # 97 !!!
   #print(big_tuple[4]) #indices
   #  [[(array([[2]], dtype=uint8), array([[3]], dtype=uint8), array([[4]], dtype=uint8), array([[1]], dtype=uint8), array([[7]], dtype=uint8), array([[8]], dtype=uint8), array([[9]], dtype=uint8), array([[5]], dtype=uint8))]]

   print(a.dtype) # found it
   print(len(a.dtype))

   # [('BeeID', 'O'), ('BeeSpecies', 'O'), ('BeeSex', 'O'), ('StackFolder', 'O'), ('StackLabels', 'O'), ('LeftEyeTrans', 'O'), ('RightEyeTrans', 'O'), ('AddEyeScale', 'O'), ('VoxelSize', 'O'), ('ITW', 'O'), ('LeftEyeOriginal', 'O'), ('headTrans', 'O'), ('headID', 'O'), ('headStackFolder', 'O'), ('headVoxelSize', 'O'), ('FacetSizeFile', 'O'), ('OrigHexF', 'O'), ('headEyeF', 'O'), ('SamplingF', 'O'), ('histFig', 'O'), ('hexInterpF', 'O'), ('AreaTestF', 'O'), ('testAngF', 'O'), ('testOnEyeF', 'O'), ('testOnEyeFExtra', 'O'), ('testFOV', 'O'), ('testOnWorldF', 'O'), ('testOnWorldFExtra', 'O'), ('LSetF', 'O'), ('LabelsStack', 'O'), ('eyeVolSize', 'O'), ('EyeVolume', 'O'), ('lensVolume', 'O'), ('CCVolume', 'O'), ('retinaVolume', 'O'), ('EyeFrontSurfInds', 'O'), ('borderInds', 'O'), ('EyeFrontSurfSubsTrans', 'O'), ('EyeLength', 'O'), ('mirrorEyeSubs', 'O'), ('HeadCenter', 'O'), ('facetSizes', 'O'), ('facetInds', 'O'), ('facetLocsTrans', 'O'), ('HexagonCoords', 'O'), ('HexLinkOrder', 'O'), ('hexArea', 'O'), ('interpPoints', 'O'), ('samplePointConnections', 'O'), ('numConnections', 'O'), ('normals', 'O'), ('lensThickness', 'O'), ('CCThickness', 'O'), ('RetThickness', 'O'), ('borderNormals', 'O'), ('interpFacetArea', 'O'), ('areaRatio', 'O'), ('projectedAngle', 'O'), ('RadiusUsed', 'O'), ('avgDiameter', 'O'), ('lensSurfaceArea', 'O'), ('NumFacets', 'O'), ('densityFromArea', 'O'), ('fovFilledFromIntegral', 'O'), ('WorldIO', 'O'), ('calInterO', 'O'), ('AxisDensity', 'O'), ('sensitvityApproximation', 'O'), ('expectedSensInt', 'O'), ('avgRadius', 'O'), ('worldRatio', 'O'), ('sphereIntersect', 'O'), ('sphereIntersectBorders', 'O'), ('inFOV', 'O'), ('inFOVRight', 'O'), ('inFOVBorderSubs', 'O'), ('inFOVBino', 'O'), ('frontalPoint', 'O'), ('fovFilledFromPoints', 'O'), ('diameterOnSphere', 'O'), ('curvatureOnSphere', 'O'), ('InterOOnSphere', 'O'), ('AxisDensityOnSphere', 'O'), ('areaProjectionOnSphere', 'O'), ('facetAreaOnSphere', 'O'), ('SensitvityOnSphere', 'O'), ('InterOWorldOnSphere', 'O'), ('InterORatioOnSphere', 'O'), ('lensTOnSphere', 'O'), ('CCTOnSphere', 'O'), ('retTOnSphere', 'O'), ('integratedAreaProj', 'O'), ('integratedAngleDensity', 'O'), ('integratedSensitvity', 'O'), ('fitUsed', 'O'), ('facetError', 'O'), ('htStepUsed', 'O')]

   # normals
   # SphereIntersect
   # sphereIntersectBorders
   # HeadCenter
   # fields = ['normals', 'SphereIntersect', 'sphereIntersectBorders', 'HeadCenter']
   normals = a['normals']
   print('normals len', len(normals))
   print('normals[0]', len(normals[0])) # 8
   print('normals[0,0]', len(normals[0,0])) # 3187
   CnT = len(normals[0]) # 8
   for idx in range(0, CnT):
     normals = a['normals'][0,idx]
     sphereIntersect = a['sphereIntersect'][0,idx]
     sphereIntersectBorders = a['sphereIntersectBorders'][0,idx]
     HeadCenter = a['HeadCenter'][0,idx]
     #print(sphereIntersect)
     #print(sphereIntersectBorders)
     #print(HeadCenter)
     print(a['BeeID'][0,idx], a['BeeSpecies'][0,idx]) # 'AM' 'BT'

'''
  0 ['AM_60185'] ['AM']
  1 ['AM_60186'] ['AM']
  2 ['BT_77970'] ['BT']
  3 ['BT_77971'] ['BT']
  4 ['BT_77966'] ['BT']
  5 ['BT_77967'] ['BT']
  6 ['BT_77973'] ['BT']
  7 ['BT_77974'] ['BT']
'''
def load_relevant(idx):
   f = scipy.io.loadmat(BEES_DATA_FILE)
   a = f['fullBeeData']
   normals = a['normals'][0,idx]
   sphereIntersect = a['sphereIntersect'][0,idx]
   sphereIntersectBorders = a['sphereIntersectBorders'][0,idx]
   HeadCenter = a['HeadCenter'][0,idx]
   BeeID = a['BeeID'][0,idx]
   return (normals, sphereIntersect, sphereIntersectBorders, HeadCenter, BeeID)

from scipy.spatial import SphericalVoronoi

def make_midpoints(polyg, regions_side_count):
    n, maxsides = polyg.shape[:2]
    assert polyg.shape == (n, maxsides, 3)
    assert regions_side_count.shape == (n,)

    return np.nansum(polyg, axis=1) / regions_side_count.astype(polyg.dtype)[:,None]

'''
Usage:
   polyg_d = make_deviations(polyg, midpoints, regions_side_count)
   var_s = np.sum(np.sum(polyg_d * polyg_d ,axis=1),axis=1) / regions_side_count.astype(polyg.dtype)
   sd_s = np.power(var_s, 0.5)
'''
def make_deviations(polyg, midpoints, regions_side_count):
     nv, maxsides = polyg.shape[:2]
     assert polyg.shape == (nv, maxsides, 3)
     assert midpoints.shape == (nv, 3)
     assert regions_side_count.shape == (nv,)

     polyg_d = polyg - midpoints[:,None,:]
     return polyg_d

def eyes_demo():
   '''
     0: 3187
     1: 3250
     2: 2873
     3: 3239
     4: 4300
     5: 4827
     6: 1889
     7: 2105
   '''
   bee_idx = 1

   (normals, sphereIntersect, sphereIntersectBorders, HeadCenter, BeeID) = \
      load_relevant(bee_idx)
   print(BeeID)

   print(normals.shape) # (3250, 3)

   print('HeadCenter', HeadCenter) # [[1627.86042584 1256.85655492  782.86597872]]

   SZ=8.0*1.2 * 3

   ax3d = plt.figure() \
      .add_subplot(
         projection='3d', autoscale_on=True,
         #xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0)
      )

   qv = ax3d.quiver( \
     sphereIntersect[:,0],sphereIntersect[:,1],sphereIntersect[:,2], \
     normals[:,0],normals[:,1],normals[:,2], \
     pivot='tail', length=0.1, normalize=True, color='b'
    )

   # voronoi  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.html

   radius = 1.0
   center = np.array([0, 0, 0])
   sv = SphericalVoronoi(sphereIntersect, radius, center)
   sv.sort_vertices_of_regions()
   ax3d.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
                   c='g')

   sides_l = []
   for region in sv.regions:
      n = len(region)
      # print(n)
      sides_l.append(n)
      for i in range(n):
          polygon = sv.vertices[region]
          # print(polygon.shape) # (n, 3)

          start = sv.vertices[region][i]
          end = sv.vertices[region][(i + 1) % n]
          '''
          result = geometric_slerp(start, end, t_vals)
          ax.plot(result[..., 0],
                  result[..., 1],
                  result[..., 2],
                  c='k')
          '''

   print('regions:', len(sv.regions)) # 3250
   print('eyes:', (sphereIntersect.shape))  # (3250, 3)
   print()

   # sv.regions = sv.regions[:10]

   plt.figure()
   plt.title('Number of sides: Delanuey')
   plt.hist(np.array(sides_l)+0.2, bins=range(2,11))

   print('sv.points', sv.points.shape)  # (3250, 3)
   print('sv.radius', sv.radius) #  1.0
   print('sv.center', sv.center) # [0,0,0]
   print('sv.vertices', sv.vertices.shape) # (6496, 3)
   print('sv.regions', len(sv.regions), '  [0]->  ', sv.regions[0])
   # 3250   [0]->   [1146, 3039, 3040, 1979, 1147, 5258, 5259, 4273]

   # regions_fast, _ = make_fast_regions(sv.regions, maxsides=MAX_SIDES, default=NULL_INDEX)

   regions_fast, regions_side_count = make_fast_regions(sv.regions, maxsides=MAX_SIDES)
   print('regions_fast***', regions_fast.shape)
   print(regions_fast)
   print('regions_side_count', regions_side_count)
   polyg = spread1(sv.vertices, regions_fast, default=0.0)
   print('polyg.shape') # (3250, MAX_SIDES, 3)
   print(polyg.shape)
   print(polyg)

   midpoints = make_midpoints(polyg, regions_side_count)
   print(midpoints)
   print(midpoints.shape) # (3250, 3)

   # standard deviations
   polyg_d = make_deviations(polyg, midpoints, regions_side_count)
   var_s = np.nansum(np.nansum(polyg_d * polyg_d ,axis=1),axis=1) / regions_side_count.astype(polyg.dtype)
   print('var_s', var_s.shape)
   sd_s = np.power(var_s, 0.5)

   plt.figure()
   plt.title('standard deviations')
   plt.hist(sd_s, bins=np.arange(0,3,0.1))
   print('max:', np.max(sd_s)) # 1.914556928682494

   #regions_fast =

   plt.figure()
   ax3d2 = plt.figure() \
      .add_subplot(
         projection='3d', autoscale_on=True,
      )

   ax3d.scatter(midpoints[:,0], midpoints[:,1], midpoints[:,2], marker='.', color='k')
   #§ax3d2.scatter(midpoints[:,0], midpoints[:,1], midpoints[:,2], marker='.', color='k')
   ax3d2.scatter(midpoints[:,0], midpoints[:,1], midpoints[:,2], marker='.', color='k')

   #rf3 = np.tile(regions_fast[:,:,None], (1,1,3))

   if False:
       #rf1 = regions_fast[:,:,None]
       rf3x = sv.vertices[regions_fast, 0]
       rf3y = sv.vertices[regions_fast, 1]
       rf3z = sv.vertices[regions_fast, 2]
       #ommatidia = sv.regions
       #for ommatidium in ommatidia:
       pass



   plt.show()


NULL_INDEX = -1
NULL_VALUE = np.NaN

'''
regions: list of polygon_t
polygon_t: list of int
'''
def make_fast_regions(regions, maxsides, default=NULL_INDEX):
   regions_fast = np.full((len(regions), maxsides), default, dtype=np.intp)
   regions_side_count = np.full((len(regions), ), 0, dtype=np.intp)
   for ri,reg_verts in enumerate(regions):
       nl = min(len(reg_verts), maxsides)
       regions_fast[ri, 0:len(reg_verts)] = reg_verts[:maxsides]
       regions_side_count[ri] = nl
   return regions_fast, regions_side_count

def test1():
   MAX_SIDES = 6 #14 # 6
   regions = [[0,1,2], [1,2,3,4]]
   regions_fast, _ = make_fast_regions(regions, maxsides=MAX_SIDES)
   print('regions_fast', regions_fast)

   #vertices = np.random.randn(10, 3)
   #vertices = np.tile(np.arange(10)[:,None], (1,3)) + 0.5
   #vertices = np.arange(10)[:,None] + 0.5
   #polyg = vertices[regions_fast, :]
   vertices = np.arange(10) + 0.5
   vertices[-1] = NULL_VALUE
   polyg = vertices[regions_fast]
   print('polyg', polyg)

'''
    Replaces empty ones with `default` using `-1` trick.
'''
def spread1(vertices, regions_fast, default=NULL_VALUE):
   # assert len(vertices.shape) == 2
   last_row = np.zeros((1, *vertices.shape[1:]), dtype=vertices.dtype)
   vertices_augmented = np.concatenate((vertices, last_row), axis=0)
   vertices_augmented[-1] = default  # [-1,:,:,...] = default
   polyg = vertices_augmented[regions_fast]
   # polyg.shape: (3250, MAX_SIDES, 3)
   return polyg

def test2():
    MAX_SIDES = 6
    regions = [[0,1,2], [1,2,3,4]]
    regions_fast, _ = make_fast_regions(regions,maxsides=MAX_SIDES)
    print(regions_fast)
    vertices = np.arange(10) + 0.5
    polyg = spread1(vertices, regions_fast)
    print(polyg)


def load_eyes():
   '''
     0: 3187
     1: 3250
     2: 2873
     3: 3239
     4: 4300
     5: 4827
     6: 1889
     7: 2105
   '''
   bee_idx = 1

   (normals, sphereIntersect, sphereIntersectBorders, HeadCenter, BeeID) = \
      load_relevant(bee_idx)
   print(BeeID)

   print(normals.shape) # (3250, 3)

   print('HeadCenter', HeadCenter) # [[1627.86042584 1256.85655492  782.86597872]]
   return (sphereIntersect, normals, HeadCenter, BeeID)

''' Takes, from a given list, elements based on a boolean numpy array'''
def my_index(l1, np_bool):
    return [l1[i] for i in np_bool.tolist()]

# voronoiPoints
'''
   returns (
      corner points,  # haxagonals corner points
      regions, # (list of list [][] of)  indices corners off points
      normals at corner points
   )
'''
def polygon_points(sphereIntersect, normals):
   # sphereIntersect[:,0],sphereIntersect[:,1],sphereIntersect[:,2]
   # normals[:,0],normals[:,1],normals[:,2]

   radius = 1.0
   center = np.array([0, 0, 0])
   sv = SphericalVoronoi(sphereIntersect, radius, center)
   sv.sort_vertices_of_regions()
   print(dir(sv))
   num_vertices = sv.vertices.shape[0]

   #sv.regions = sv.regions[:10]
   #normals = normals[:10]

   three_corners = three_neighbnours(sv.points, sv.regions, num_vertices)
   a3 = three_corners_claculat(sv.points, three_corners, num_vertices)
   print(sv.vertices)
   print(a3-sv.vertices)  # not precise though

   assert sv.points.shape == normals.shape
   normals_at_corners = three_corners_claculat(normals, three_corners, num_vertices)

   #ax3d = ax3dCreate()
   #print('111')
   ##visualise_all(ax3d, sv.vertices, normals)
   #print('2222')
   ##visualise_all(ax3d, sphereIntersect, normals_at_corners)

   #visualise_all(ax3d, sv.vertices, normals_at_corners, 'r')  # corners
   #print('2222')
   #visualise_all(ax3d, sphereIntersect, normals, 'b') # centers
   #plt.show()

   areas = sv.calculate_areas()


   # select
   if False:
   #AREA_THRESHOLD = 0.050
       AREA_THRESHOLD = 0.01/2/4

       w = areas < AREA_THRESHOLD
       sv_regions_sel = my_index(sv.regions, w)
       '''
       unionvi = np.array(list(set().union(*sv_regions_sel)))
       sv.vertices = sv.vertices[unionvi, :]
       normals_at_corners = normals_at_corners[unionvi, :]
       '''
   else:
       sv_regions_sel = sv.regions

   return sv.vertices, sv_regions_sel, normals_at_corners, areas

'''
Three neighbours of each
  points = centers (one for each region/simplex)   (centers)
  vertices = corners (of each vertex of region/simplex)  (corners)
  # regions[region index] = [ point index, ]
'''
def three_neighbnours(points, regions, num_vertices):
   MAX_EDGES = 3 # three regions/centers for each corner
   three = np.full((num_vertices, MAX_EDGES), -1, dtype=np.intp)
   regions_side_count = np.full((num_vertices, ), 0, dtype=int)
   for ri,reg_verts in enumerate(regions):
       nl = len(reg_verts)
       for vi in reg_verts:
           ci = ri
           three[vi, regions_side_count[vi]] = ci
           regions_side_count[vi] += 1
   print(three)
   assert np.sum(regions_side_count-3) == 0, 'Must have exactly three points for each vertex'
   # return three[:,;3]
   # three_corners
   return three

''' creates corners again '''
def three_corners_claculat(points, three, num_vertices):
   cnx3x3 = np.reshape(points[three.ravel(),:], (num_vertices,3,-1))
   a = np.sum(cnx3x3, axis=1) / 3.0
   return a

'''
   (3250, MAX_SIDES, 3)
'''
def ommatidia_polygons_fast_representation(sv_vertices, sv_regions, maxsides, default=np.NaN):
   # Each row is (the points on) a full polygon, padding empty elements with default value
   regions_fast, regions_side_count = make_fast_regions(sv_regions, maxsides=maxsides)

   # Delanuey polygons:
   polyg = spread1(sv_vertices, regions_fast, default=default)
   # polyg  # (3250, MAX_SIDES, 3)

   # print(regions_fast.shape, regions_side_count.shape, polyg.shape)
   # regions_fast: (3250, 6)
   # regions_side_count: (3250,)
   # polyg: (3250, 6, 3)
   #return regions_fast, regions_side_count, polyg
   return polyg, regions_side_count

def ax3dCreate():
   SZ=8.0*1.2 * 3
   ax3d = plt.figure() \
      .add_subplot(
         projection='3d', autoscale_on=True,
         #xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0)
         #xlim=(0.7, 1.0),  ylim=(-0.3, 0.15),zlim=(-0.2, 0.25)
         xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0)
      )
   ax3d.set_xlabel('x')
   ax3d.set_ylabel('y')
   ax3d.set_zlabel('z')
   return ax3d

def visualise_all(ax3d, X, N, color='b'):

   print('X.shape', X.shape)
   print('N.shape', N.shape)
   qv = ax3d.quiver( \
     X[:,0], X[:,1], X[:,2], \
     N[:,0],N[:,1],N[:,2], \
     pivot='tail', length=0.1/10, normalize=True, color=color
    )
   #plt.show()

'''
calculting norms at corners of hexagonals by interpolation (normals_at_corners)
returns (3250, MAX_SIDES, 3)
'''
def ommatidia_polygons():

   (sphereIntersect, normals, HeadCenter, BeeID) = \
      load_eyes()

   #sphereIntersect = sphereIntersect[:100]
   #normals = normals[:100]
   #sphereIntersect = np.random.randn(150,3)
   #sphereIntersect = sphereIntersect / np.linalg.norm(sphereIntersect, axis=1, ord=2)[:,None]
   #normals = sphereIntersect / np.linalg.norm(sphereIntersect, axis=1, ord=2)[:,None]

   (sv_vertices, sv_regions, normals_at_corners, areas) = \
      polygon_points(sphereIntersect, normals)

   return sv_vertices, sv_regions, normals, normals_at_corners, sphereIntersect, areas


def main():
   #test_data()
   #main_data()
   #exit()
   #test1()
   #test2()
   #exit()
   #eyes_demo()

   ommatidia_polygons_fast_representation(*ommatidia_polygons()[:2])

if __name__ == "__main__":
    main()
