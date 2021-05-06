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
   plt.figure()
   plt.hist(np.array(sides_l)+0.2, bins=range(2,11))
   plt.show()

def main():
   #test_data()
   #main_data()
   #exit()
   eyes_demo()

main()