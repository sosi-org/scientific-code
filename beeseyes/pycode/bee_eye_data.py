import numpy as np
import matplotlib.pyplot as plt

def main_data():
   import scipy.io
   mat = scipy.io.loadmat('../hadi/doi_10.5061_dryad.23rj4pm__v1/DataForPlots.mat')
   # print(mat)
   print()
   print()
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
   print(mat['fullBeeData'][1])
   normals_xyz
   #SphereIntersect
   HeadeCenter 

main_data()
#exit()
