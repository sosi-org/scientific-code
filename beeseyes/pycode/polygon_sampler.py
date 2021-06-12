import math
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely

def array_minmax(x):
    mn = np.min(x)
    mx = np.max(x)
    md = (mn + mx)/2.0
    (mn,mx) = ((mn-md)*1.2+md, (mx-md)*1.2+md)
    return (mn, mx)

def create_samples(count, uv_poly):
    print(uv_poly)
    #polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    a = [(uv_poly[i,0], uv_poly[i,1]) for i in range(uv_poly.shape[0])]
    print(a)
    polygon = Polygon(a)
    print(polygon)
    x0,x1 = array_minmax(uv_poly[:,0])
    y0,y1 = array_minmax(uv_poly[:,1])
    GRID_FR = 0.1/10
    meshgridx, meshgridy = np.meshgrid(
      np.arange(x0,x1, (x1-x0)*GRID_FR),
      np.arange(y0,y1, (y1-y0)*GRID_FR),
    )
    print(meshgridx.shape)
    print(meshgridy.shape)
    grid_xy = np.array([meshgridx.ravel(), meshgridy.ravel()]).T
    print(grid_xy.shape)
    print(polygon.within(Point(0,0)))
    print(polygon.contains(Point(0,0)))
    #print(polygon.contains(grid_xy))
    #print(polygon.contains([(uv_poly[i,0], uv_poly[i,1]) for i in range(uv_poly.shape[0])]))
    # TBC
    #Shapely.
    #shapely.
    points = [Point(grid_xy[i,0], grid_xy[i,1]) for i in range(grid_xy.shape[0])]
    #polygon.within(points)
    which = [polygon.contains(Point(grid_xy[i,0], grid_xy[i,1])) for i in range(grid_xy.shape[0])]
    return grid_xy[which,:]


def create_master_grid(xa, ya):
  #GRID_FR = 0.1 * 0.5 # *0.1
  meshgridx0, meshgridy0 = np.meshgrid(xa, ya)
  grid_xy0 = np.array([meshgridx0.ravel(), meshgridy0.ravel()]).T
  return grid_xy0

master_grids = {}

class G:
    def __init__(self):
      #self.__x = x
      pass

def init_grid(key, GRID_FR):
  #GRID_FR = 0.1 * 0.5 # *0.1
  obj = G() #{}
  xa = np.arange(0,1.0, 1.0*GRID_FR)
  ya = np.arange(0,1.0, 1.0*GRID_FR)
  obj.grid_xy0 = create_master_grid(xa,ya)
  master_grids[key] = obj

def create_samples_region(uv, region, grid_xy0):
  #return create_samples(count, uv[region,:])
    # const

    uv_poly = uv[region,:]
    a = [(uv_poly[i,0], uv_poly[i,1]) for i in range(uv_poly.shape[0])]

    polygon = Polygon(a)

    x0,x1 = array_minmax(uv_poly[:,0])
    y0,y1 = array_minmax(uv_poly[:,1])

    grid_xy = grid_xy0.copy()
    grid_xy[:,0] = grid_xy0[:,0]*(x1-x0)+x0
    grid_xy[:,1] = grid_xy0[:,1]*(y1-y0)+y0
    # meshgridy = meshgridx0*(x1-x0)+x0

    which = [polygon.contains(Point(grid_xy[i,0], grid_xy[i,1])) for i in range(grid_xy.shape[0])]
    return grid_xy[which,:]

def poly_region_points(uv, regions):
  KEY = '0'
  grid_xy0 = master_grids[KEY].grid_xy0
  l = []
  for i in range(len(regions)):
    region_xy = create_samples_region(uv, regions[i], grid_xy0)
    l.append(region_xy)
  gxy =  np.concatenate(l, axis=0)
  plot_scatter_uv(gxy)
  return gxy

def sample_poly_region(uv, regions, texture):
  poly_region_points
  exit()

def create_samples_region1(uv, region, grid_xy0):
    uv_poly = uv[region,:]
    a = [(uv_poly[i,0], uv_poly[i,1]) for i in range(uv_poly.shape[0])]
    a = filter(lambda t: not math.isnan(t[0]), a)
    # What to do if it is partially outside?

    polygon = Polygon(a)

    x0,x1 = array_minmax(uv_poly[:,0])
    y0,y1 = array_minmax(uv_poly[:,1])

    grid_xy = grid_xy0.copy()
    grid_xy[:,0] = grid_xy0[:,0]*(x1-x0)+x0
    grid_xy[:,1] = grid_xy0[:,1]*(y1-y0)+y0
    # meshgridy = meshgridx0*(x1-x0)+x0

    which = [polygon.contains(Point(grid_xy[i,0], grid_xy[i,1])) for i in range(grid_xy.shape[0])]
    return grid_xy[which,:]

def sample_colors_polygons(uv, regions, texture):

  xa = np.arange(0, texture.shape[0], 1.0)
  ya = np.arange(0, texture.shape[1], 1.0)
  grid_xy0 = create_master_grid(xa,ya)

  l = []
  for i in range(len(regions)):
    region_xy = create_samples_region1(uv, regions[i], grid_xy0)
    l.append(region_xy)
    plot_scatter_uv(l[-1], False)
  plt.show()
  dfgdkj
  exit()
  # poly_region_points(uv, regions)
  
  


def plot_scatter_uv(xy, show=True):
  import matplotlib.pyplot as plt
  plt.plot(xy[:,0], xy[:, 1], '.')
  if show:
    plt.show()

def run_tests():
  GRID_FR = 0.1 * 0.5 # *0.1
  init_grid('0', GRID_FR)

  KEY = '0'
  grid_xy0 = master_grids[KEY].grid_xy0

  #import image_loader
  #image_loader.load_image_withsize
  uv = np.array([[0,1],[0.4,0.8],[0.8, 0.5],[1,0.3],[0,-1],[-1,0],[-0.3,0.3]])
  assert uv.shape[1] == 2
  xy = create_samples_region(uv, [0,1,2,3,4,5,6], grid_xy0)
  plot_scatter_uv(xy)
  print(xy)
  print(xy.shape) # 34 points
  # 3208 / 10000

  # test 2
  import image_loader
  BLUE_FLOWER = "../art/256px-Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpeg"
  texture1, physical_size1, dpi1 = image_loader.load_image_withsize(BLUE_FLOWER,
      sample_px=200, sample_cm=10.0)

  uv = np.random.rand(*(100,2))*100

  regions = [[0,1,2,3], [3,4,6,7], [5,6,7,8,9]]
  test1_points = poly_region_points(uv, regions)
  rgba = sample_poly_region(uv, regions, texture1)
  print(rgba)

if __name__ == "__main__":
    run_tests()
