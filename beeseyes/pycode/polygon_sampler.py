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


GRID_FR = 0.1*0.1
meshgridx0, meshgridy0 = np.meshgrid(
  np.arange(0,1.0, 1.0*GRID_FR),
  np.arange(0,1.0, 1.0*GRID_FR),
)
grid_xy0 = np.array([meshgridx0.ravel(), meshgridy0.ravel()]).T

def create_samples_region(count, uv, region):
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

# def sample_poly_region(uv, regions, texture):


def run_tests():
  #import image_loader
  #image_loader.load_image_withsize
  uv = np.array([[0,1],[0.4,0.8],[0.8, 0.5],[1,0.3],[0,-1],[-1,0],[-0.3,0.3]])
  assert uv.shape[1] == 2
  xy = create_samples_region(100, uv, [0,1,2,3,4,5,6])
  import matplotlib.pyplot as plt
  plt.plot(xy[:,0], xy[:, 1], '.')
  plt.show()
  print(xy)
  print(xy.shape) # 34 points
  # 3208 / 10000

if __name__ == "__main__":
    run_tests()
