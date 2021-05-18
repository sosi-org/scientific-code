from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def array_minmax(x):
    mn = np.min(x)
    mx = np.max(x)
    md = (mn + mx)/2.0
    (mn,mx) = ((mn-md)*1.2+md, (mx-md)*1.2+md)
    return (mn, mx)

def create_samples(count, uv_poly):
    #polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon = Polygon(uv_poly)
    meshgrid = np.meshgrid(np.)
    polyon
    TBC
