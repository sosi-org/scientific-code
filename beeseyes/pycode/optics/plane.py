from .ray_cast import ray_cast
class Plane:

    def __init__(self, physical_size_u=30, physical_size_h=30, C0_pos=(0,0,0)):
       # centimeters
       # todo: rename U -> A, a->uz
       plane = {
          'U': (float(physical_size_u),0,0),
          'V': (0,float(physical_size_h),0),
          'C0': C0_pos,
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
    def raycastRaysOnPlane(self, O, D, clip, return_casted_points=False):
        return _raycastRaysOnPlane(O, D, self, clip, return_casted_points=return_casted_points)


def _raycastRaysOnPlane(O, D, plane, clip, return_casted_points=False):
    (u,v),t = ray_cast(plane.U, plane.V, plane.C0, D,O, clip=clip)
    if return_casted_points:
       assert t.shape == D.shape[:1]
       casted_points = O + D * t[:, None]
       return (u,v), casted_points
    else:
       return (u,v)
