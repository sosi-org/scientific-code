
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

