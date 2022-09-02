import numpy as np

from cte import CAST_CLIP_FULL, CAST_CLIP_NONE, CAST_CLIP_FRONT

'''
     Based on `derive_formulas.py`

     Uncasted are returned as NaN
'''
def ray_cast(
              U : tuple[float, float, float],
              V : tuple[float, float, float],
              C0 :tuple[float, float, float],
              D,
              O,
              clip #clip:bool=True
            ):
   n = D.shape[0]
   assert D.shape == (n,3)
   assert O.shape == (n,3)
   assert len(C0) == 3
   assert len(U) == 3
   assert len(V) == 3

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
   t = (-ox*uy*vz + ox*uz*vy + oy*ux*vz - oy*uz*vx - oz*ux*vy + oz*uy*vx + ux*vy*z0 - ux*vz*y0 - uy*vx*z0 + uy*vz*x0 + uz*vx*y0 - uz*vy*x0) / denom

   if clip == CAST_CLIP_FULL:
       MIN_A = -1.9
       MIN_B = -1.9
       MAX_A = +2.9
       MAX_B = +2.9

       if False:
        MIN_A = 0.0
        MIN_B = 0.0
        MAX_A = +1.0
        MAX_B = +1.0

       w = np.logical_and(t > 0,   a > MIN_A)
       w = np.logical_and(w,   b > MIN_B)
       w = np.logical_and(w,   a < MAX_A)
       w = np.logical_and(w,   b < MAX_B)
       #a = a[w]
       #b = b[w]
       #t = t[w]
       not_w = np.logical_not(w)
       a[not_w] = np.NaN
       b[not_w] = np.NaN
       t[not_w] = np.NaN
   elif clip == CAST_CLIP_FRONT:
       w = t > 0
       not_w = np.logical_not(w)
       a[not_w] = np.NaN
       b[not_w] = np.NaN
       t[not_w] = np.NaN
   elif clip == CAST_CLIP_NONE:
       pass
   else:
       raise

   #plane = UV*ab + C0
   #ray = O + t * D
   #plane - ray == 0
   #castedPoints = O + D * t

   return (a,b),t
