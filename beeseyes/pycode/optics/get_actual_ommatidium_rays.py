from cte import DIM3
import numpy as np

def getActualOmmatidiumRays(eye_points, rays_dirs, bee_R, bee_pos, bee_eye_sphere_size_cm):
   n = rays_dirs.shape[0]
   assert eye_points.shape == (n, DIM3)
   assert rays_dirs.shape == (n, DIM3)
   assert bee_pos.shape == (1,DIM3)
   assert bee_R.shape == (DIM3, DIM3)
   print(type(bee_eye_sphere_size_cm))
   print((bee_eye_sphere_size_cm))
   assert np.isscalar(bee_eye_sphere_size_cm)

   print('@bee_eye_sphere_size_cm', bee_eye_sphere_size_cm)
   O = np.dot(bee_R, (bee_eye_sphere_size_cm * eye_points).T).T + bee_pos
   D = np.dot(bee_R, rays_dirs.T).T
   return O,D
