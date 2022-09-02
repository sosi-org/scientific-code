import numpy as np

def tuple3_to_np(pxyz):
   x,y,z = pxyz
   return np.array([x,y,z], dtype=float)[None, :]

def normalise_np(pxyz):
   return pxyz * 1.0 / (np.linalg.norm(pxyz, axis=0, keepdims=True) + 0.0000001)

# not used
def rotation_matrix(bee):
   # right hand
   U = tuple3_to_np(bee['u'])
   # top
   V = tuple3_to_np(bee['v'])
   U = normalise_np(U)
   # straight ahead: into the plane
   W = -np.cross(U, V)
   W = normalise_np(W)
   # again re-create the "top" direction
   V = np.cross(U, W)
   V = normalise_np(V)
   # return np.eye(3)
   # [U,V,W] concat as rows
   return np.concatenate((U,V,W), axis=0)
