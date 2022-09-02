import numpy as np
from optics.geom_helpers import tuple3_to_np

# Old usage used Rotation:
# from scipy.spatial.transform import Rotation

'''
BeeHead: in fact, Bee Eye
Orientation of the look of the Bee
The camera
The head direction
'''
class BeeHead:
    def __init__(self):
        #bee = {
        #  'pos': (15.0, 15.0, -10.0/3),
        #  #'u': (15.0, 15.0, -10.0),
        #  'u': (1, 0.1,-0.2), # Bee's right hand  (1,0,-0.2)
        #  'v': (0,1,0),  # Bee's top
        #}
        #
        # bee_R = rotation_matrix(bee)
        #bee_R = np.eye(3)

        #bee_R = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        #print('bee_R.shape', bee_R.shape)
        #if matrix1 is not None:
        #    bee_R = np.dot( bee_R, matrix1.T)

        #bee_pos = tuple3_to_np(bee['pos'])
        # self.np = {}

        #self.R = bee_R
        #self.pos = bee_pos
        self.R = None
        self.pos = None

    def set_eye_position(self, eye_pos):
        print(eye_pos.shape, '<<<<')
        self.pos = eye_pos[None,:]
        assert self.pos.dtype == np.dtype(float)
        assert self.pos.shape == (1,3)


    def set_direction(self, dirc, eye_size_cm):
        """
        dirc: bee_direction
        eye_size_cm: eye_sphere_size_cm
        The size of the unit sphere is multiplied by this `eyeSphereSize`
        beeHead.R will be the matrix

        note:
        `M` and `head_transformation` are later ignored
        """
        #bee_R = Rotation.from_rotvec(dirc, 180, degrees=True).as_matrix()
        #self.R = bee_R
        #assert self.R.dtype == np.dtype(float)
        #assert self.R.shape == (3,3)

        dirc=dirc.ravel()
        n=np.linalg.norm(dirc,axis=0)
        #print('n>',n)
        dirc=dirc/n

        up = tuple3_to_np( (0,1,0) )
        bee_right = np.cross(dirc, up)
        bee_right=bee_right.ravel()
        n=np.linalg.norm(bee_right,axis=0)
        #print('n>',n)
        bee_right=bee_right/n
        # `bee_right` ⟂ `up`
        # `bee_right` ⟂ `dirc`
        # But `up` (not) ⟂̸ `dirc`

        u = np.cross(bee_right, dirc).ravel()
        u=u.ravel()
        n=np.linalg.norm(u,axis=0)
        #print('n>',n)
        u=u/n
        # u,bee_right,dirc  are all mutually perpendicular

        #print(u.shape)
        #print(bee_right.shape)
        #print(dirc.shape)
        #print('----')

        # (bee_right, u, dirc) are analogus to:  (1,0,0), (0,1,0), (0,0,1)
        B = np.concatenate((bee_right[:,np.newaxis], u[:,np.newaxis], dirc[:,np.newaxis]), axis=1)
        # B * x = new physical vector in Eulidian basis
        # R * (1,0,0).T = bee_right .T
        # R * (0,1,0).T = u .T
        # R * (0,0,1).T = dirc .T
        # R * I = (bee_right.T, u.T, dirc.T)
        self.R = B
        assert self.R.dtype == np.dtype(float)
        assert self.R.shape == (3,3)
        print(self.R)
        n=np.linalg.norm(self.R,axis=(0))
        assert np.all(np.isclose(n, (1,1,1)))

        self.eye_size_cm = eye_size_cm

