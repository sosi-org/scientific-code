
import matplotlib.pyplot as plt

# SZ=8.0*1.2 * 3 == 28,8
def ax3dCreate(SZ=28.8):
   ax3d = None
   if SZ is None:
      ax3d = plt.figure() \
       .add_subplot(projection='3d', autoscale_on=True)
   else:
      ax3d = plt.figure() \
       .add_subplot(
         projection='3d', autoscale_on=True,
         #xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0)
         #xlim=(0.7, 1.0),  ylim=(-0.3, 0.15),zlim=(-0.2, 0.25)
         xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0)
      )
   ax3d.set_xlabel('x')
   ax3d.set_ylabel('y')
   ax3d.set_zlabel('z')
   return ax3d

