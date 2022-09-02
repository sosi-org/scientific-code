

def visualise_quiver(ax3d, X, N, color='b'):

   assert X.shape == N.shape
   assert X.shape[0] == N.shape[0]

   qv = ax3d.quiver( \
     X[:,0], X[:,1], X[:,2], \
     N[:,0],N[:,1],N[:,2], \
     pivot='tail', length=0.1/10, normalize=True, color=color
    )
