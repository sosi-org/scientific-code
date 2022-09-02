from visualise_plane import visualise_plane

def visualise_3d(rays_origins, rays_dirs, points_xyz, plane):
    SZ=8.0*1.2 * 3
    ax3d = plt.figure().add_subplot(projection='3d', autoscale_on=False,
       xlim=(0, +SZ), ylim=(0, +SZ), zlim=(-SZ/2.0, +SZ/2.0))
    # ax3d = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    #ax3d.set_aspect('equal')
    #ax3d.set_aspect(1)

    # only a single hex being casted
    qv = ax3d.quiver( \
     rays_origins[:,0],rays_origins[:,1],rays_origins[:,2], \
     rays_dirs[:,0],rays_dirs[:,1],rays_dirs[:,2], \
     pivot='tail', length=1.0, normalize=True, color='r'
    )
    '''
    ax3.quiverkey(qv, 0.9, 0.9, 1, r'$xxxx$', labelpos='E',
               coordinates='figure')
    '''

    # All ommatidia
    ax3d.scatter(points_xyz[:,0],points_xyz[:,1],points_xyz[:,2], marker='.')

    visualise_plane(ax3d, plane, color='g')

