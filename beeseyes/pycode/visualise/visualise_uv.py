import matplotlib.pyplot as plt

def visualise_uv(u,v, u_few, v_few, texture, uv_rgba=None, title=None, fig=None):
    # (u,v) visualisation on plane (pixels)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    tt = texture
    # tt = np.transpose(texture, axes=(1,0,2))
    plt.imshow(tt, extent=(0.0,1.0,0.0,1.0), alpha=0.6) #, origin='lower')
    #plt.plot(u, v, '.', facecolors=uv_rgba)
    plt.scatter(v, 1-u, marker='.', facecolors=uv_rgba)
    if (u_few is not None) and (v_few is not None):
       plt.plot(v_few, 1-u_few, 'o', color='r')
    #plt.plot(u6,v6, 'r.')
    plt.xlabel('u')
    plt.ylabel('v')
    '''
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    if title is not None:
        ax.set_title(title)
    '''

    # https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
    return ax
