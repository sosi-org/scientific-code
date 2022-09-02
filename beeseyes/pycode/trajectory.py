import numpy as np

from path_data import load_trajectory_cached

def trajectory_provider(positions_xls_file):
    """
    The .xls spreadsheet containing the positions in a flight path
    """

    bee_traj = load_trajectory_cached(positions_xls_file)

    def trajectory_transformation():
        '''
        Affine transormation for correcting the mismtch between the units in Excel file and assumed orientation of the plane.
        Do do, apply the reverse of this to the plane only.
        The plane's dimentions are correct. But its orientation.
        Todo:
          Just rescale (nd not rotate) the trajectory data.
          But apply the reverse of this transform to the plane.
        '''
        maxy = np.array([7,+372,272])[None,:]
        M = np.eye(3)
        # '''
        M = M * 0
        # M[to,from]
        _X = 0
        _Y = 1
        _Z = 2
        M[_X ,_X] = 1.0
        M[_Y ,_Z] = -1.0
        M[_Z ,_Y] = -1.0 # should be negative

        # z,x -> (y,x)
        # z,x,y -> (y,x,z)
        # x,y,z -> (x,z,y)
        M = M / 10.0

        return M, maxy

    # Apply a linear transformation on bee trajectory
    M, maxy = trajectory_transformation()

    print(bee_traj._RWSmoothed)
    #print('bee_traj', bee_traj)
    frame_times = bee_traj['fTime']
    bee_path = np.dot( bee_traj['RWSmoothed'] - maxy, M.T) + np.array([0,0,0])[None,:]
    bee_directions = bee_traj['direction']

    #return (M, bee_head_pos, bee_direction)
    #return (M, bee_path, bee_directions, frame_times)
    return (bee_path, bee_directions, frame_times)

