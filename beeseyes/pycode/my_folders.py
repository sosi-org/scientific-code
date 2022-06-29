import pathlib

#pathlib.Path(__file__).parent.resolve()
#POSITIONS_XLS = CURRENT_PATH + '/Setup/beepath.xlsx'

def base_folder():
    # CURRENT_PATH = '/Users/a9858770/cs/scientific-code/beeseyes'
    CURRENT_PATH = str(pathlib.Path(__file__).parent.parent.resolve())
    # POSITIONS_XLS = CURRENT_PATH + '/Setup/beepath.xlsx'
    # POSITIONS_XLS = CURRENT_PATH + '/data/beepath.xlsx'
    return CURRENT_PATH

# Setup of experiment
def get_setup_path(file_basename):
    return base_folder() + '/Setup/' + file_basename

# Image files from public domain
def get_art_path(file_basename):
    # return base_folder() + '/../art/' + file_basename
    return base_folder() + '/art/' + file_basename

# Files given directly by Hadi
def get_data_path(file_basename):
    return base_folder() + '/hadi/' + file_basename

# https://www.dropbox.com/home/lbg-macbook/cs/scientific-code/beeseyes/Setup
