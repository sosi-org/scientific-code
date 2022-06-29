import pathlib

def base_folder():
    # old path:
    # CURRENT_PATH = '/Users/a9858770/cs/scientific-code/beeseyes'
    CURRENT_PATH = str(pathlib.Path(__file__).parent.parent.resolve())
    return CURRENT_PATH

# Setup of experiment
def get_setup_path(file_basename):
    return base_folder() + '/Setup/' + file_basename

# Image files from public domain
def get_art_path(file_basename):
    return base_folder() + '/art/' + file_basename

# Files given directly by Hadi
def get_data_path(file_basename):
    return base_folder() + '/hadi/' + file_basename

"""
            Data files catalogue:
"""

"""
Data file: doi_10.5061_dryad.23rj4pm__v1/DataForPlots.mat
    Folder: ./hadi

    From paper:
       Taylor, Gavin J. et al. (2019),
       Data from: Bumblebee visual allometry results in locally improved resolution and globally improved sensitivity,
       Dryad, Dataset, https://doi.org/10.5061/dryad.23rj4pm

    Doenload:   https://datadryad.org/stash/dataset/doi:10.5061/dryad.23rj4pm
        includes: Data description
        companion matlab code: https://github.com/gavinscode/compound-eye-plotting-elife (aka BeeLobulaModel_2017 )
                    https://www.dropbox.com/home/lbg-macbook/cs/scientific-code/beeseyes/hadi/BeeLobulaModel_2017

    The doi_10.5061_dryad.23rj4pm__v1.zip:
        ./DataForPlots.mat    * (used)
        ./Bee_Eye_Data.zip
        ./README_for_Bee_Eye_Data.rtf
        ./Facet_Sizes.zip

"""

"""
Files only available from Dropbox:
Setup files: origin:
    Folder: https://www.dropbox.com/home/lbg-macbook/cs/scientific-code/beeseyes/Setup

    https://www.dropbox.com/s/qrn79rfqhwizbdr/IMG_2872.MOV.BkCorrectedPerspCroppedColourContrast.png?dl=0

Provided by hadi:
    flower-sept.png
    beepath.xlsx

    IMG_2872.MOV.BkCorrectedPerspCroppedColourContrast.png

    pinkRandomDots.png



"""

"""
Arts:

    BLUE_FLOWER:
        # https://en.wikipedia.org/wiki/Blue_flower#/media/File:Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpg
        # https://en.wikipedia.org/wiki/Blue_flower
        #BLUE_FLOWER = "../art/256px-Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpeg"
        BLUE_FLOWER = my_folders.get_art_path("256px-Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpeg")
        BLUE_FLOWER_DPI_INFO = {_PIXELS: 200, _CM: 10.0}


    NEW_FLOWER:
        flower-sept.png

        #FLOWER_XY = '/Users/a9858770/Documents/xx/3bebe3b139b7e0e01573faabb4c92934.jpeg'
        #BEE_CARTOON = '/Users/a9858770/Documents/bee-walt-Spike_art.PNG.png'
        #NEW_FLOWER = '/Users/a9858770/cs/scientific-code/beeseyes/Setup/flower-sept.png'
        NEW_FLOWER = my_folders.get_setup_path('flower-sept.png')


    PINK_WALLPAPER:
        # pink texture
        # PINK_WALLPAPER = CURRENT_PATH + '/Setup/pinkRandomDots.png'
        PINK_WALLPAPER = my_folders.get_setup_path('pinkRandomDots.png')


"""
