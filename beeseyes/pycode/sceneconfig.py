from cte import _PIXELS, _CM
from cte import DIM3, HEX6
import my_folders

PRODUCTION_FIGS = True


#0.00125 , 0.01/2/4
AREA_THRESHOLD = 0.02*10

#0.2
SD_THRESHOLD=0.03

AREA_THRESHOLD = 0.02*10 *100000+1000
SD_THRESHOLD=0.03 * 10000+1000




BLUE_FLOWER = my_folders.get_art_path("256px-Bachelor's_button,_Basket_flower,_Boutonniere_flower,_Cornflower_-_3.jpeg")
BLUE_FLOWER_DPI_INFO = {_PIXELS: 200, _CM: 10.0}


NEW_FLOWER = my_folders.get_setup_path('flower-sept.png')


#NEW_FLOWER_DPI_INFO = {_PIXELS: 1268, _CM: 5.0}
NEW_FLOWER_DPI_INFO = {_PIXELS: 1268, _CM: 3.5}


# 4 x 2 stimuli on a pink background
EIGHT_PANEL = my_folders.get_setup_path('IMG_2872.MOV.BkCorrectedPerspCroppedColourContrast.png')

# pink texture
PINK_WALLPAPER = my_folders.get_setup_path('pinkRandomDots.png')
# choose:
TEXTURES_FILES = [EIGHT_PANEL, PINK_WALLPAPER]
TEXTURES_FILES = [NEW_FLOWER, NEW_FLOWER]
TEXTURE_DPI_INFO = NEW_FLOWER_DPI_INFO


POSITIONS_XLS = my_folders.get_setup_path('beepath.xlsx')
