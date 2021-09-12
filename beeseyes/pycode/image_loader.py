import numpy as np
import scipy.misc
import imageio

# todo: deprecate public use
def load_image(image_path):
   img = imageio.imread(image_path)
   pict_array2d = np.asarray(img)
   #pict_array2d = numpy.array(Image.fromarray(pict_array2d).resize())
   #pict_array2d = scipy.misc.imresize(pict_array2d, FIXED_SIZE[0:2])

   return pict_array2d

UNIT_LEN_CM = 1.0
UNIT_LEN_MM = 0.1

def load_image_withsize(image_path, sample_px=200, sample_cm=10.0, dpi=None):
   '''
      Loads texture with physical size in cm
      Either specify the `dpi` or the `(sample_px,sample_cm)` pair.
   '''

   TRANSPOSE = False

   texture = load_image(image_path)

   #  (192, 256, 3)
   # Ignore image alpha
   if (texture.shape[2] == 4):
       texture = texture[:,:,:3]
   # (701, 1162, 3)
   if (TRANSPOSE):
     texture = np.transpose(texture, axes=(1,0,2))

   # dpi:  in fact, "dots per centimeters", D.P.C.
   # Each circle is 200x200 px^2, 6.0 cm <---> 200 px
   # sample_cm = 6.0
   #sample_cm = 10.0
   #sample_px = 200

   if dpi is None:
       #dpi = texture.shape * sample_cm / sample_px
       dpi = 1.0 * sample_cm / float(sample_px)
   else:
       pass

   print('dpi', dpi, '(dots per cm)')
   physical_size = (texture.shape[0]* dpi, texture.shape[1]* dpi)
   print(physical_size, '(cm x cm)')
   return texture, physical_size, dpi

'''
Samples the area in the image (pixels)
inside the given 6-points (hexagon)
that is casted on the image.
The image coords (and u,v) are within [0,1]x[0,1]
6-point area/pixel sampling
'''
def sample_hex(u6,v6, texture):
    (w,h,rgb3) = texture.shape # (192, 256, 3)
    print(u6*w)
    print(v6*h)
    pass

