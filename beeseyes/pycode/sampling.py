import numpy as np
import math
import polygon_sampler

nan_rgb = np.zeros((3,)) + np.NaN

# sampler session: texture, W_,H_,W,H
'''
Samples a single pixel.
Using square pixels.

[0, ... ,W-1] (incl.)
By mapping [0,1) -> [0,W) (int)
(mapping u,v)
'''
def sample1(um,vm, texture, W_,H_,W,H):
   if np.isnan(um) or np.isnan(vm):
       rgb = nan_rgb
   else:
       # sample
       py = math.floor(um * H_)
       px = math.floor(vm * W_)
       if px < 0 or py < 0 or px >= W or py >= H:
           rgb = nan_rgb
       else:
          rgb = texture[py,px]
   return rgb

'''
slow.
"Pixel at Centroid" sampler
One pixel is taken for each region

'''
def sample_colors_squarepixels(uv, regions, texture):
    # print('uv.shape', uv.shape)
    if texture.shape[2] == 4:
        texture = texture[:,:, 0:3]
    #print('uv', uv)
    #print('regions', regions)
    #exit()

    EPS = 0.00000001
    # (H,W) mmove to slow part.
    (H,W) = texture.shape[0:2]
    # print('W,H', W,H)
    W_ = (W - EPS)
    H_ = (H - EPS)
    nf = len(regions)
    uvm_for_debug = np.zeros((nf,2),dtype=float)

    regions_rgb = np.zeros((nf,3),dtype=float)
    for i in range(nf):
        # temporary solution: sample at center only
        #if np.isnan(uv[regions[i], 0]):
        um = np.mean(uv[regions[i], 0])
        vm = np.mean(uv[regions[i], 1])
        uvm_for_debug[i, :] = [um, vm]
        rgb = sample1(um,vm, texture, W_,H_,W,H)
        regions_rgb[i] = rgb

    return regions_rgb, uvm_for_debug



def sample_colors(uv, regions, texture):
  # return sample_colors_squarepixels (uv, regions, texture)
  return polygon_sampler.sample_colors_polygons (uv, regions, texture)
