Poly-Sampler
or
Tesselation-Sampler

Needs to solve three problems:
* Wide areas of image can fall into a single polygon (Many pixels: downsampling)
* Sometimes points (polygons) are subpixel (upsampling)
    * Can have thin and long tiles/triangles
* Sometimes a single pixel falls in multiple tessellation blocks
* It's a mix of above
* May be hexagonal but the numbr of sides N=6 is not fixed.
* Has to be relatively fast (but not real-time)
* So called "Sub-pixel precision" is needed


* Extreme:
   * Sides may be slightly bent
   * Should be amortised

* Compromises:
   * Maybe we don't need upsampling: Nearby => The insect (ommatidia) sees square pixels when too close.

However, there are some good news:
* Polygons are convex
* Tessellation: no overlaps, no gaps
   * The polygons cover the plane
   * and have no overlap
* It is not real-time. Can be slow.
* (May be) limitd to voronoid tesselation (but may be on sphere)
