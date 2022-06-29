Poly-Sampler
or
Tesselation-Sampler

Needs to solve three problems:
* Wide areas of image can fall into a single polygon (Many pixels: downsampling)
* Sometimes points (polygons) are subpixel (upsampling)
* Sometimes a single pixel falls in multiple tessellation blocks
* May be hexagonal but the numbr of sides N=6 is not fixed.
* Has to be relatively fast (but not real-time)

* Extreme:
   * Sides may be slightly bent

However, there are some good news:
* Polygons are convex
* Tessellation: no overlaps, no gaps
   * The polygons cover the plane
   * and have no overlap
* It is not real-time
* (May be) limitd to voronoid tesselation (but may be on sphere)

