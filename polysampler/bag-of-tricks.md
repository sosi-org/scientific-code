
1. Scan the pixels. Find for each one the tessel-index that falls into. (downsamping)

Use simple addition.

Not good for split pixels. Not good for subpixels (upsampling)

Good for wide tiles that cover many pixels.

Good for vectorized code. (local access)

2. Tiangulate the hexagonal. Sample triangle. Add each one.

3. Various interpolations for upsampling.

4. Are there standard (textbook or published) algorithm in Computational Geometry for sampling of tessellation ? What is the problem name?

5. Need an algorithm to calculate the area shared between a polygon (or triangle) and a square.
(We need two loops: It's a kind of convolution.)
Solution: bounding box is only useful if it is done in the pixels (coord) space.

Triangle or (convex) polygon?

6. (Pixel-cover)-set (of a trialngle or convex-polygon): To go through (these pixels; enmerate them), to calculate only for them.

7. Pixel-(complement-cover)-set. Together with pixel-cover set: we can switch between modes.

8. (Useful to have) algorithm to intersection of two polygons. Or their area only. Both are convex.
polyclipping?

Weiler-Atherton algorithm
Boost.Geometry

9. Interseciton of two convex is convex. Is theer a way to just compute their area?
   * The result Can be a polygon of many sides (18).

10. Can we re-use lines in a clipping? (kind of amortised)

11. It is a rasterization algorithm: In the space of pixels (image).
