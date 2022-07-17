An implementation of the polygon sampler

A Tessellation (a covering set of convex polygons) is used to sample
another Tessellation (pixels).

Implemented using C++20 (featuring modules).

This is not the fastest implementaion, but one way of implementing it.

### How to
To build and run:
```bash
./scripts/remote-build.bash && ./scripts/remote-run.bash
```

To cleanup the build:
```bash
./scripts/remote-cleanup.bash
```
### Gallery
Intersection of two boxes:
[commit](https://github.com/sosi-org/scientific-code/tree/d304871b762190e121be019de415f0eca1426ba0),
[commit](https://github.com/sosi-org/scientific-code/tree/d93736d3001ce90d4cbf36ec0c97671016f28297).

![x](./gallery/intersection1_17j22-3.svg)
![x](./gallery/intersection1-17-jul-22-4c.svg)

