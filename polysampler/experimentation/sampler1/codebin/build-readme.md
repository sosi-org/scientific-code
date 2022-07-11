```bash
docker run -it --rm -v $(pwd):/sosi  conanio/clang14-ubuntu16.04:latest  bash -c    'cd /sosi; clang++  -std=c++20 -S  point_t.hpp sampler1.cpp  '
```
