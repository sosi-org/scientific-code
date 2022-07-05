```bash
cd experimentation/sampler1

docker run -it --rm -v $(realpath ..):/sosi -e FLAGS="$FLAGS"  conanio/clang14-ubuntu16.04:latest  bash -c 'cd /sosi/sampler1; exec $SHELL'

# inside the docker shell:

clang++ $FLAGS -fprebuilt-module-path=. -c   -Xclang -emit-module-interface  point_t.hpp -o point_t.module.precompiled

clang++ $FLAGS -fprebuilt-module-path=. -fmodule-file=point_t=point_t.module.precompiled pmain.cpp 

# to run:
./a.out

```

Based on:
* https://blog.ecosta.dev/en/tech/cpp-modules-with-clang
* https://itnext.io/c-20-modules-complete-guide-ae741ddbae3d

