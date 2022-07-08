
# Linux, bash only. Will not work on Arm-based m1 MacBook

FLAGS="-std=c++20 -stdlib=libc++ -fmodules -fbuiltin-module-map"
docker run -it --rm -v $(realpath ..):/sosi -e FLAGS="$FLAGS"  conanio/clang14-ubuntu16.04:latest  bash -c '
cd /sosi/sampler1;
clang++ $FLAGS -fprebuilt-module-path=. -c   -Xclang -emit-module-interface  point_t.hpp -o point_t.module.precompiled;
clang++ $FLAGS -fprebuilt-module-path=. -fmodule-file=point_t=point_t.module.precompiled pmain.cpp;
date
'
