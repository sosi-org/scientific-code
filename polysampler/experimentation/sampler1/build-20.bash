
# Linux, bash only. Will not work on Arm-based m1 MacBook

FLAGS="-std=c++20 -stdlib=libc++ -fmodules -fbuiltin-module-map"
docker run -it --rm -v $(realpath ..):/sosi -e FLAGS="$FLAGS"  conanio/clang14-ubuntu16.04:latest  bash -c '
cd /sosi/sampler1;
mkdir -p ./target ;
clang++ $FLAGS -fprebuilt-module-path=./target -c   -Xclang -emit-module-interface  point_t.hpp -o ./target/point_t.module.precompiled;
clang++ $FLAGS -fprebuilt-module-path=./target -fmodule-file=point_t=./target/point_t.module.precompiled pmain.cpp -o ./target/a.out;
date
'
