# run inside linux docker only (linux/amd64). Use `dk.bash` to run this

FLAGS="-std=c++20 -stdlib=libc++ -fmodules -fbuiltin-module-map"

cd /sosi/sampler1
mkdir -p ./target
clang++ $FLAGS -fprebuilt-module-path=./target -c   -Xclang -emit-module-interface  point_t.hpp -o ./target/point_t.module.precompiled
clang++ $FLAGS -fprebuilt-module-path=./target -fmodule-file=point_t=./target/point_t.module.precompiled pmain.cpp -o ./target/a.out
date
