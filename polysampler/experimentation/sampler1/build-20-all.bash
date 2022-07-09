# run inside linux docker only (linux/amd64). Use `dk.bash` to run this

set -exu
PS4='___Line_${LINENO}:______\n'

FLAGS="-std=c++20 -stdlib=libc++ -fmodules -fbuiltin-module-map"

cd /sosi/sampler1
mkdir -p ./target

ls -1 ./target/point_t.module.precompiled || \
clang++ $FLAGS \
    -fprebuilt-module-path=./target \
    -c  \
    -Xclang -emit-module-interface  \
    point_t.hpp \
    -o ./target/point_t.module.precompiled

ls -1 ./target/pmain.out || \
clang++ $FLAGS \
    -fprebuilt-module-path=./target -fmodule-file=point_t=./target/point_t.module.precompiled \
    pmain.cpp \
    -o ./target/pmain.out


clang++ $FLAGS \
    -fprebuilt-module-path=./target -fmodule-file=point_t=./target/point_t.module.precompiled \
    -c -Xclang -emit-module-interface  \
    tesselation_t.hpp \
    -o ./target/tesselation_t.module.o
# .module .mo

clang++ $FLAGS \
    -fprebuilt-module-path=./target \
    -fmodule-file=point_t=./target/point_t.module.precompiled \
    -Xclang -emit-module-interface  \
    -c \
    export_svg.hpp \
    -o ./target/export_svg.module.o
# necesary ^ : -Xclang -emit-module-interface  \

echo 'skip' || \
clang++ $FLAGS \
    -v \
    -fprebuilt-module-path=./target \
        -fmodule-file=export_svg_module=./target/export_svg.module.o \
    export_svg_demo.cpp \
    -o ./target/export_svg_demo.o
#         -fmodule-file=point_t=./target/point_t.module.precompiled \

clang++ $FLAGS \
    -fprebuilt-module-path=./target \
    -fmodule-file=point_t=./target/point_t.module.precompiled \
    -fmodule-file=tesselation_t=./target/tesselation_t.module.o \
    -v \
    sampler1.cpp \
    -o ./target/sampler1.out

date
