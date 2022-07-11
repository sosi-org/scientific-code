# run inside linux docker only (linux/amd64). Use `dk.bash` to run this

set -exu
PS4='___Line_${LINENO}:______\n'

FLAGS="-std=c++20 -stdlib=libc++ -fmodules -fbuiltin-module-map"

# cd /sosi/sampler1
#cd ./sampler1
THIS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export SOURCE_FILES=$(realpath "$THIS_SCRIPT_DIR/../../..")   # ...../sampler1
# export BTARGET="./target"
export BTARGET="$SOURCE_FILES/target"

cd $SOURCE_FILES
pwd
mkdir -p $BTARGET

find $BTARGET
rm -rf $BTARGET  # clean up the files moved from remote client
mkdir -p $BTARGET
find $BTARGET

#     -O3 \
ls -1 $BTARGET/point_t.module.precompiled || \
clang++ $FLAGS \
    -fprebuilt-module-path=$BTARGET \
    -c  \
    -Xclang -emit-module-interface  \
    point_t.hpp \
    -o $BTARGET/point_t.module.precompiled

echo 'skip2' || \
ls -1 $BTARGET/pmain.out || \
clang++ $FLAGS \
    -fprebuilt-module-path=$BTARGET -fmodule-file=point_t=$BTARGET/point_t.module.precompiled \
    pmain.cpp \
    -o $BTARGET/pmain.out

#     -O3 \
clang++ $FLAGS \
    -fprebuilt-module-path=$BTARGET -fmodule-file=point_t=$BTARGET/point_t.module.precompiled \
    -c -Xclang -emit-module-interface  \
    tesselation_t.hpp \
    -o $BTARGET/tesselation_t.module.o
# .module .mo

clang++ $FLAGS \
    -fprebuilt-module-path=$BTARGET \
    -fmodule-file=point_t=$BTARGET/point_t.module.precompiled \
    -Xclang -emit-module-interface  \
    -c \
    export_svg.hpp \
    -o $BTARGET/export_svg.module.o
# necesary ^ : -Xclang -emit-module-interface  \

echo 'skip' || \
clang++ $FLAGS \
    -v \
    -fprebuilt-module-path=$BTARGET \
        -fmodule-file=export_svg_module=$BTARGET/export_svg.module.o \
    export_svg_demo.cpp \
    -o $BTARGET/export_svg_demo.o
#         -fmodule-file=point_t=$BTARGET/point_t.module.precompiled \

#     -O3 \
clang++ $FLAGS \
    -fprebuilt-module-path=$BTARGET \
    -fmodule-file=point_t=$BTARGET/point_t.module.precompiled \
    -fmodule-file=tesselation_t=$BTARGET/tesselation_t.module.o \
    -v \
    sampler1.cpp \
    -o $BTARGET/sampler1.out

date
