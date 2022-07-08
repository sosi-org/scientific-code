
# Linux/amd64 bash only. Will not work on Arm-based m1 MacBook


docker run -it --rm -v $(realpath ..):/sosi conanio/clang14-ubuntu16.04:latest  \
    bash /sosi/sampler1/build-20-all.bash

