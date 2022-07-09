#!/usr/bin/env bash

# Linux/amd64 bash only. Will not work on Arm-based m1 MacBook


set -xu

export ORIG_FOLDER=$(pwd)

# Can be executed from anywhere:
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT
# cd $REPO_ROOT/polysampler/experimentation/sampler1
# realpath ..
export DOCKER_FOLDER="$REPO_ROOT/polysampler/experimentation"

docker run -it --rm -v "$DOCKER_FOLDER":/sosi conanio/clang14-ubuntu16.04:latest  \
    bash /sosi/sampler1/build-20-all.bash

