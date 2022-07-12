#!/bin/bash

# Linux/amd64 bash only. Will not work on Arm-based m1 MacBook


set -exu

export ORIG_FOLDER=$(pwd)

# Can be executed from anywhere:
THIS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
#REPO_ROOT=$(git rev-parse --show-toplevel)
#cd $REPO_ROOT
# cd $REPO_ROOT/polysampler/experimentation/sampler1
# realpath ..
#export DOCKER_FOLDER="$REPO_ROOT/polysampler/experimentation"

export DOCKER_FOLDER=$(realpath "$THIS_SCRIPT_DIR/../../../../../polysampler/experimentation")
# todo: DOCKER_FOLDER=$(realpath "$THIS_SCRIPT_DIR/../../..")
# -v "$DOCKER_FOLDER":"$DOCKER_FOLDER"

#DEFAULT_COMMAND="bash /sosi/sampler1/scripts/remote/clang-container/build-20-all.bash"
DEFAULT_COMMAND="bash ./sampler1/scripts/remote/clang-container/build-20-all.bash"
#DEFAULT_COMMAND="pwd"

# -v "$DOCKER_FOLDER":/sosi -w="/sosi"
docker run -it --rm \
    -v "$DOCKER_FOLDER":"$DOCKER_FOLDER" \
    -w="$DOCKER_FOLDER" \
    conanio/clang14-ubuntu16.04:latest  \
    ${@:-$DEFAULT_COMMAND}

# outputs put in CWD can eb found in $DOCKER_FOLDER
