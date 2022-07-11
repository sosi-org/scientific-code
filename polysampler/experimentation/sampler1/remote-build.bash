# ssh  ssss@206.189.2.78

#  bash #./dk.bash
echo '' || \
ssh -t ssss@206.189.2.78 '
 cd ~/cs/scientific-code/polysampler/experimentation/sampler1;
 bash ./dk.bash;
 echo;
 echo "******\n*\n* ssh:bye-remote"
'
# ssh the input device is not a TTY docker
# this works:
#ssh  ssss@206.189.2.78 '
#docker run --rm -ti alpine cat
#'
# solution: ssh -t


#WD="."
#mkdir  -p $WD/remote-target
#rm -rf $WD/remote-target
#mkdir  -p $WD/remote-target


# REMOTE_LOCATION_OF_INTEREST
#export LOCAL_BUILD="$WD/remote-build"
#export REMOTE_LOI="~/remote-build/polysampler"
#export LOCAL_LOI="~/cs/scientific-code/polysampler/experimentation"
export LOCAL_SRC="$HOME/cs/scientific-code/polysampler/experimentation"
export REMOTE_SRC="~/remote-target/polysampler"
export REMOTE_BUILD="$REMOTE_SRC/experimentation/sampler1/target"
export LOCAL_BUILD_COPY="$LOCAL_SRC/sampler1/remote-target"

mkdir  -p $LOCAL_BUILD_COPY

set -ex

ssh -t ssss@206.189.2.78 "
mkdir -p $REMOTE_SRC
"

#echo 'skip pre-sync' || \
rsync \
   -rv \
   "$LOCAL_SRC" \
   ssss@206.189.2.78:$REMOTE_SRC

#echo 'skip build' || \
ssh -t ssss@206.189.2.78 "
 set -eux;
 cd $REMOTE_SRC;
 cd experimentation/sampler1;
 bash ./dk.bash;
 echo;
 echo '******\n*\n* ssh:bye-remote'
"

mkdir -p $LOCAL_BUILD_COPY

rsync \
   -rv \
   ssss@206.189.2.78:$REMOTE_BUILD \
   "$LOCAL_BUILD_COPY"

#   ~/cs/scientific-code/polysampler/experimentation/sampler1/target
#   -p "$RSYNC_PASSWORD" \

ls -alt $LOCAL_BUILD_COPY

#find $LOCAL_BUILD_COPY
export RELATIVE=$(realpath --relative-to="$(pwd)" "$LOCAL_BUILD_COPY")
find $RELATIVE
