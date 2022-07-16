# ssh  ssss@206.189.2.78

THIS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

set -eux

# local
export LOCAL_SRC="$HOME/cs/scientific-code/polysampler/experimentation"
export LOCAL_BUILD_COPY="$LOCAL_SRC/sampler1/remote-target"
# remote
export REMOTE_SRC="~/remote-target/polysampler"
export REMOTE_BUILD="$REMOTE_SRC/experimentation/sampler1/target"
# from inside docker in remote
export REMOTE_RUNTIME="$REMOTE_SRC/experimentation"
export LOCAL_RUNTIME="$LOCAL_BUILD_COPY/output"

{
echo "Gogin to clean up remote:"
ssh -t ssss@206.189.2.78 "
 set -eux;
 ls -alt $REMOTE_SRC;
 echo 'Goging to delete: rm -rf :' $REMOTE_SRC;
 rm -rf $REMOTE_SRC;
 echo 'ok'
"
echo 'back to surface'
exit
}

