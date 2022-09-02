#!/usr/bin/env bash

set -xu

#!/usr/bin/env bash

export ORIG_FOLDER=$(pwd)

# Can be executed from anywhere:
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

export VENV_NAME="p3-for-me"

mkdir -p temp
source $REPO_ROOT/temp/my-bash-utils.sh || curl -k \
  https://raw.githubusercontent.com/sohale/implisolid/revival-sohale/scripts/bash-utils.sh \
  >$REPO_ROOT/temp/my-bash-utils.sh

source $REPO_ROOT/temp/my-bash-utils.sh

# todo: flags, states
# define flags for states:
# some tagets, flags for conditionals
# a reset (refresh) option (not brew)
# conditions for too-slow installations (brew or pyqt): $IF_QT, $IF_BREW
# separate the global ones (brew)
# "but only if needed"
# idea: to use a local folder `.state-system/`

# nobrew: Ignore all brew commands
export IF_BREW=
export IF_QT=

set -e

# if behind a firewall
export PIPFLAGS="\
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org"

# not behind a firewall
export PIPFLAGS=""

echo "PIPFLAGS>>>> $PIPFLAGS"

# does a `cd`
function chk_venv() {
  # a solution based on `venv` as opposed to `virutalenv`

  #set -ex
  if [[ -d $REPO_ROOT/temp/$VENV_NAME ]]; then
    echo "venv exists"
    return 0
  fi

  echo "INSTALLING THEM: cleanup"
  # never parametrize/tokenie/env-ize an `rm -rf`` command
  rm -rf $REPO_ROOT/temp/p3-for-me || :

  cd $REPO_ROOT/temp
  # venv is shipped with python3
  #python3 -m venv -v --python=python3 $VENV_NAME
  python3 -m venv $VENV_NAME
  source "$REPO_ROOT/temp/$VENV_NAME/bin/activate"

  python --version
  # Python 3.9.12

  # For trusted sources: see  https://stackoverflow.com/questions/49324802/pip-always-fails-ssl-verification

  python -m \
    pip install \
    $PIPFLAGS \
    --upgrade pip

  cd $REPO_ROOT
}

# to refresh: `rm -rf $REPO_ROOT/temp/p3-for-me`
MAKE_HAPPEN "$REPO_ROOT/temp/$VENV_NAME/bin/activate" || {

  chk_venv
}

# cd $REPO_ROOT/temp

source $REPO_ROOT/temp/$VENV_NAME/bin/activate
# export PYTHON_VER="python3.9"
export PYTHON_VER="$(ls -1t $REPO_ROOT/temp/$VENV_NAME/lib/ | grep -i "python" | head -n 1)"

export VENV_PACKAGES="$REPO_ROOT/temp/$VENV_NAME/lib/$PYTHON_VER/site-packages"
echo VENV_PACKAGES=$VENV_PACKAGES

MAKE_HAPPEN "$VENV_PACKAGES/numpy/LICENSE.txt" || {
  pip install \
    $PIPFLAGS \
    numpy

  pip install \
    $PIPFLAGS \
    matplotlib
}

MAKE_HAPPEN "$VENV_PACKAGES/scipy/LICENSE.txt" || {
  pip install $PIPFLAGS scipy
}

# echo "Warning: MAKE_HAPPEN not tested with *" # ; exit
# MAKE_HAPPEN "$VENV_PACKAGES/scikit_image-*" || { ... }
#MAKE_HAPPEN "$VENV_PACKAGES/scikit_image-0.19.3.dist-info/LICENSE.txt" || {
MAKE_HAPPEN "$VENV_PACKAGES/scikit_image-*/LICENSE.txt" || {
  pip install scikit-image
}

MAKE_HAPPEN "$VENV_PACKAGES/sympy/__init__.py" || {
  pip install $PIPFLAGS sympy
}

if false; then
  MAKE_HAPPEN "$VENV_PACKAGES/yaml/__init__.py" || {
    pip install $PIPFLAGS PyYAML
  }
  #MAKE_HAPPEN "$VENV_PACKAGES/pdb/__init__.py" || {
  #  pip install $PIPFLAGS pdb
  #}

  # python -m pip install -U autopep8

  MAKE_HAPPEN "$VENV_PACKAGES/graphviz/__init__.py" || {
    pip install $PIPFLAGS graphviz
  }

  # but only if needed
  MAKE_HAPPEN "$VENV_PACKAGES/numexpr/__init__.py" || {
    pip install numexpr
  }
fi

# qt-vtk-mayavi for 3d
if false; then
  #[[ -n $IF_QT ]] && {

  exit # hard stop

  echo "Warning: MAKE_HAPPEN not tested with *"; exit

  MAKE_HAPPEN "$VENV_PACKAGES/numpy_stl*" || {
    pip install numpy-stl
  }

  ######################################################
  # Attempts to install mayavi-related packages
  ######################################################

  # I documented the steps as an answer here:
  #   https://stackoverflow.com/questions/71695844/how-to-install-mayavi-on-macos/72487758#72487758

  # Run manually only: ($IF_BREW)
  #     brew install vtk  # installs vtk@9.1

  MAKE_HAPPEN "$VENV_PACKAGES/vtk.py" || {
    pip install vtk
  }

  if false; then
    pip install wheel
    pip install ipdb
    pip install numexpr
  fi

  if false; then
    [[ -n $IF_BREW ]] && brew install vtk # installs vtk@9.1
    pip install vtk
    pip install mayavi
  fi

  if false; then
    [[ -n $IF_BREW ]] && brew install vtk
  fi

  ###########
  # QT
  ###########

  echo >/dev/null '
    brew install qt
    brew install pyqt5 ?
    brew install pyside #?
  '
  #MAKE_HAPPEN "$VENV_PACKAGES/???" || {
  #  very slow
  #  pip install pyqt5
  #}
  #qmake

  if false; then
    #brew install pyqt5 ?
    #pip install pyqt5 ?
    [[ -n $IF_BREW ]] && brew install qt5
    # no such thing? : brew install pyqt5

    # important:
    brew info qt5
    # then run those commands
    #  echo 'export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"' >> ~/.zshrc
    echo $PATH | grep qt@5 # make sure PATH is set for qt5
    export LDFLAGS="-L/opt/homebrew/opt/qt@5/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/qt@5/include"
    # try qmake

    #export QT_API=pyqt5
    #export ETS_TOOLKIT=pyqt5
    # No pyface.toolkits plugin found for toolkit pyqt5
    # ?? https://pypi.org/project/pyface/

    # test if qt5 command `qmake`` works
    qmake             # test brew qt5
    pip install pyqt5 # slow: builds using clang

    export ETS_TOOLKIT=
    export QT_API=pyqt5
  fi

  # end of QT

  MAKE_HAPPEN "$VENV_PACKAGES/mayavi/__init__.py" || {
    pip install mayavi
  }

  ######################################################
  # End of attempts to run mayavi-based script
  ######################################################

fi # qt-vtk-mayavi for 3d

###############################################
# Attempts to install a different backend for matplotlib (not for implisolid)
###############################################
if false; then
  #brew install pkg-config
  #brew link pkg-config
  #brew install pygtk
  #brew install freetype
  #brew install libpng

  true || MAKE_HAPPEN "$VENV_PACKAGES/mplcairo/__init__.py" || {
    # for matplotlib only:

    # mplcairo: for attempt notes about cairo (mplcairo), see https://github.com/sohale/point-process-simple-example/blob/82a62d013d909f365a391aa254dc598d62a0c2d4/run_script.bash

    # brew install llvm
    brew info llvm # keg-only
    # echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc

    # bundled libc++:
    #   LDFLAGS="-L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib"
    export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
    # export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
    echo $PATH | grep llvm # make sure PATH is includes llvm

    # for llvm (failed attempt)
    export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
    # export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
    echo $PATH | grep llvm

    # end of mplcairo

    ######## another solution for another matplotlib backend

    #MAKE_HAPPEN "$VENV_PACKAGES/mpl_interactions/__init__.py" || {
    #  pip install mpl_interactions
    #}

  }

######################################################
# End of attempts for new backends for matplotlib
######################################################
fi

# ../p3-for-me/lib/python3.9/site-packages/simplejson/__init__.py
MAKE_HAPPEN "$VENV_PACKAGES/simplejson/__init__.py" || {
  pip install simplejson
}

MAKE_HAPPEN "$VENV_PACKAGES/typeguard/__init__.py" || {
  pip install typeguard
}

MAKE_HAPPEN "$VENV_PACKAGES/openpyxl/__init__.py" || {
  pip install openpyxl
}


MAKE_HAPPEN "$VENV_PACKAGES/shapely/__init__.py" || {
  pip install shapely
}

echo "venv is at $REPO_ROOT/temp/$VENV_NAME/bin/activate"

######################################################
# End of instllations
######################################################

echo "Main script"


source $REPO_ROOT/temp/$VENV_NAME/bin/activate
python --version

echo "
    source $REPO_ROOT/temp/$VENV_NAME/bin/activate
    # export ETS_TOOLKIT=
    # export QT_API=pyqt5

    cd $ORIG_FOLDER
    python xyz.py
"

# todo: fetch this script from my repo, make it public, and push via a separate repo
cd $ORIG_FOLDER
python pydemo.py
# python old_demo.py


echo 'fine'
exit
"""
Forked from: https://github.com/sohale/scientific-code-private/blob/main/dependence-paritial-order/run_script.bash

Forked from ~/cs/implisolid/sandbox/sympy-experiment/run_script.bash
Forked from: point-process-simple-example/run_script.bash: https://github.com/sohale/point-process-simple-example/blob/82a62d013d909f365a391aa254dc598d62a0c2d4/run_script.bash
Forked from https://github.com/sosi-org/scientific-code/blob/main/timescales-state/run-script.bash
Forked from https://github.com/sosi-org/primsoup/blob/master/actn/run-actn.bash

https://github.com/sosi-org/scientific-code/blob/main/beeseyes/install-macos.sh
https://github.com/sosi-org/scientific-code/blob/main/beeseyes/pycode/how.sh


https://github.com/sosi-org/scientific-code/blob/main/timescales-state/run-script.bash
4 june -  13 days ago

https://github.com/sosi-org/scientific-code/blob/main/beeseyes/pycode/how.sh
19 May 2021


https://github.com/sohale/bitsurf/blob/master/deploy-run-levels.sh
15 Jan 2021
?

https://github.com/sosi-org/grpc-arch-practice/blob/master/nov-2020/init.sh
19 Nov 2020

4 October 2020
?


https://github.com/sosi-org/grpc-arch-practice/blob/master/tf-serving-18-sept-2020/prepare-env.sh
22 Sep 2020


*
https://github.com/sosi-org/grpc-arch-practice/blob/master/tfserving-example/source-tf1.sh
11 Jul 2020



https://github.com/sohale/nedanepy/blob/master/vend-instructions.md
3 Apr 2020


https://github.com/sosi-org/neural-networks-sandbox/blob/master/glyphnet/tests/run_all_tests.sh
30 Dec 2019

https://github.com/sosi-org/neural-networks-sandbox/blob/master/glyphnet/used_refs.md
22 Dec 2019
(as note)

notes from:
https://github.com/sosi-org/neural-networks-sandbox/blob/master/dataset-from-stan/extract_plans_from_grid.py
11 Nov 2019

https://github.com/ladanalavi/phd-thesis/blob/master/gan/linux%20tesla%20kent.txt
11 Nov 2019
16 Feb 2019
https://github.com/ladanalavi/phd-thesis/blob/master/gan/notes.txt
16 Feb 2019

https://github.com/sosi-org/REST-practice/blob/master/readme.md
19 Sep 2018

"""
