#!/bin//bash
set -exu

# re-copied from a later .../dynamic_optic_flow/install-venv.bash
# ... (scintific-private-trekk?)
# ...
# from .https://github.com/Opteran/sohail-dev-notes/blob/af519bdff4414ae47ba7c19c6f103818229c389f/ce_visualise_csv/pyinstall.bash#L5
# from: /home/sohail/opteran/sohail-dev-notes/localisation_heuristic/pyinstall.bash

# from: /home/sohail/opteran/sohail-dev-notes/gold_digger/pyinstall.bash

#  See https://github.com/sosi-org/scientific-code/blob/main/timescales-state/run-script.bash
#  https://github.com/sosi-org/scientific-code/blob/256365e82b97fc529fc3626f312848e55eacc3c0/timescales-state/run-script.bash

# On Debian/Ubuntu systems, you need to install the python3-venv package using the following command: (You may need to use `sudo`)

# apt install python3.10-venv

VNAME="p3-for-me"

# rm -rf "$VNAME"

ls "$VNAME" || \
    python3 -m venv "$VNAME"
#--python=python3.5

source "./$VNAME/bin/activate"

# linter
pip install autopep8

pip install numpy
pip install matplotlib
# pip install cppyy
# pip install pygame
# pip install sympy
pip install scipy
pip install scikit-image
pip install opencv-python
# Not: ffmpeg-python, ffmpeg

# pip install --upgrade pip

# On MacOS:
# brew install opencv
# brew install ffmpeg

ffmpeg  -version  # This is an assertion. If error, you need to install ffmpeg

python --version

# Python 3.10.6

echo "source \"./$VNAME/bin/activate\""
# source "./$VNAME/bin/activate"
# python approach_1_demo.py
