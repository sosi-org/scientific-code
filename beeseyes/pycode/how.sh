# brew install tcl-tk

# level 1
virtualenv -v --python=python3 p3
# ^ Intalled 3.9.4
source ./p3/bin/activate
pip install scipy
pip install matplotlib

# level 2
source ./p3/bin/activate
python pydemo.py
