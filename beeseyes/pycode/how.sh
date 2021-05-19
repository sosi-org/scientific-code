# brew install tcl-tk

# level 1
virtualenv -v --python=python3 p3
# ^ Intalled 3.9.4
source ./p3/bin/activate
pip install scipy
pip install matplotlib
pip install imageio

pip install Shapely
#pip install pandas
#pip install xlrd
# xlrd does not work:
#            our version of xlrd is 2.0.1. In xlrd >= 2.0, only the xls format is supported. Install openpyxl instead.
# pip install openpyxl

# level 2
source ./p3/bin/activate
python pydemo.py
