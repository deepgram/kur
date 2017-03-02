# optionally, can set dev=False in make_data.py to train on all the text
python make_data.py
python view_data.py
cd model
kur -v train kurfile.yaml
kur -v evaluate kurfile.yaml
cd ..
python view_outputs.py model/output.pkl
