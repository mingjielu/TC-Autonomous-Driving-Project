# To download the nuScenes dataset, you should 
# 1. Register and log in at https://www.nuscenes.org/nuscenes to obtain access permissions.
# 2. Set the useremail, password, output_dir and region variables in the script `download_nuscenes.py` before executing it.
python download_nuscenes.py

# Use script from mmdet3d to preprocess the dataset for training and inference
cd mmdetection3d
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
cd ..
