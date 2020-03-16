# Bi-Directional Attention for Joint Instance and Semantic Segmentation in Point Clouds(BAN)
## Dependencies
The code has been tested with Python 3.7 on Ubuntu 16.04.
## Data and Model
Download 3D indoor parsing dataset (S3DIS Dataset). Version 1.2 of the dataset is used in this work.
'''
python collect_indoor3d_data.py
python gen_h5.py
cd data && python generate_input_list.py
cd ..
'''
