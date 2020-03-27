# Bi-Directional Attention for Joint Instance and Semantic Segmentation in Point Clouds(BAN)
The full paper is available at: [https://arxiv.org/abs/2003.05420](https://arxiv.org/abs/2003.05420)
## Dependencies
The code has been tested with Python 3.7 on Ubuntu 16.04.
* tensorflow 1.14
* h5py
* IPython
* scipy
## Data and Model
Download 3D indoor parsing dataset (S3DIS Dataset). Version 1.2 of the dataset is used in this work.
```
python collect_indoor3d_data.py
python gen_h5.py
cd data && python generate_input_list.py
cd ..
```
## Usage
* Compile TF Operators  
Refer to [PointNet++](https://github.com/charlesq34/pointnet2)
* Training
```
python train.py
```
* Estimate_mean_ins_size
```
python estimate_mean_ins_size.py
```
* Test
```
python test.py
```
* Evaluation
```
python eval_iou_accuracy.py
```
## Acknowledgemets
This code largely benefits from following repositories: [PointNet++](https://github.com/charlesq34/pointnet2), [SGPN](https://github.com/laughtervv/SGPN), [DGCNN](https://github.com/WangYueFt/dgcnn), [DiscLoss-tf](https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow) and [ASIS](https://github.com/WXinlong/ASIS)
