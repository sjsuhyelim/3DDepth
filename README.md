# 3DDepth
## Readthedocs

The document for 3D detection model design and Kitti and Waymo data training: https://mytutorial-lkk.readthedocs.io/en/latest/mydetector3d.html

The document for 3D detection based on V2X cooperative Lidar sensing data: https://mytutorial-lkk.readthedocs.io/en/latest/3DV2X.html

Nuscence dataset and BEV transform based on Lift Splat is available on: https://mytutorial-lkk.readthedocs.io/en/latest/nuscenes.html

## Setup repo
Clone this repository, install this package (need NVIDIA CUDA environment)
```bash
python setup.py develop
```

Install the SparseConv library from [spconv](https://github.com/traveller59/spconv) and [numba](https://numba.pydata.org/numba-doc/latest/user/installing.html):
```bash
pip install spconv-cu117 #pip install spconv-cu118
pip install numba
```

build additional cuda ops libraries via
```bash
(mycondapy39) [010796032@cs001 3DDepth]$ module load cuda-11.8.0-gcc-9.5.0-bnam7p6 #updated by Hyelim Yang (It's for HPC)
pip install nuscenes-devkit #required by nuscenes dataset
pip install efficientnet_pytorch==0.7.0 #required by lss
pip install pynvml
pip install nvidia-ml-py3 #required by import nvidia_smi
pip3 install --upgrade pyside2 pyqt5 #qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
pip install kornia #required by BEVFusion
$ module load gcc-11.2.0-gcc-10.4.0-r4tpxno # updated by Hyelim Yang (It's for HPC)
$ python cudasetup.py build_ext --inplace
```

Install 'mayavi' (ref: https://docs.enthought.com/mayavi/mayavi/installation.html) and open3d (ref: http://www.open3d.org/docs/release/getting_started.html) for 3d point cloud visualization
```bash
conda create --name mycondapy311 python=3.11
conda activate mycondapy311
pip install pyqt6
#test pyqt6: sdrpysim/testpyqt6.py
pip install pyqtgraph
#Successfully installed numpy-1.26.1 pyqtgraph-0.13.3
#import pyqtgraph as pg
#test pyqtgraph: sdrpysim\pyqt6qtgraphtest.py
pip install matplotlib #conda install matplotlib will install pyqt5
#Successfully installed contourpy-1.2.0 cycler-0.12.1 fonttools-4.44.0 kiwisolver-1.4.5 matplotlib-3.8.1 packaging-23.2 pillow-10.1.0 pyparsing-3.1.1 python-dateutil-2.8.2 six-1.16.0
pip install opencv-python-headless
pip install mayavi
#pip3 install PySide6 #will cause pyqt6 not working, but mayavi needs PySide6
pip install pyqt5 #needed by mayavi and matplotlib
conda install -c conda-forge jupyterlab
python VisUtils/testmayavi.py #test mayavi installation
pip install open3d #does not support python3.11, only 3.7-3.10
#install development version of open3d: http://www.open3d.org/docs/latest/getting_started.html
pip install -U --trusted-host www.open3d.org -f http://www.open3d.org/docs/latest/getting_started.html open3d
# Verify installation
python -c "import open3d as o3d; print(o3d.__version__)"
# Open3D CLI
open3d example visualization/draw
python VisUtils/testopen3d.py #test open3d installation
```

## BEV Fusion Training in HPC now
```bash
(mycondapy310) [010796032@cs001 3DDepth]$ python ./mydetector3d/tools/mytrain.py
```

## Kitti Dataset
Check [kittidata](Kitti/kittidata.md) for detailed information of Kitti dataset.

## Waymo Dataset
Check [waymodata](Waymo/waymodata.md) for detailed information of Waymo dataset.

# Fall 2023 CMPE 249 Project by Hyelim Yang
 
### Topic: In the PointPillars model, which structural component improves more effective when modified?
## Data
- Waymo KITTI point cloud data in the lab computer `/DATA5T/Dataset/WaymoKitti/4c_train5678`
- Total number of the training data: 13,632
| Class | The number of classes in the training data |
| ------| -------------------------------------------|
| Car   | 229,664|
| Pedestrian | 87,812 |
| Cyclist | 4,390 |

## Implementation details
### Modify BACKBONE_2D to BaseBEVResBackbone for the PointPillars
#### Reference: https://github.com/open-mmlab/OpenPCDet
1. Added `BasicBlock` and `BaseBEVResBackbone` classes in base_bev_backbone.py
2. Added  `BaseBEVResBackbone: BaseBEVResBackbone` in mydetector3d/models/backbones_2d/__init__.py
3. Created `pointpillar_resnet.yaml` with `BaseBEVResBackbone` for BACKBONE_2D under waymokitti_models

### Modify Anchor-based DENSE_HEAD to Anchor-free DENSE_HEAD, CenterPoint
#### Reference: https://github.com/open-mmlab/OpenPCDet
1. Added `centerpoint.py` under mydetector3d/models/detectors
2. Added `CenterHead: CenterHead` in mydetector3d/models/dense_heads/__init__.py
3. Created `centerpoint_pillar.yaml` with `CenterHead` for DENSE_HEAD and `CenterPoint` for MODEL NAME under waymokitti_models
4. Added `CenterPoint: CenterPoint` in __modelall__ dictionary in mydetector3d/tools/mytrain.py

### Training
- Performed in the lab computer equipped with Tesla P100 GPU.
- batch_size = 2 to avoid CUDA memory error
- All models were trained for epoch 32
| Model | the number of parameters |
|-------|--------------------------|
| PointPillars | 4,838,728 |
| PointPillars with Resnet | 5,348,168 |
| CenterPoint Pillars | 5,224,011 |

### Evaluation
- Performed in HPC
- Created `myevaluatev3.py` to create `.pkl` file saving numpy array instead of torch tensor with CUDA for visualization.
- Used KITTI evaluation metric
- The number of test dataset = 3,409
PointPillars
```
Car AP@0.70, 0.70, 0.70:
bbox AP:0.0000, 0.0000, 0.0000
bev  AP:0.0000, 0.0000, 0.0000
3d   AP:0.0000, 0.0000, 0.0000
aos  AP:0.00, 0.00, 0.00

Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:9.6122, 9.5926, 9.5926
bev  AP:27.7538, 27.7461, 27.7461
3d   AP:14.5113, 14.5087, 14.5087
aos  AP:0.76, 0.75, 0.75

Cyclist AP@0.50, 0.50, 0.50:
bbox AP:0.0297, 0.0296, 0.0296
bev  AP:12.3881, 12.4471, 12.4471
3d   AP:0.1436, 0.1435, 0.1435
aos  AP:0.00, 0.00, 0.00

Inference time:  54.495 sec
```
PointPillars with Resnet
```
Car AP@0.70, 0.70, 0.70:
bbox AP:0.0000, 0.0000, 0.0000
bev  AP:0.0000, 0.0000, 0.0000
3d   AP:0.0000, 0.0000, 0.0000
aos  AP:0.00, 0.00, 0.00

Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:2.7522, 2.7401, 2.7401
bev  AP:24.1469, 24.1210, 24.1210
3d   AP:13.7812, 13.7829, 13.7829
aos  AP:0.60, 0.60, 0.60

Cyclist AP@0.50, 0.50, 0.50:
bbox AP:0.0084, 0.0083, 0.0083
bev  AP:12.5783, 12.4889, 12.4889
3d   AP:0.0602, 0.0602, 0.0602
aos  AP:0.01, 0.01, 0.01

Inference time: 54.435 sec
```
CenterPoint Pillars
```
Car AP@0.70, 0.70, 0.70:
bbox AP:5.2894, 5.2820, 5.2820
bev  AP:32.7146, 32.6993, 32.6993
3d   AP:12.7012, 12.6918, 12.6918
aos  AP:2.03, 2.02, 2.02

Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:1.3936, 1.3884, 1.3884
bev  AP:20.6926, 20.6477, 20.6477
3d   AP:6.0248, 6.0274, 6.0274
aos  AP:1.12, 1.12, 1.12

Cyclist AP@0.50, 0.50, 0.50:
bbox AP:0.0008, 0.0008, 0.0008
bev  AP:14.5301, 14.5288, 14.5288
3d   AP:0.3376, 0.3367, 0.3367
aos  AP:0.00, 0.00, 0.00

Inference time: 123.375 sec
```

### Visualization 
- Performed in the local machine (Mac M1 chip)
- mayavi libray is required
- Modify `visonebatch.py` not to import torch and to be able to plot only ground truth data 

