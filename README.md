# TX-cloud-models

## FlashOCC
1. Prepare nuScenes dataset as introduced in nuscenes_det.md and create the pkl for FlashOCC by running:
````
python tools/create_data_bevdet.py
````
2. For Occupancy Prediction task, download (only) the 'gts' from CVPR2023-3D-Occupancy-Prediction
3. rename customer ops in projects/mmdet3d_plugin/ops to solve "ninja: error: build.ninja:26: multiple rules generate" issue:
````
cd projects/mmdet3d_plugin/ops
mv bev_pool/src/bev_sum_pool_cuda.cu bev_pool/src/bev_sum_pool_hip_cuda.cu
mv bev_pool/src/bev_max_pool_cuda.cu bev_pool/src/bev_max_pool_hip_cuda.cu
mv bev_pool_v2/src/bev_pool_cuda.cu bev_pool_v2/src/bev_pool_hip_cuda.cu
mv nearest_assign/src/nearest_assign_cuda.cu nearest_assign/src/nearest_assign_hip_cuda.cu
````
4. Modify setup.py the cuda file name.
5. Modify projects/mmdet3d_plugin/core/evaluation/ray_metrics.py L13
````
#dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=[])
````
6. Modify the cuda interface specially for torch2.7
````
lib/dvr/dvr.cu L371: sigma.type() -> sigma.scalar_type()
lib/dvr/dvr.cu L683: sigma.type() -> sigma.scalar_type()
lib/dvr/dvr.cu L736: points.type() -> points.scalar_type()
````
7. Modify get_ego_coor in projects/mmdet3d_plugin/models/necks/view_transformer.py L153 to speedup torch.bmm.
8. Training
````
python tools/train.py projects/configs/flashocc/flashocc-r50.py
````
## Sparse4D
1. Prepare data
2. rename customer ops in projects/mmdet3d_plugin/ops to solve "ninja: error: build.ninja:26: multiple rules generate" issue:
````
cd projects/mmdet3d_plugin/ops
mv src/deformable_aggregation_cuda.cu src/deformable_aggregation_hip_cuda.cu
````
3. modify setup.py the cuda file name
4. Download pre-trained weights and Training
````
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704
````
## Deformable-DETR

1. Dataset preparation
Please download COCO 2017 dataset and organize them as following:
````
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
````
2. Compiling CUDA operators
````
cd ./models/ops
sh ./make.sh
python test.py
````
3. Modified:   util/misc.py
comment L30~L59
4. Training 
````
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
````

## CenterPoint

1. Dataset preparation
- Check and follow the instruction in `prepare_nuscenes.sh` to set related variables properly and then run the sctipt to prepare the nuScenes dataset.
- It will take a long time (a few hours) to download and process the dataset.
```bash
bash prepare_nuscenes.sh
```
2. Model training
- Set config file, working directory properly, GPU numbers in `run_centerpoint.sh` then execute it for training.
```bash
bash run_centerpoint.sh
```

## Mask2Former

1. Dataset preparation
Please download COCO 2017 dataset and organize them as following:
````
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
````

2. Modified:
change the value of data_root to the value of code_root in ./configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py
3. Training 
````
bash ./tools/dist_train.sh configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py 8
````

## FCOS_EfficientNetB0

1. Modified:
use COCO 2017 dataset, change the value of data_root to the value of code_root in ./configs/fcos/fcos_efficientnet_caffe_fpn_gn-head_1x_coco.py
2. Training 
````
python ./tools/train.py configs/fcos/fcos_efficientnet_caffe_fpn_gn-head_1x_coco.py
````
## MapTRv2

Please refer to the guide file in MapTRv2-rocm/READ_rocm_guide.md

## BEVFormer

1. Install dependency
````
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0
````
2. Prepare data 
````
# prepare data before install detectron2, otherwise cause path error.
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
````
3. Install Detectron2
````
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
````
4. Insert `torch.multiprocessing.set_start_method('fork')` before the `main()` call at **line 271** of `tools/train.py` to fix **TypeError: cannot pickle 'dict_keys' object**. 
5. Run the scripts:
````
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
````

## ResNet50
1. prepare code
````
cd resnet50
git clone https://github.com/pytorch/vision.git
````
2. MIOpen tuning (optional)
````
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=4
bash run_train.sh
````
3. Train
````
unset MIOPEN_FIND_MODE=1
unset MIOPEN_FIND_ENFORCE=4
bash run_train.sh
````
4. Train with NHWC layout (optional)
````
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=1
````
also need to change layout of input and model:
````
data = data.to(memory_format=torch.channels_last) # in train.py line 28
model = model.to(memory_format=torch.channels_last) # in train.py line 248
model = torch.compile(model) # to get better performance
````

