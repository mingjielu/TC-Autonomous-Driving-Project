# Project Setup and Execution Guide

## 1. Install Environment

### 1.1 Use Docker Provided by AMD.
- Download and build the Docker image provided by AMD.
- Run the Docker container and enter the environment and it includes required mmcv, mmdet, mmseg, mmdet3d libs.

### 1.2 Modify Cuda Ops and install BEV_Pool, BEV_Pool_v2 and GKT
- copy the sub-directories of bev_pool and bev_pool_v2 from orignial MapTR/mmdetection3d/mmdet3d/ops/ to MapTR/projects/mmdet3d_plugin/maptr/modules/ops/ . 
- Minor modifications are required for successful compilation, including creating a setup.py file and adjusting the module import interfaces. Please refer to our code for specific changes.
- Remove the original mmdet3d in maptrv2.

- Install BEV_Pool:
```bash
cd /path/to/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/bev_pool

rename bev_pool_cuda.cu to bev_pool_hip_cuda.cu in src.

python setup.py install
or 
python setup.py build install

```

- Install BEV_Pool_v2:
```bash
cd /path/to/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/bev_pool_v2


rename bev_pool_cuda.cu to bev_pool_hip_cuda.cu in src.

python setup.py install
or 
python setup.py build install
```
- Install GKT:
Modify the cuda interface specially for torch2.7
````
projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/src/geometric_kernel_attn_cuda.cu L56: value.type() -> value.scalar_type()
projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/src/geometric_kernel_attn_cuda.cu L125: value.type() -> value.scalar_type()
````
```bash
cd /path/to/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```
- Install other libs:
```bash
pip install timm
pip install -r requirement.txt
```
### 1.3 Prepare pretrained models.
```bash
cd /path/to/MapTR
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```
## 2. Modify Code of MapTRv2
- Insert `torch.multiprocessing.set_start_method('fork')` before the `main()` call at **line 271** of `MapTR/tools/fp16/train.py` to fix **TypeError: cannot pickle 'dict_keys' object**. 
- Comment out **line 156** of `MapTR/projects/mmdet3d_plugin/models/backbones/efficientnet.py` to fix the registration bug of `EfficientNet`.
- Comment out **line 263** of `MapTR/tools/fp16/train.py` to fix the bug of 'unexcepted key'.
- Modify the import of `bev_pool` and `bev_pool_v2` at **line 8** of `MapTR/projects/mmdet3d_plugin/maptr/modules/encoder.py` to match the updated module structure.
```bash
# from mmdet3d.ops import bev_pool
# from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from projects.mmdet3d_plugin.maptr.modules.ops.bev_pool import bev_pool 
from projects.mmdet3d_plugin.maptr.modules.ops.bev_pool_v2 import bev_pool_v2
```
## 3.Run Training and Evaluation
- refer to docs/prepare_dataset.md and train_eval.md

