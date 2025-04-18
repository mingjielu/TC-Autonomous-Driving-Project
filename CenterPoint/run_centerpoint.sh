cd mmdetection3d

CONFIG_FILE=configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py
# CONFIG_FILE=configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py
# CONFIG_FILE=configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py
# CONFIG_FILE=configs/centerpoint/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py
WORK_DIR=centerpoint_outputs
GPU_NUM=8

## single GPU training
# python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

## multi-GPU training
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --work-dir ${WORK_DIR}
