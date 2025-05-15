cd mmdetection3d

CONFIG_FILE=configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py
WORK_DIR=pointpillars_outputs
GPU_NUM=8

## single GPU training
# python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

## multi-GPU training
# ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --work-dir ${WORK_DIR}
## use torchrun instead of torch.distributed.launch under pytorch 2
./tools/dist_train_torchrun.sh ${CONFIG_FILE} ${GPU_NUM} --work-dir ${WORK_DIR}

## Evaluation
# CKPT=${WORK_DIR}/latest.pth
# LOG=${WORK_DIR}/pointpillars_eval_$(date +%Y%m%d_%H%M%S).log
# python tools/test.py ${CONFIG_FILE} ${CKPT} --eval bbox 2>&1 | tee ${LOG}
# ./tools/dist_test_torchrun.sh ${CONFIG_FILE} ${CKPT} ${GPU_NUM} --eval bbox 2>&1 | tee ${LOG}

