# add the parent directory of `my_hooks` to PYTHONPATH
export PYTHONPATH=${PWD}:${PYTHONPATH}

cd /Your/Path/Of/MMDetection3d

# python download_nuscenes.py
# python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes

# add custom_imports and custom_hooks to the config
CONFIG_FILE=${PWD}/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d-profile.py
WORK_DIR=${PWD}/profiler_logs

# CONFIG_FILE=${PWD}/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py
# CONFIG_FILE=${PWD}/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
# WORK_DIR=${PWD}/my_scripts/pointpillars_kitti
# GPU_NUM=8
# PORT=3179 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --work-dir ${WORK_DIR}
python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

## Analysis Training Performance
## Ref: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/zh_cn/user_guides/useful_tools.md
# LOG_JSON=${WORK_DIR}/20250417_063606.log.json
# python tools/analysis_tools/analyze_logs.py cal_train_time ${LOG_JSON}
# python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
