mkdir ckpts && cd ckpts

# Pretrained weights of bevformer
# Also the initial state of training stage1 model
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth

# Pretrained weights of stage1 model (perception part of UniAD)
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth

# Pretrained weights of stage2 model (fully functional UniAD)
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
