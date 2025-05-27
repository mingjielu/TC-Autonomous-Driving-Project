DATAPATH=/mnt/nvme0/data/datasets/imagenet # modify to your own path
LOGPATH=./resnet50_bs128.log
MODEL=resnet50
GBS=128
torchrun --nproc_per_node=8 vision/references/classification/train.py --data-path $DATAPATH \
        --model $MODEL --epochs 300 --batch-size $GBS --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224 2>&1 | tee $LOGPATH
