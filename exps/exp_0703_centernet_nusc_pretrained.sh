./tools/dist_train.sh configs/monovoxel/monovoxel_r101_1x8_nuscenes_centerhead_aug_pretrained.py 8 # 8
./tools/dist_train.sh configs/monovoxel/monovoxel_r101_1x8_nuscenes_centerhead_aug_pretrained.py 8 --cfg-options data.samples_per_gpu=2 # 8
