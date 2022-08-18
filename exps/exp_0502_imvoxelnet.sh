./tools/dist_train.sh configs/monovoxel/imvoxelnet_r101_1x8_nuscenes_centerhead.py 8 # 8
./tools/dist_train.sh configs/monovoxel/imvoxelnet_r101_1x8_nuscenes_centerhead_aug.py 8 # 8
./tools/dist_train.sh configs/monovoxel/imvoxelnet_r101_1x8_nuscenes_centerhead_aug.py 8 --cfg-options valid_sample=True # 8