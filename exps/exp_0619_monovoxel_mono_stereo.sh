./tools/dist_train.sh configs/imvoxelnet/imvoxelnet_4x8_kitti-3class.py 2 42322 # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-3class_noflip.py 2 14333 --cfg-options model.filter_invalid_point=True # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-stereo-3class_noflip.py 2 15322 --cfg-options model.filter_invalid_point=True # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-stereo-3class_noflip.py 2 12322 # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-3class_noflip.py 2 14333 # 2

