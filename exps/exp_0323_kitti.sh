./tools/dist_train.sh configs/centernet3d/centernet3d_kitti.py 1 12328 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_kitti.py 1 22328 --seed 1232 # 1
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 2328 # 1
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 3328 --cfg-options model.nocs_head.tanh_activate=True # 1
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 4328 --seed 123 # 1
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 5328 --seed 123 --cfg-options model.nocs_head.tanh_activate=True # 1
./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 2 1111 # 2
./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 2 2111 --cfg-options model.use_grid_mask=False # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-car.py 2 32328 # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-car.py 2 42328 --seed 1234 # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-3class.py 2 52328 # 2
./tools/dist_train.sh configs/monovoxel/monovoxel_4x8_kitti-3class.py 2 62328 --seed 1234 # 2
