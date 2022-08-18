./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 4 1111 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 4 2111 --cfg-options model.use_grid_mask=False # 4
