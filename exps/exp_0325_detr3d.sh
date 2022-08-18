./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 4 1111 --cfg-options load_model=ckpts/fcos3d.pth data.train.times=20 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 4 2111 --cfg-options load_model=ckpts/fcos3d.pth model.use_grid_mask=False data.train.times=20 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 4 1111 --cfg-options data.train.times=20 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res101_gridmask_kitti.py 4 2111 --cfg-options model.use_grid_mask=False data.train.times=20 # 4
