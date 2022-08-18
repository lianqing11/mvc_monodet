./tools/dist_train.sh configs/detr3d/detr3d_res50_nuscenes_vanilla.py 4 12328 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res50_nuscenes_vanilla.py 4 22328 --cfg-options optimizer.lr=2e-4 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res50_nuscenes_vanilla.py 4 32328 --cfg-options model.pts_bbox_head.num_query=1600 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res50_nuscenes_vanilla.py 4 42328 --cfg-options data.train.times=2 # 4

