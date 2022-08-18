./tools/dist_train.sh configs/detr3d/nocs_detr3d_res101_nuscenes_halfres.py 4 32322 # 4
./tools/dist_train.sh configs/detr3d/nocs_detr3d_res101_nuscenes.py 4 42322 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res101_nuscenes_halfres.py 4 52322 --cfg-options model.pts_bbox_head.num_query=500 # 4