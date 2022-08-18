./tools/dist_train.sh configs/detr3d/detr3d_res50_nuscenes_halfres.py 4 32322 --cfg-options data.samples_per_gpu=1 # 4
./tools/dist_train.sh configs/detr3d/detr3d_res50_nuscenes_halfres_vanilla.py 4 42322 --cfg-options data.samples_per_gpu=1 # 4
