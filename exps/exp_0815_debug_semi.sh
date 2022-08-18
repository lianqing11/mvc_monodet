bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_debugv4.py 2 # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_debugv5.py 2 # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_debugv6.py 2 # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_debugv4.py 1 --cfg-options data.samples_per_gpu=8 # 1
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_debugv5.py 1 --cfg-options data.samples_per_gpu=8 # 1

