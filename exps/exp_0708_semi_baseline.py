./tools/dist_train.sh configs/centernet3d/centernet3d_dla34_kitti.py 1 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_dla34_kitti_color_aug.py 1 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_dla34_kitti_filter_hard.py 1 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_dla34_kitti_filter_hard_color_aug.py 1 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_dla34_kitti_filter_hardv2.py 1 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_dla34_kitti_filter_hardv2.py 1 --cfg-options data.train.dataset.remove_hard_instance_level=3 # 1
