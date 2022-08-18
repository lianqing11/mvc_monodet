bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss.py 2 # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss.py 2 --cfg-options model.loss_semi_bbox.loss_mode="weighted_l1" # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss.py 2 --cfg-options model.loss_semi_bbox.match_threshold=0.1 # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_nocorner.py 2 --cfg-options model.loss_semi_bbox.loss_mode="weighted_l1" # 2
bash tools/dist_train.sh configs/centernet3d/semi/semi_stereo_centernet_dla34_kitti_box_loss_nocorner.py 2 --cfg-options model.loss_semi_bbox.loss_weight=0 # 2
