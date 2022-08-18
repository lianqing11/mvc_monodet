./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 12322
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 22322 --cfg-options model.nocs_head.supervised_loss_weight=1.
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 42322 --cfg-options model.nocs_head.tanh_activate=True
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 32322 --cfg-options model.nocs_head.supervised_loss_weight=1. model.nocs_head.tanh_activate=True
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 15322
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 25322 --cfg-options model.nocs_head.supervised_loss_weight=1.
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 45322 --cfg-options model.nocs_head.tanh_activate=True
./tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1 35322 --cfg-options model.nocs_head.supervised_loss_weight=1. model.nocs_head.tanh_activate=True
