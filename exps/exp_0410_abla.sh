./tools/dist_train.sh configs/centernet3d/centernet3d_res50_kitti.py 1 31232 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res50_kitti.py 1 32232 --cfg-options optimizer.lr=1e-4 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res101_kitti.py 1 51232 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res101_kitti.py 1 61232 --cfg-options optimizer.lr=1e-4 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res18_kitti.py 1 31232 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res18_kitti.py 1 32232 --cfg-options optimizer.lr=1e-4 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res50_kitti.py 1 81232 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res50_kitti.py 1 38232 --cfg-options optimizer.lr=1e-4 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res101_kitti.py 1 58232 # 1
./tools/dist_train.sh configs/centernet3d/centernet3d_res101_kitti.py 1 68232 --cfg-options optimizer.lr=1e-4 # 1
