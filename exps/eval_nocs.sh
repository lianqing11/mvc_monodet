bash tools/dist_eval_loss.sh configs/centernet3d/nocs/centernet3d_nocs_kitti_vis.py work_dirs/centernet3d_nocs_kitti/0607_1542-/epoch_20.pth
bash tools/dist_eval_loss.sh configs/centernet3d/nocs/centernet3d_nocs_kitti_vis.py work_dirs/centernet3d_nocs_kitti/0607_1552-model.nocs_head.tanh_activate/epoch_20.pth --cfg-options model.nocs_head.tanh_activate=True
bash tools/dist_eval_loss.sh configs/centernet3d/nocs/centernet3d_nocs_kitti_vis.py work_dirs/centernet3d_nocs_kitti/0607_1547-model.nocs_head.supervised_loss_weight:1.0-/epoch_20.pth 
bash tools/dist_eval_loss.sh configs/centernet3d/nocs/centernet3d_nocs_kitti_vis.py work_dirs/centernet3d_nocs_kitti/0607_1557-model.nocs_head.supervised_loss_weight:1.0-model.nocs_head.tanh_activate:True-/epoch_20.pth --cfg-options model.nocs_head.tanh_activate=True



