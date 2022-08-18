# MVC-MonoDet

This repo contains the implementations of [MVC-MonoDet](). Our implementations are built on top of MMDetection3D.

### Prerequisite
Please install/build the following package (follow the getting_started):
 + install mmcv==1.4.8 
 + build mmseg in ``software/``
 + build mmdet in ``software/``
 + build mmdet3d in ``software/``
 + build mvc_monodet with ``python setup.py develop``

We provide a simple script to conduct the above envirnment setup. For the details, please check the README.md of each package in ``software/``.
```
# 1. install MMVC with version of 1.4.8
# you may need to change the cuda version with replace cu111 and torch version with replace torch1.9.0
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 2. build mmdet
cd software/mmdet/
python setup.py develop

# 3. build mmseg
cd ../mmseg/
python setup.py develop

# 4. build mmdet3d
cd ../mmdet3d/
python setup.py develop

# 5. build the DCNv2
cd ../../det3d/models/backbones/DCNv2_t18/
bash install.sh
cd ../../../../

```

## Data

Please follow [data_prepartion](./data_prepartion.md) to prepare the training data.

## Training ane evaluation.

### 1. Training:

We provide the config of the standard supervised and semi-supervised training in ``configs/centernet/semi``

For the semi-supervised training setting, we provide the pre-trained script that first trained using the supervised training config in [google drive]().

### 2. Evaluation:
For the evaluation, please follow mmdet3d to evaluate the trained model.

For the model in MVC-MonoDet:
<!-- |  Backbone   | mAP (Easy) | Download |
| :---------: | :----: | :------: |
|[Baseline]()|21.99 |[model]() &#124; [log]()|
|[MVC-MonoDet (w/ semi-supervised training)]()|26.85 |[model]() &#124; [log]()| -->

|  Backbone   | mAP (Easy) |
| :---------: | :----: |
|[Baseline]()|21.99 |
|[MVC-MonoDet (w/ semi-supervised training)]()|26.85 |
If you find this repo useful for your research, please consider citing the papers
```
@inproceedings{
   mvc-monodet,
   title={Semi-Supervised Monocular 3D Object Detection by Multi-View Consistency},
   author={Lian, Qing and Xu, Yanbo and Yao, Weilong and Chen, Yingcong and Zhang, Tong},
   booktitle={ECCV},
   year={2022}
}
```

### Acknowledgement
The codebase is heavily leanred from sevreal open source repo: MMDetection3D, Detr3D, MonoFlex.