# TRNET19

Mission for Hairui Zhu:
  how to test and train? 
  where to find pre-trained models?

## Requirements

The code runs with Python 3.7, Pytorch 1.3.0, CUDA 10.1 The following additional dependencies need to be installed:

* [PyTorch](https://pytorch.org/)
* Numpy


## Download dataset

Run `python data/download_pclouds.py` to download PCPNET data.

## Model

Download pre-trained model from this [link]()

## Normal Estimation

To estimate point cloud properties using default settings:
```
python calculate.py
```

## Traning

To train on the PCPNet train dataset, run train.sh and note the following parameters in train.py:

* --name: Model file name to store.

* --patch_radius: single value for single-scale models and multiple values for multi-scale model.

* --indir2: if in_points_dim=6, it should not empty.

* --generate_points_num: number of points output form the net.

* --hypotheses number: of planes hypotheses sampled for each patch.


Example:

```
python train.py --name='' --patch_radius='k256_s007_nostd_sumd_pt32_pl32_num_c' --points_per_patch=[256] --patch_point_count_std=0.0 --patch_radius=[0.07] --sym_op='sumd --generate_points_num=32   --hypotheses=32 
```

## Evaluate




