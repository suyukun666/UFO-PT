## General Object Pose Transformation Network from Unpaired Data

## Introduction

Object pose transformation is a challenging task. Yet, most existing pose transformation networks only focus on synthesizing humans. These methods either rely on the keypoints information or rely on the manual annotations of the paired target pose images for training. However, collecting such paired data is laboring and the cue of keypoints is inapplicable to general objects. In this paper, we address a problem of novel general object pose transformation from unpaired data. Given a source image of an object that provides appearance information and a desired pose image as reference in the absence of paired examples, we produce a depiction of that object in that pose, retaining the appearance of both the object and background. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660286.pdf)]

## Demo

Download the *Bird* checkpoint from [here](https://drive.google.com/drive/folders/1YifKgVu0GUY3IQsY8Mju8ksBzxFzCCs_?usp=sharing) and save them in `code/checkpoints/bird` and execute the following command, find the results in `code/output/test/bird`

- Install the dependencies

```
cd ./code
pip install -r requirements.txt
```

- Install the [Synchronized-BatchNorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) Library

```
cd ./models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

- Then run the command

```
sh ./demo.sh
```

## Training

Download the VGG checkpoint from [here](https://drive.google.com/file/d/1IalcghHLCQ8hvhJDnw6HC7Y18Nv3Zuui/view?usp=sharing) and save them in `code/models` 

- Then run the command

```
cd ./code
python train.py --name bird --dataset_mode bird --dataroot bird_data/ --niter 100 --niter_decay 100 --use_attention --maskmix --noise_for_mask --mask_epoch 150 --warp_mask_losstype direct --weight_mask 100.0 --PONO --PONO_C --vgg_normal_correct --batchSize 1 --gpu_ids 0
```

## Result

- #### Video Imitation

<img src="./asset/horse.gif" width="50%"/><img src='./asset/sheep.gif' width="50%">

- #### More Visualization

<img src="./asset/animal.png">

## Acknowledgement

Our code is heavily borrowed from [CoCosNet](https://github.com/microsoft/CoCosNet). We also thanks  [VTON](https://github.com/sergeywong/cp-vton). Many thanks for their hard work.

