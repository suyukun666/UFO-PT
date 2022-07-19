## General Object Pose Transformation Network from Unpaired Data

## Introduction

Object pose transformation is a challenging task. Yet, most existing pose transformation networks only focus on synthesizing humans. These methods either rely on the keypoints information or rely on the manual annotations of the paired target pose images for training. However, collecting such paired data is laboring and the cue of keypoints is inapplicable to general objects. In this paper, we address a problem of novel general object pose transformation from unpaired data. Given a source image of an object that provides appearance information and a desired pose image as reference in the absence of paired examples, we produce a depiction of that object in that pose, retaining the appearance of both the object and background.

## Demo

Download the *Bird* checkpoint from [here](https://drive.google.com/drive/folders/1YifKgVu0GUY3IQsY8Mju8ksBzxFzCCs_?usp=sharing) and save them in `code/checkpoints/bird` and execute the following command, find the results in `code/output/test/bird`

- #### Install the [Synchronized-BatchNorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) Library

```
cd ./code/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

Run

```
sh ./demo.sh
```

## Result

- #### Video Imitation

<img src="./asset/horse.gif" width="50%"/><img src='./asset/sheep.gif' width="50%">

- #### More Visualization

<img src="./asset/animal.png">

## Acknowledgement

Our code is heavily borrowed from [CoCosNet](https://github.com/microsoft/CoCosNet). We also thanks  [VTON](https://github.com/sergeywong/cp-vton). Many thanks for their hard work.

