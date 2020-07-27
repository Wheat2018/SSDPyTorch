# Lightweight-Face-Detection-Playground
Implentment several single stage lightweight face detetors.

## Anchor-Base methods
- [ ] FaceBoxes
- [ ] RetinaFace

## Anchor-Free methods
- [ ] FCOS
- [x] CenterNet

## WIDER FACE results
|      | Params|Flops(640X640)|Easy|Medium|Hard|
|:----:|:-----:|:-----:|:----:|:-----:|:-----:|
|fcosface  |   |     | | | |
|centerface|146.68k|299.31MMac|| | |

## Demo
![1_Handshaking_Handshaking_1_579_bbox](images/1_Handshaking_Handshaking_1_579_bbox.jpg)

## How to play
```
python models/centerface.py
```

## Reference
```
@inproceedings{zhang2017faceboxes,
  title = {Faceboxes: A CPU Real-time Face Detector with High Accuracy},
  author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.},
  booktitle = {IJCB},
  year = {2017}
}

@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}

@inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```
