# SegStereo

Caffe implementation of SegStereo and ResNetCorr models.

## Requirements

This code is tested with Caffe, CUDA 8.0 and Ubuntu 16.04.

* Basic caffe implementation is from [Caffe](https://github.com/BVLC/caffe).
* The correlation and correlation1d layers are from [FlowNet 2.0](https://github.com/lmb-freiburg/flownet2).
* The Interp layer is from [PSPNet](https://github.com/hszhao/PSPNet).
* The disparity tool is from [OpticalFlowToolkit](https://github.com/liruoteng/OpticalFlowToolkit)

## Data
Our models require rectified stereo pairs. We provide several examples in `data` directory

* [KITTI Stereo 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI Stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Cityscapes](https://www.cityscapes-dataset.com/downloads/)

## Models

* ResNetCorr\_SRC\_pretrain.caffemodel: [Google Drive](https://drive.google.com/open?id=18s1WwVwo1T9i7Mfpy8ioV-ZdB3imHIO1)
* SegStereo\_SRC\_pretrain.caffemodel: [Google Drive](https://drive.google.com/open?id=1lIb2DzKSnbFq4V75QNYfJBsGrAxZTftq)
* SegStereo\_pre\_corr\_SRC\_pretrain.caffemodel: [Google Drive](https://drive.google.com/file/d/1SdurOp3OxXSQem0jeKVXVIh0FCLpFh9P/view?usp=sharing)
* ResNetCorr\_KITTI\_finetune.caffemodel: Google Drive
* SegStereo\_KITTI\_finetune.caffemodel: Google Drive
* SegStereo\_pre\_corr\_KITTI\_finetune.caffemodel: [Google Drive](https://drive.google.com/file/d/1oOm4hTaKgJdScfhUVbhDJ0cAcuQU__Ru/view?usp=sharing)

## Evaluation

To test or evaluate the disparity model, you can use the script in `model/get_disp.py`. We recommend that you put the model under correponding directory.
```
python get_disp.py --model_weights ./ResNetCorr/ResNetCorr_SRC_pretrain.caffemodel --model_deploy ./ResNetCorr/ResNetCorr_deploy.prototxt --data ../data/KITTI --result ./ResNetCorr/result/kitti --gpu 0 --colorize --evaluate
```

## Reference

* If our **SegStereo** or **ResNetCorr** models help your research, please consider citing:

```
@inproceedings{yang2018SegStereo,
  author    = {Yang, Guorun and
               Zhao, Hengshuang and
               Shi, Jianping and
               Deng, Zhidong and
               Jia, Jiaya},
  title     = {SegStereo: Exploiting Semantic Information for Disparity Estimation},
  booktitle = ECCV,
  year      = {2018}
}
```

* If you find our synthetic realistic collaborative (SRC) training strategy useful, please consider citing:

```
@inproceedings{yang2018srcdisp,
  author    = {Yang, Guorun and
               Deng, Zhidong and
               Lu, Hongchao and
               Li, Zeping},
  title     = {SRC-Disp: Synthetic-Realistic Collaborative Disparity Learning for Stereo Mathcing},
  booktitle = ACCV,
  year      = {2018}
}
```

## Questions

Please contact ygr13@mails.tsinghua.edu.cn
