# SegStereo

Caffe implementation of SegStereo and ResNetCorr models by Guorun Yang. 

If you find the code useful for your research, please consider citing our latest works:
* SegStereo: 

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

* ResNetCorr or synthetic-realistic collaborative disparity learning (SRC-Disp)

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

## Usage

### Requirements

This code was tested with Caffe, CUDA 8.0 and Ubuntu 16.04. 

Notes
	- Basic caffe implementation is from [Caffe](https://github.com/BVLC/caffe).
	- The Correlation and Correlation1D layers are from [FlowNet 2.0](https://github.com/lmb-freiburg/flownet2).
	- The Interp layer is from [PSPNet](https://github.com/hszhao/PSPNet).

### Evaluation
	- Download datasets. We provide several examples in folder 'data/KITTI_2015' and 'data/FlyingThings3D'.
	- Download trained models and put them in corresponding folder 'model/ResNetCorr' and 'model/SegStereo':
		- ResNetCorr\_SRC\_pretrain.caffemodel: [Google Drive](https://drive.google.com/open?id=18s1WwVwo1T9i7Mfpy8ioV-ZdB3imHIO1)
		- SegStereo\_SRC\_pretrain.caffemodel: [Google Drive](https://drive.google.com/open?id=1lIb2DzKSnbFq4V75QNYfJBsGrAxZTftq)

## Questions

Please contact 'ygr13@mails.tsinghua.edu.cn'