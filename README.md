
## Exploiting the patch manifold for inverse imaging

**Generative Patch Priors for Practical Compressive Image Recovery**  
[Rushil Anirudh](https://rushila.com/), [Suhas Lohit](https://suhaslohit.github.io/), [Pavan Turaga](https://pavanturaga.com/). In [WACV](https://openaccess.thecvf.com/content/WACV2021/html/Anirudh_Generative_Patch_Priors_for_Practical_Compressive_Image_Recovery_WACV_2021_paper.html), 2021.

<img src='https://rushilacom.files.wordpress.com/2021/01/color_figure_v2.jpg' width=1200>

### Overview of the approach for patch-based compressive sensing
<img src='https://rushilacom.files.wordpress.com/2021/01/presentation2-1.gif' width=1200>

### Dependencies
There are two versions of GPP, with python 3.6: 
* Pytorch `1.6.0` (also works with `1.4.0+`)
* Tensorflow `1.8.0`
We have included the corresponding patch-generators trained on CIFAR-32 for each framework. There are some performance differences; we report results from Tensorflow in the paper, but the PyTorch numbers are better on most examples (!!). 

The code also has the option of using BM3D as part of the _inverse patch transform_ in order to mitigate some of the patching artifacts. Any implementation should work, we used two of them -- [`pybm3d`](https://github.com/ericmjonas/pybm3d) and [`bm3d`](https://pypi.org/project/bm3d/). **GPP does not need it to work, but will work better with BM3D**.

This is ongoing work, if you find errors or bugs please let us know! 

### Description
This section will be updated in the coming days. Please see the paper for details about GPP and its workings.

## Citation
If you find this code useful in your work, please consider citing our paper:
```
@inproceedings{Anirudh2021GPP,
  title={Generative Patch Priors for Practical Compressive Image Recovery},
  author={Anirudh, Rushil and Lohit, Suhas and Turaga, Pavan},
  booktitle={WACV},
  year={2021}
}
```

### License
This code is distributed under the terms of the MIT license. All new contributions must be made under this license.
LLNL-CODE- 812404
SPDX-License-Identifier: MIT

