# Towards Universal Texture Synthesis by Combining Spatial Noise Injection with Texton Broacasting in StyleGAN-2

The code is largely adapted from https://github.com/rosinality/stylegan2-pytorch

## Requirements

* PyTorch 1.9.0
* CUDA 11.1

## Dataset 

We collect 500 distinct textures with high spatial dimensions (around 2000 by 2000). Please download from [here](https://doi.org/10.5281/zenodo.7127079)

## Pretrained Model

A pretrained model (as describled in the paper) can be downloaded from [here](https://doi.org/10.5281/zenodo.8000592)

## Training

> python train.py --input PATH_TO_DATASET --iter 300000 --model_name texture --n_textons 16 --max_texton_size 64

## Generate samples

> python inference.py --n_textures 500 --samples_per_texture 10 --image_size 256,512,1024

## Calculate TIPP

Thresholded Invariant Pixel Percentage (TIPP) directly quantifies a commonly observed artifact (which we call spatial anchoring for regular textures. Please refer to our paper for details.). The root directory should contain multiple folders, each of which has different crops of the same textures.

> python TIPP.py --root PATH_TO_TESTSET --save_std_map

## Reference

Please cite the following paper if you use the provided data and/or code:

~~~bibtex
@article{LIN2023100092,
author = {Jue Lin and Gaurav Sharma and Thrasyvoulos N. Pappas},
title = {Toward universal texture synthesis by combining texton broadcasting with noise injection in StyleGAN-2},
journal = {e-Prime - Advances in Electrical Engineering, Electronics and Energy},
volume = {3},
pages = {100092},
year = {2023},
issn = {2772-6711},
doi = {https://doi.org/10.1016/j.prime.2022.100092},
url = {https://www.sciencedirect.com/science/article/pii/S2772671122000638},
}
~~~

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

## Disclaimer: 

The dataset and code are provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied. Use at your own risk.