# Towards Universal Texture Synthesis by Combining Spatial Noise Injection with Texton Broacasting in StyleGAN-2

The code is largely adapted from https://github.com/rosinality/stylegan2-pytorch

## Requirements

* PyTorch 1.9.0
* CUDA 11.1

## Dataset 

We collect 500 distinct textures with high spatial dimensions (around 2000 by 2000). Please download from [here](https://drive.google.com/file/d/15tM8vlc-ZnYVQpyjf63QyQQ9inqtijmt/view?usp=sharing)

## Pretrained Model

A pretrained model (as describled in the paper) can be downloaded from [here](https://drive.google.com/file/d/1s0aZM9-IHMLFNIJOJXnRH6Y3iF5Grezu/view?usp=sharing)

## Training

> python train.py --input PATH_TO_DATASET --iter 300000 --model_name texture --n_textons 16 --max_texton_size 64

## Generate samples

> python inference.py --n_textures 500 --samples_per_texture 10 --image_size 256,512,1024

## Calculate TIPP

Thresholded Invariant Pixel Percentage (TIPP) directly quantifies a commonly observed artifact (which we call spatial anchoring for regular textures. Please refer to our paper for details.). The root directory should contain multiple folders, each of which has different crops of the same textures.

> python TIPP.py --root PATH_TO_TESTSET --save_std_map

## Reference

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2
