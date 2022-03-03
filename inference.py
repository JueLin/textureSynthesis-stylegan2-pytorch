import math
import random
import os
import numpy as np

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from args import args

import utils

def inference_one_latent_and_save(generator, z=None, image_size=256, output=None, texture_idx=0, crop_idx=0):
    i_h, i_w =image_size//64, image_size//64
    with torch.no_grad():
        fake_img, _ = generator([z], input_is_latent=False, i_h=i_h, i_w=i_w)
        fake_img = utils.deprocess_image(fake_img)
        img_folder = os.path.join(output, "%04dby%04d"%(image_size,image_size), "%04d_th_crop"%texture_idx)
        utils.mkdir(img_folder)
        img_filename = os.path.join(img_folder,"%04d_th_texture_%04d_th_crop_%04dby%04d.png" % (texture_idx, crop_idx, image_size, image_size)) 
        save_image(fake_img[0], img_filename)

if __name__ == "__main__":
    device = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_name == "texture":
        from model import MultiScaleTextureGenerator
        generator = MultiScaleTextureGenerator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier, max_texton_size=args.max_texton_size, n_textons=args.n_textons)
    else:
        from model import Generator
        generator = Generator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier)
    generator.to(device)
    generator.eval()

    if args.load_ckpt is not None:
        print("Loading %s model from %s:" % (args.model_name, args.load_ckpt))
        ckpt = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        ckpt_name = os.path.splitext(os.path.basename(args.load_ckpt.strip("/")))[0]

    try:
        sample_z = torch.load(args.input).to(device)
        print("Successfully loaded pre-defined latent vectors for inference")
        latent_name = os.path.splitext(os.path.basename(args.input.strip("/")))[0]
        folder_path = os.path.join(args.output, "offline_inference", ckpt_name, latent_name, "seed"+str(int(args.seed))) 
        utils.mkdir(folder_path)
        for i in tqdm(range(sample_z.shape[0])):
            z = sample_z[i:i+1]
            for j in range(args.samples_per_texture):
                for img_size in args.image_size:
                    inference_one_latent_and_save(generator, z, img_size, folder_path, i, j)
    except:
        print("No pre-defined latent vectors provided. Latent vectors will be sampled online")
        folder_path = os.path.join(args.output, "online_inference", ckpt_name, "seed"+str(int(args.seed))) 
        utils.mkdir(folder_path)
        for i in tqdm(range(args.n_textures)):           
            z = torch.randn(1, args.latent_dim, device=device, requires_grad=False)
            for img_size in args.image_size:
                for j in range(args.samples_per_texture):   
                    inference_one_latent_and_save(generator, z, img_size, folder_path, i, j)
            z_filename = os.path.join(folder_path, "%09d.pt" % i)
            torch.save(z, z_filename)


