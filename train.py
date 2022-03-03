import math
import random
import os
import sys

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torchvision import utils as tvutils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from model import Discriminator
from dataset import TextureDataset, RandomMultiCrop, TextureDatasetLmdb
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from args import args
import utils

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return make_noise(batch, latent_dim, 1, device)

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def compute_gradient_penalty(real_samples, fake_samples, discriminator):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates).view(-1, 1)
    fake = torch.autograd.Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1, device="cuda").fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty            

def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
    pbar = range(args.iter+1)
    if get_rank() == 0:
        print("Training set path: %s, output directory path: %s" % (args.input, args.output))
        print("General settings: image size: %d, channel_multiplier: %d, lr: %.5f"%(args.image_size[0], args.channel_multiplier, args.lr))
        print("Intra-texture settings: textures_per_batch: %d, crops_per_texture: %d"%(args.textures_per_batch, args.crops_per_texture))
        print("Inter-texture settings: WGAN: True, gp weight: %.5f, n_critic: %d, noise_dx: %r, noise_std: %.5f"%(args.gp ,args.n_critic, args.noise_dx, args.noise_std))
        print("Augmentation settings: Random Flip: %r, Random 90-degree Rotattion: %r"%(args.random_flip, args.random_90_rotate))
        print("TextonBroadcast settings: number of textons per module: %d, max resolution to apply module: %d, phase noise is %r"%(args.n_textons, args.max_texton_size, args.random_phase_noise))
        pbar = tqdm(pbar, initial=args.start_iter, file=sys.stdout)

    d_loss_val, g_loss_val, loss_dict = 0, 0, {}
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator
    accum = 0.5 ** (32 / (10 * 1000))
    sample_z = torch.randn(args.batch_size, args.latent_dim, device=device) # Fixed latent vector for monitoring during training 
    phase_noise = None if args.random_phase_noise else 0
    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break
        ##### Training Discriminator ##############################################    
        requires_grad(generator, False)
        requires_grad(discriminator, True) 
        for j in range(args.n_critic):
            real_img = next(loader)
            real_img = real_img.to(device).view(-1, 3, args.image_size[0], args.image_size[0])
            noise = [mixing_noise(args.batch_size, args.latent_dim, args.mixing, device)]
            fake_img, _ = generator(noise, phase_noise=phase_noise)
            fake_pred = discriminator(fake_img.detach())
            real_pred = discriminator(real_img)
            gp_loss = args.gp*compute_gradient_penalty(real_img.detach(), fake_img.detach(), discriminator)
            d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gp_loss      
            loss_dict["d"] = d_loss.detach()
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()
            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

        ##### Training Generator ##################################################    
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        noise = [mixing_noise(args.batch_size, args.latent_dim, args.mixing, device)]
        fake_img, _ = generator(noise, phase_noise=phase_noise)
        fake_pred = discriminator(fake_img)
        g_loss = -torch.mean(fake_pred)
        loss_dict["g"] = g_loss.detach()
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        accumulate(g_ema, g_module, accum)

        ##### Logging #############################################################
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        if get_rank() == 0:
            pbar.set_description(f"d: {d_loss_val:4.4f}; g: {g_loss_val:4.4f}")
            if i % args.save_img_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z.to(device)])
                    tvutils.save_image(sample, f"{args.output}/{str(i).zfill(10)}.png", nrow=int(sample.size(0)**0.5), normalize=True, range=(-1, 1)) 
                    
            if wandb and args.wandb:
                logs = {"Generator": g_loss_val, "Discriminator": d_loss_val, "Real Score": real_score_val, "Fake Score": fake_score_val}
                if i % args.save_img_every == 0:
                    sample = utils.deprocess_image(sample.detach())
                    images = wandb.Image(sample, caption="Generated textures")
                    logs["images"] = images
                wandb.log(logs)    

            if i % args.save_ckpt_every == 0:
                torch.save({"g": g_module.state_dict(), "d": d_module.state_dict(), "g_ema": g_ema.state_dict(), "g_optim": g_optim.state_dict(), "d_optim": d_optim.state_dict(), "args": args}, f"{args.output}/{str(i).zfill(10)}.pt")

if __name__ == "__main__":
    device = args.device     
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if args.model_name == "texture":
        from model import MultiScaleTextureGenerator
        generator = MultiScaleTextureGenerator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier, max_texton_size=args.max_texton_size, n_textons=args.n_textons)
        g_ema = MultiScaleTextureGenerator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier, max_texton_size=args.max_texton_size, n_textons=args.n_textons)
    else:
        from model import Generator
        generator = Generator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier)
        g_ema = Generator(size=args.image_size[0], style_dim=args.latent_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier)
    generator.to(device), g_ema.to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    discriminator = Discriminator(args.image_size[0], channel_multiplier=args.channel_multiplier, add_noise=args.noise_dx).to(device)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

    if args.load_ckpt is not None:
        print("load model:", args.load_ckpt)
        ckpt = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.load_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name.strip("/"))[0])
            pass
        except ValueError:
            pass 
        generator.load_state_dict(ckpt["g"], strict=False) if "g" in ckpt else None
        discriminator.load_state_dict(ckpt["d"], strict=False) if "d" in ckpt else None
        g_ema.load_state_dict(ckpt["g_ema"], strict=False) if "g_ema" in ckpt else None
        g_optim.load_state_dict(ckpt["g_optim"]) if "g_optim" in ckpt else None
        d_optim.load_state_dict(ckpt["d_optim"]) if "d_optim" in ckpt else None
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")  
    args.output = os.path.join(args.output, "training", args.model_name,"start_iter_%08d"%args.start_iter, timestr)
    utils.mkdir(args.output) 

    train_sub_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)])
    train_transforms = transforms.Compose([RandomMultiCrop(args.image_size[0], args.crops_per_texture, args.random_90_rotate, args.random_flip), transforms.Lambda(lambda crops: torch.stack([train_sub_transforms(crop) for crop in crops]))])
    
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        dataset = TextureDatasetLmdb(args.input, train_transforms, args.image_size[0])
    else:
        dataset = TextureDataset(args.input, train_transforms, args.image_size[0])        

    loader = data.DataLoader(dataset, batch_size=args.textures_per_batch, sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed), drop_last=True)

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan2 texture")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)