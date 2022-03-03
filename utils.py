import torch
import os
from PIL import Image

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def mkdir(path=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch_vgg(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # batch = batch.div_(255.0)
    return (batch - mean) / std

def denormalize_batch_vgg(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return batch * std + mean

def deprocess_image(batch, in_high=1, in_low=-1, out_high=1, out_low=0):
    # from [in_low, in_high] to [0, 1] then to [out_low, out_high]
    assert in_high>in_low, "Invalid input range"
    assert out_high>out_low, "Invalid output range"
    return (batch-in_low)/(in_high-in_low)*(out_high-out_low)+out_low