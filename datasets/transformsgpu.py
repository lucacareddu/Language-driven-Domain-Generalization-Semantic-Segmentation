# Adapted from DACS (Domain Adaptation via Cross-domain Mixed Sampling) https://arxiv.org/abs/2007.08702

import numpy as np
import kornia
import torch


def colorJitter(images, p=0.5, value=0.25):
    if torch.rand(1) < p:
        # pixel values expected to be in [0,1]
        f = kornia.augmentation.ColorJiggle(brightness=value,contrast=value,saturation=value,hue=value)
        images = f(images)
    return images

def gaussianBlur(images, p=0.5):
    if torch.rand(1) < p:
        sigma = np.random.uniform(0.15,1.15)
        kernel_size_y = int(np.floor(np.ceil(0.1 * images.shape[2]) - 0.5 + np.ceil(0.1 * images.shape[2]) % 2))
        kernel_size_x = int(np.floor(np.ceil(0.1 * images.shape[3]) - 0.5 + np.ceil(0.1 * images.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        f = kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma))
        images = f(images)
    return images

def flip(images, labels, p=0.5):
    if torch.rand(1) < p:
        images = torch.flip(images,[2])
        labels = [torch.flip(x,[1]) for x in labels] if isinstance(labels, list) else torch.flip(labels,[1])
    return images, labels

# def mix(masks, images = None, labels = None):
#     #Mix
#     if not (images is None):
#         if masks.shape[0] == images.shape[0]:
#             images = torch.cat([(masks[i] * images[i] + (1 - masks[i]) * images[(i + 1) % images.shape[0]]).unsqueeze(0) for i in range(images.shape[0])])
#         elif masks.shape[0] == images.shape[0] / 2:
#             images = torch.cat((torch.cat([(masks[i] * images[2 * i] + (1 - masks[i]) * images[2 * i + 1]).unsqueeze(0) for i in range(int(images.shape[0] / 2))]),
#                               torch.cat([((1 - masks[i]) * images[2 * i] + masks[i] * images[2 * i + 1]).unsqueeze(0) for i in range(int(images.shape[0] / 2))])))
#     if not (labels is None):
#         labels = torch.cat([(masks[i] * labels[i] + (1 - masks[i]) * labels[(i + 1) % labels.shape[0]]).unsqueeze(0) for i in range(labels.shape[0])])
#     return images, labels

# def oneMix(mask, images, labels):
#     #Mix
#     mask_images, _ = torch.broadcast_tensors(mask[0], images[0])
#     images = (mask_images * images[0] + (1-mask_images) * images[1])
#     mask_labels, _ = torch.broadcast_tensors(mask[0], labels[0])
#     labels = (mask_labels * labels[0] + (1-mask_labels) * labels[1])
#     return images.unsqueeze(0), labels.unsqueeze(0)

# def cowMix(masks, images = None, labels = None):
#     #Mix
#     if not (images is None):
#         stackedMask, images = torch.broadcast_tensors(masks, images)
#         stackedMask = stackedMask.clone()
#         stackedMask[1::2]=1-stackedMask[1::2]
#         images = (stackedMask*torch.cat((images[::2],images[::2]))+(1-stackedMask)*torch.cat((images[1::2],images[1::2]))).float()
#     if not (labels is None):
#         stackedMask, labels = torch.broadcast_tensors(masks, labels)
#         stackedMask = stackedMask.clone()
#         stackedMask[1::2]=1-stackedMask[1::2]
#         labels = (stackedMask*torch.cat((labels[::2],labels[::2]))+(1-stackedMask)*torch.cat((labels[1::2],labels[1::2]))).float()
#     return images, labels

def normalize(images, mean, std):
    std = (255,255,255)
    images *= 255
    STD = torch.Tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(images.device)
    MEAN = torch.Tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(images.device)
    STD, _ = torch.broadcast_tensors(STD, images)
    MEAN, _ = torch.broadcast_tensors(MEAN, images)
    images = ((images-MEAN)/STD).float()
    return images


def train_aug(images, labels, mean=None, std=None, normalization=True):
    if normalization:
        assert mean is not None and std is not None
    images, labels = flip(images, labels)
    images = colorJitter(images)
    # images = gaussianBlur(images)
    if normalization:
        images = normalize(images=images, mean=mean, std=std)
    return images, labels

def eval_aug(images, labels, mean=None, std=None, normalization=True):
    if normalization:
        assert mean is not None and std is not None
    images, labels = flip(images, labels)
    if normalization:
        images = normalize(images=images, mean=mean, std=std)
    return images, labels
