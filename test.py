import os
import datetime
timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import DGSSModel

from datasets import GTA5Dataset, CityscapesDataset
from datasets.transformscpu import *
from datasets import transformsgpu

from torch.utils.data import DataLoader

from utils import *

from tqdm import tqdm, trange
from torch.utils import tensorboard
import json


#################################################################################################

resume_path = None

if resume_path is not None:
    config = json.load(open(os.path.join('/'.join(resume_path.split("/")[:-1]), "config.json")))
    print("Resuming from: ", resume_path)
else:
    config = json.load(open("configs/config.json"))

encoder_name = config["encoder"]
use_text = "clip" in encoder_name and config["use_text"]
gta_inp_size = tuple(config["gta"]["input_size"])
city_inp_size = tuple(config["city"]["input_size"])
rcs_enabled = config["rcs"]["enable"]
rcs_temperature = config["rcs"]["temperature"]
ignore_index = config["preprocessing"]["ignore_index"]
crop_size = tuple(config["preprocessing"]["crop_size"])
batch_size = config["training"]["batch_size"]
num_workers = config["training"]["num_workers"]
max_iterations = config["training"]["max_iterations"]
iters_per_val = config["training"]["iters_per_val"]
log_dir = config["training"]["log_dir"]
do_checkpoints = config["training"]["do_checkpoints"]
iters_per_save = config["training"]["iters_per_save"]
checkpoint_dir = config["training"]["checkpoint_dir"]
lr = config["optimizer"]["learning_rate"]
lr_power = config["optimizer"]["lr_power"]
lr_warmup_iters = config["optimizer"]["lr_warmup_iterations"]

#################################################################################################

SEED = 0

if True:
    fix_seed(SEED=SEED, deterministic=True)

#################################################################################################

gta_augmentations = Compose([CentroidCCrop(crop_size)])
city_val_augmentations = Compose([TwoCropsCityVal(crop_size)])

gta_root_path = "/home/luca/data/gta" #"/home/thesis/datasets/GTAV" #
city_root_path = "/home/luca/data/cityscapes" #"/home/thesis/datasets/Cityscapes" #

train_gta = GTA5Dataset(root=gta_root_path, ignore_index=ignore_index, resize=gta_inp_size, transforms=gta_augmentations, rcs=rcs_enabled, rcs_temp=rcs_temperature)
val_city = CityscapesDataset(root=city_root_path, split="val", ignore_index=ignore_index, resize=city_inp_size, transforms=city_val_augmentations)

gta_train_loader = DataLoader(train_gta, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
city_val_loader = DataLoader(val_city, batch_size=batch_size//2, num_workers=num_workers, collate_fn=collate_fn)

text_prompts = None

if use_text:
    if True:
        print("Class definitions employed.")
        with open("class_definition/class_definition.json","r") as f:
            class_definition = json.load(f)
            class_definition = df_dict_search(dictionary=class_definition, class_names=CITY_VALID_CLASSES)
            text_prompts = [f"{c}: " + class_definition[c] for c in CITY_VALID_CLASSES]
            # print([len(x) for x in text_prompts])
    else:
        print("Class names employed.")
        text_prompts = [f"a photo of a {c}." for c in CITY_VALID_CLASSES]

#################################################################################################

model = DGSSModel(encoder_name=encoder_name, ignore_value=ignore_index, text_prompts=text_prompts)
model.to(device)

model.print_trainable_params()
model.print_frozen_modules()

params = []

if "clip" in model.encoder_name and model.freeze_text_encoder:
    params.append({'params': model.encoder.vision_model.parameters()})
else:
    params.append({'params': model.encoder.parameters()})

params.append({'params': model.neck.parameters()})
params.append({'params': model.vision_decoder.parameters(), 'lr': lr * 10})

if model.has_text_decoder:
    params.append({'params': model.text_decoder.parameters()})#, 'lr': lr * 10})
    
optimizer = torch.optim.AdamW(params, lr=lr)

#################################################################################################

log_dir = os.path.join(log_dir, timestamp)
os.makedirs(log_dir)
tb_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

#################################################################################################

import re
import glob

def f3(string):
    return re.findall(r'[0-9]+', string)

path = "checkpoints/20-11_11-45-24"

files = sorted(glob.glob(f"{path}/*.pth"), key = lambda x: int(f3(x)[-1]))

for resume_path in files:
    i_iter = resume_checkpoint(resume_path, model, optimizer) - 1

    print()
    print(path, " : ", i_iter)
    print()
    
    model.eval()
    with torch.no_grad():
        runn_loss = torch.zeros((1)).to(device)
        runn_bins = torch.zeros((3, 19)).to(device)
        loop = tqdm(city_val_loader, leave=False)
        
        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            classes = [x.to(device) for x in batch["classes"]]
            binmasks = [x.to(device) for x in batch["bin_masks"]]                

            if False:
                images = transformsgpu.normalize(images, mean=IN_MEAN, std=IN_STD)

            loss, upsampled_logits = model(pixel_values=images, bin_masks=binmasks, classes=classes, return_logits=True)

            upsampled_logits = upsampled_logits.detach()

            runn_loss.add_(loss)
            runn_bins.add_(get_confBins(predictions=upsampled_logits, references=labels, ignore_index=ignore_index))
        
        mloss = runn_loss.item() / len(city_val_loader)
        jaccard, accuracy = get_metrics(runn_bins)
        miou = torch.nanmean(jaccard).item()
        macc = torch.nanmean(accuracy).item()

        perclass_repr(torch.stack((jaccard, accuracy)).cpu().numpy().transpose())
        print("Loss (Val): ", mloss)
        print("mIoU (Val): ", miou)
        print("mAcc (Val): ", macc)
        
        try:
            tb_writer.add_scalar("Loss (Val):", mloss, i_iter)
            tb_writer.add_scalar("mIoU (Val):", miou, i_iter)
            tb_writer.add_scalar("mAcc (Val):", macc, i_iter)
        except:
            pass

        del upsampled_logits, labels
