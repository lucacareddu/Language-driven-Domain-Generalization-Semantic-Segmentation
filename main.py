import os
import datetime
timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import DGSSModel

from datasets import GTA5Dataset, CityscapesDataset
from datasets.transformscpu import *

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

from torch.utils.data import DataLoader

from utils import *

from tqdm import tqdm, trange
from torch.utils import tensorboard
import json


#################################################################################################

resume_path = "checkpoints/09-12_18-37-17/checkpoint-iter31999.pth"

if resume_path is not None:
    config = json.load(open(os.path.join('/'.join(resume_path.split("/")[:-1]), "config.json")))
    print("Resuming from: ", resume_path)
else:
    config = json.load(open("configs/config.json"))


debug = config["debug_mode"]

encoder_name = config["encoder"]["name"]
use_text = "clip" in encoder_name and config["encoder"]["use_text"]
gta_root = config["gta"]["remote_root"] if config["remote"] else config["gta"]["local_root"]
gta_inp_size = tuple(config["gta"]["input_size"])
city_root = config["city"]["remote_root"] if config["remote"] else config["city"]["local_root"]
city_inp_size = tuple(config["city"]["input_size"])
crop_size = tuple(config["preprocessing"]["crop_size"])
ignore_index = config["preprocessing"]["ignore_index"]
batch_size = config["training"]["batch_size"]
num_workers = config["training"]["num_workers"]
max_iterations = config["training"]["max_iterations"]
iters_per_val = config["training"]["iters_per_val"]
log_dir = config["training"]["log_dir"]
do_checkpoints = config["training"]["do_checkpoints"]
iters_per_save = config["training"]["iters_per_save"]
checkpoint_dir = config["training"]["checkpoint_dir"]
grad_clip = config["grad_clip"]["enable"]
grad_clip_value = config["grad_clip"]["small_model"] if encoder_name == "tiny_clip" else config["grad_clip"]["large_model"]
lr = config["optimizer"]["learning_rate"]
lr_power = config["optimizer"]["lr_power"]
lr_warmup_iters = config["optimizer"]["lr_warmup_iterations"]

#################################################################################################

SEED = 0

if True:
    fix_seed(SEED=SEED)

#################################################################################################

gta_augmentations = Compose([CentroidCCrop(crop_size)])
city_val_augmentations = Compose([TwoCropsCityVal(crop_size)])

train_gta = GTA5Dataset(root=gta_root, ignore_index=ignore_index, resize=gta_inp_size, transforms=gta_augmentations)
val_city = CityscapesDataset(root=city_root, split="val", ignore_index=ignore_index, resize=city_inp_size, transforms=city_val_augmentations)

gta_train_loader = DataLoader(train_gta, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
city_val_loader = DataLoader(val_city, batch_size=batch_size//2, num_workers=num_workers, collate_fn=collate_fn)

text_prompts = None

if use_text:
    text_prompts = [f"a photo of a {c}." for c in CITY_VALID_CLASSES]

#################################################################################################

model = DGSSModel(encoder_name=encoder_name, ignore_value=ignore_index, text_prompts=text_prompts, freeze_vision_encoder=True)
model.to(device)

model.print_trainable_params()
model.print_frozen_modules()

params = []

if "clip" in model.encoder_name and model.freeze_text_encoder:
    params.append({'name':"encoder", 'params': model.encoder.vision_model.parameters()})
else:
    params.append({'name':"encoder", 'params': model.encoder.parameters()})

params.append({'name':"neck", 'params': model.neck.parameters()})
params.append({'name':"vision_decoder", 'params': model.vision_decoder.parameters()})

if model.has_text_decoder:
    params.append({'name':"text_decoder", 'params': model.text_decoder.parameters()})
    
optimizer = torch.optim.AdamW(params, lr=lr)

#################################################################################################

if not debug:
    log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(log_dir)
    tb_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

    checkpoint_dir = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir)
    save_json(checkpoint_dir, config)
else:
    print("\nWARNING! PROCEDING IN DEBUG MODE (no logs are saved).\n")

#################################################################################################

iter_start = 0
if resume_path is not None:
    iter_start = resume_checkpoint(resume_path, model, optimizer)

train_iter = iter(gta_train_loader)


for i_iter in trange(iter_start, max_iterations):        
    model.train()
    adjust_learning_rate(lr=lr, lr_power=lr_power, i_iter=i_iter, warmup_iters=lr_warmup_iters, max_iterations=max_iterations, optimizer=optimizer)

    true_loss = 0
    for _ in range(1):
        batch, train_iter = get_batch(train_iter, gta_train_loader)

        images = batch["image"].to(device)
        classes = [x.to(device) for x in batch["classes"]]
        binmasks = [x.to(device) for x in batch["bin_masks"]]

        if 1:
            images = normalize(images)

        loss = model(pixel_values=images, bin_masks=binmasks, classes=classes)

        loss = loss / 1
        loss.backward()

        true_loss += loss.detach().item()
    
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

    optimizer.step()
    optimizer.zero_grad()

    if not debug:
        try:
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], i_iter)
            tb_writer.add_scalar("lr_dec", optimizer.param_groups[-1]["lr"], i_iter)
            tb_writer.add_scalar("Loss", true_loss, i_iter)
        except:
            pass

    if do_checkpoints and (i_iter+1) % iters_per_save == 0:
        if not debug:
            try:
                save_checkpoint(checkpoint_dir, i_iter, model, optimizer)
            except:
                pass

    if (i_iter+1) % iters_per_val == 0:
        print("Loss: ", true_loss)
        
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

                if 1:
                    images = normalize(images)

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
            
            if not debug:
                try:
                    tb_writer.add_scalar("Loss (Val):", mloss, i_iter)
                    tb_writer.add_scalar("mIoU (Val):", miou, i_iter)
                    tb_writer.add_scalar("mAcc (Val):", macc, i_iter)
                except:
                    pass

            del upsampled_logits, labels
