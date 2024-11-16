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

debug = config["debug_mode"]

#################################################################################################

SEED = 0

if True:
    fix_seed(SEED=SEED, deterministic=True)

#################################################################################################

gta_augmentations = Compose([CentroidCCrop(crop_size)])
city_val_augmentations = Compose([TwoCropsCityVal(crop_size)])

gta_root_path = "/home/thesis/datasets/GTAV" #"/home/luca/data/gta"
city_root_path = "/home/thesis/datasets/Cityscapes" #"/home/luca/data/cityscapes"

train_gta = GTA5Dataset(root=gta_root_path, ignore_index=ignore_index, resize=gta_inp_size, transforms=None, rcs=rcs_enabled, rcs_temp=rcs_temperature, stats=True)
val_city = CityscapesDataset(root=city_root_path, split="val", ignore_index=ignore_index, resize=None, transforms=city_val_augmentations)

gta_train_loader = DataLoader(train_gta, batch_size=1, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
city_val_loader = DataLoader(val_city, batch_size=batch_size//2, num_workers=num_workers, collate_fn=collate_fn)


from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.colors import CITY_VALID_CLASSES

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

results = dict()

iter_resume = 0

model.eval()
with torch.no_grad():
    loop = tqdm(gta_train_loader)
    
    for i, batch in enumerate(loop):

        if i < iter_resume:
            continue

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        fnames = batch["fname"]

        classes = torch.unique(labels)
        classes = classes[classes != ignore_index]

        results[fnames[0]] = dict()

        # print(fnames[0])

        # conditional image captioning
        for c in classes:
            prompt = f"there is {tau}" if (tau := CITY_VALID_CLASSES[c]) in ["vegetation","terrain","sky"] else f"there is a {tau}"

            inputs = processor(images, prompt, return_tensors="pt", do_rescale=False, padding=True).to("cuda")

            out = model.generate(**inputs)

            batch_text = processor.batch_decode(out, skip_special_tokens=True)

            # print(fnames[0], batch_text[0])
            
            results[fnames[0]][c.item()] = batch_text[0]

        if i%100 == 0:
            with open(f'captions_{timestamp}.json', 'w') as of:
                json.dump(results, of, indent=2)

with open(f'captions_{timestamp}.json', 'w') as of:
    json.dump(results, of, indent=2)
