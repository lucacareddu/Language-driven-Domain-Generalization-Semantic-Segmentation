import sys
sys.path.insert(0,"/home/luca/work")

import os

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader

from tqdm import tqdm
import json

from utils.trainutils import fix_seed, collate_fn
from utils.colors import CITY_VALID_CLASSES


def get_stats(dataloader, batch_size, img_size, image_stats=True, label_stats=True, MIN_PIXELS=782, dataset_name="Dataset", out_dir=None):
    if image_stats:
        psum = torch.zeros((3,)).to(device)
        psum_sq = torch.zeros((3,)).to(device)
    if label_stats:
        ccount = torch.zeros((19,)).to(device)
        cpopul = [[] for _ in range(19)]
        ccentr = [[] for _ in range(19)]

    for batch in tqdm(dataloader):
        images, labels, names = batch["image"].to(device), batch["label"].to(device), batch["fname"] 

        if image_stats:
            psum.add_(images.sum(axis = [0, 2, 3]))
            psum_sq.add_((images ** 2).sum(axis = [0, 2, 3]))

        if label_stats:
            for lbl, name in zip(labels, names):
                class_bin = torch.bincount(lbl.flatten(), minlength=19)[:19]
                for c in torch.where(class_bin > MIN_PIXELS)[0].cpu().numpy():
                    cpopul[c].append(name)
                    centroid = torch.mean(torch.argwhere(lbl == c).float(), dim=(0,)).int()
                    ccentr[c].append(centroid.cpu().numpy().tolist())
                ccount.add_(class_bin)

    length = len(dataloader) * batch_size * img_size[0] * img_size[1]

    print(f'{dataset_name} train set statistics:')

    if image_stats:
        total_mean = psum / length
        total_var = (psum_sq / length) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)

        print(f'- mean: {total_mean.cpu().numpy()}')
        print(f'- std: {total_std.cpu().numpy()}')

        if out_dir is not None:
            with open(os.path.join(out_dir, f'{dataset_name}_image_stats.json'), 'w') as of:
                json.dump({"mean": total_mean.cpu().numpy().tolist(), "std": total_std.cpu().numpy().tolist()}, of, indent=2)

    if label_stats:
        ccount_sum = ccount.sum()
        norm_ccount = ccount / ccount_sum

        lens = torch.tensor([len(cpopul[i]) for i in range(19)])
        norm_lens = lens / lens.sum()

        for i, p in enumerate(norm_ccount):
            print('Class {} {} P {:.3f} L {:.3f}'.format(i, CITY_VALID_CLASSES[i], p, norm_lens[i]))

        ignore_percentage = 1 - ccount_sum / length
        print(f'Unlabeled P {ignore_percentage.item():.3f}')

        if out_dir is not None:
            with open(os.path.join(out_dir, f'{dataset_name}_class_pixels.json'), 'w') as of:
                json.dump(dict(zip(range(0,19),ccount.cpu().numpy().tolist())), of, indent=2)
            with open(os.path.join(out_dir, f'{dataset_name}_class_popul.json'), 'w') as of:
                json.dump(dict(zip(range(0,19),cpopul)), of, indent=2)
            with open(os.path.join(out_dir, f'{dataset_name}_class_centr.json'), 'w') as of:
                json.dump(dict(zip(range(0,19),ccentr)), of, indent=2)
    


if __name__ == "__main__":
    SEED = 0
    BATCH_SIZE = 32

    fix_seed(SEED)

    city_stats = False
    gta_stats = True
    
    if city_stats:
        from datasets import CityscapesDataset
        ds = CityscapesDataset(root="/home/luca/data/cityscapes", split="train", ignore_index=255, resize=(1024,512), stats=True)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
        if not gta_stats:
            get_stats(dl, batch_size=BATCH_SIZE, img_size=(1024,512), image_stats=True, label_stats=True, dataset_name="Cityscapes", out_dir="/home/luca/data/cityscapes")
        else:
            get_stats(dl, batch_size=BATCH_SIZE, img_size=(1024,512), image_stats=True, label_stats=False, dataset_name="Cityscapes", out_dir="/home/luca/data/cityscapes")

    if gta_stats:
        from datasets import GTA5Dataset
        ds = GTA5Dataset(root="/home/luca/data/gta", ignore_index=255, resize=(1280,720), stats=True)
        dl = DataLoader(ds, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=4, collate_fn=collate_fn)
        if not city_stats:
            get_stats(dl, batch_size=BATCH_SIZE//2, img_size=(1280,720), image_stats=True, label_stats=True, dataset_name="GTA5", out_dir="/home/luca/data/gta")
        else:
            get_stats(dl, batch_size=BATCH_SIZE//2, img_size=(1280,720), image_stats=False, label_stats=True, dataset_name="GTA5", out_dir="/home/luca/data/gta")