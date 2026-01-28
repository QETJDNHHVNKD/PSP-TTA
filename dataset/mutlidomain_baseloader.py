

import os
import torch
import json
import numpy as np
import yaml
from PIL import Image
from collections import Counter
from dataset.augmenter import AugCompose, RandomHorizontalFlip, RandomVerticalFlip, ImageResize
from dataset.augmenter import RandomRotation, RandomCrop, RandomBrightness, RandomContrast

from torchvision import transforms

class To3Channels(object):
    def __call__(self, x):
        # x: Tensor [C,H,W]
        if x.dim() == 3 and x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x

def transform_type(args, mode):
    size = args.input_size

    if mode == 'Train':
        transforms_op = AugCompose([
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
            RandomRotation(limit=15, prob=0.5),
            RandomCrop(limit=20, prob=0.5),
            RandomBrightness(limit=0.2, prob=0.3),
            RandomContrast(limit=0.2, prob=0.3),
            ImageResize(size),
        ])

        transforms_op.torch_ops = transforms.Compose([
            transforms.ToTensor(),
            To3Channels(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms_op = AugCompose([
            ImageResize(size)
        ])
        transforms_op.torch_ops = transforms.Compose([
            transforms.ToTensor(),
            To3Channels(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms_op

class _base_folder(torch.utils.data.Dataset):
    def __init__(self, sample_list, domian_num, transform):
        self.samples = sample_list
        self.labellist = [x[-2] for x in sample_list]
        self.set_num = domian_num
        self.transform = transform

    def __getitem__(self, index):
        imgpth = self.samples[index][0]
        mskpth = self.samples[index][1]
        setseq = self.samples[index][2]
        setnam = self.samples[index][3]

        MSK = Image.open(mskpth).convert('1')



    def __len__(self):
        return len(self.samples)

def baseloader(args):
    file = open(args.data_configuration, 'r')

    with open(r'....\dataset_config.yaml', 'r', encoding='utf-8') as file:
        _data_configuration = yaml.load(file.read(), Loader=yaml.FullLoader)

    file.close

    args.domian_num = len(_data_configuration)

    _sample_path_list = {'Train':[],'Valid':[],'Test':[]}

    for _dataset_name, _data_info in _data_configuration.items():
        cohort = str(_data_info['Dataset Cohort'])
        cohort = cohort.strip()

        json_path = os.path.join(args.data_path, cohort)


        if not os.path.exists(json_path):
            lower_map = {f.lower(): f for f in os.listdir(args.data_path)}
            key = os.path.basename(cohort).lower()
            if key in lower_map:
                json_path = os.path.join(args.data_path, lower_map[key])

        _single_dataset_list = json.load(open(json_path, encoding='utf8', errors='ignore'))

        for _set_name, _set_path_info in _single_dataset_list.items():           
            for _set_path in _set_path_info:
                _sample_path_list[_set_name].append([os.path.join(args.data_path,_data_info['Dataset Path'], _set_path['USImage']),
                                                     os.path.join(args.data_path,_data_info['Dataset Path'], _set_path['Mask']),
                                                     _data_info['Dataset Sequence'],
                                                     _dataset_name])

    args.sample_path_list = _sample_path_list
    _tn_folder = _base_folder(sample_list=args.sample_path_list['Train'], domian_num=args.domian_num,
                              transform=transform_type(args, 'Train'))
    _vd_folder = _base_folder(sample_list=args.sample_path_list['Valid'], domian_num=args.domian_num,
                              transform=transform_type(args, 'Valid'))
    _tt_folder = _base_folder(sample_list=args.sample_path_list['Test'], domian_num=args.domian_num,
                              transform=transform_type(args, 'Test'))


    labeldict = dict(Counter(_tn_folder.labellist))
    labellist = torch.tensor([v for k, v in labeldict.items()])
    weight = torch.max(labellist) / labellist.float()
    samples_weight = np.array([weight[t] for t in _tn_folder.labellist])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    args.sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    _tn_loader = torch.utils.data.DataLoader(_tn_folder, batch_size=args.batch_size, sampler=None,
                                             num_workers=args.num_workers, pin_memory=True, shuffle=True,
                                             drop_last=True)
    _vd_loader = torch.utils.data.DataLoader(_vd_folder, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                                             drop_last=False)
    _tt_loader = torch.utils.data.DataLoader(_tt_folder, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                                             drop_last=False)


    return _tn_loader, _vd_loader, _tt_loader





