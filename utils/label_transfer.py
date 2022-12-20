from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
from copy import copy, deepcopy
import cc3d
import argparse
import os
import h5py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

torch.multiprocessing.set_sharing_strategy('file_system')

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor

from utils.utils import get_key

ORGAN_DATASET_DIR = '/home/jliu288/data/whole_organ/'
ORGAN_LIST = 'dataset/dataset_list/PAOT.txt'
NUM_WORKER = 8
NUM_CLASS = 32
TRANSFER_LIST = ['10_08']
## full list
# TRANSFER_LIST = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10_03', '10_06', '10_07', '10_08', '10_09', '10_10', '12', '13', '14']

TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '02': [1,0,3,4,5,6,7,0,0,0,11,0,0,14],
    '03': [6],
    '04': [6,27],       # post process
    '05': [2,26,32],       # post process
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,20,21,23,24],
    '08': [6, 2, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,2],  
    '13': [6,2,1,11,8,9,7,4,5,12,13,25], 
    '14': [11,11,28,28,28],     # Felix data, post process
    '10_03': [6, 27],   # post process
    '10_06': [30],
    '10_07': [11, 28],  # post process
    '10_08': [15, 29],  # post process
    '10_09': [1],
    '10_10': [31]
}

POST_TUMOR_DICT = {
    '04': [(2,27)],
    '05': [(2,26), (3,32)],
    '10_03': [(2,27)], 
    '10_07': [(2,28)]
}

def rl_split(input_data, organ_index, right_index, left_index, name):
    '''
    input_data: 3-d tensor [w,h,d], after transform 'Orientationd(keys=["label"], axcodes="RAS")'
    oragn_index: the organ index of interest
    right_index and left_index: the corresponding index in template
    return [1, w, h, d]
    '''
    RIGHT_ORGAN = right_index
    LEFT_ORGAN = left_index
    label_raw = input_data.copy()
    label_in = np.zeros(label_raw.shape)
    label_in[label_raw == organ_index] = 1
    
    label_out = cc3d.connected_components(label_in, connectivity=26)
    # print('label_out', organ_index, np.unique(label_out), np.unique(label_in), label_out.shape, np.sum(label_raw == organ_index))
    # assert len(np.unique(label_out)) == 3, f'more than 2 component in this ct for {name} with {np.unique(label_out)} component'
    if len(np.unique(label_out)) > 3:
        count_sum = 0
        values, counts = np.unique(label_out, return_counts=True)
        num_list_sorted = sorted(values, key=lambda x: counts[x])[::-1]
        for i in num_list_sorted[3:]:
            label_out[label_out==i] = 0
            count_sum += counts[i]
        label_new = np.zeros(label_out.shape)
        for tgt, src in enumerate(num_list_sorted[:3]):
            label_new[label_out==src] = tgt
        label_out = label_new
        print(f'In {name}. Delete {len(num_list_sorted[3:])} small regions with {count_sum} voxels')
    a1,b1,c1 = np.where(label_out==1)
    a2,b2,c2 = np.where(label_out==2)
    
    label_new = np.zeros(label_out.shape)
    if np.mean(a1) < np.mean(a2):
        label_new[label_out==1] = LEFT_ORGAN
        label_new[label_out==2] = RIGHT_ORGAN
    else:
        label_new[label_out==1] = RIGHT_ORGAN
        label_new[label_out==2] = LEFT_ORGAN
    
    return label_new[None]

class ToTemplatelabel(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, totemplate: List, tumor=False, tumor_list=None) -> NdarrayOrTensor:
        new_lbl = np.zeros(lbl.shape)
        for src, tgt in enumerate(totemplate):
            new_lbl[lbl == (src+1)] = tgt
        # unique,count=np.unique(new_lbl,return_counts=True)
        # data_count=dict(zip(unique,count))
        # print(data_count)
        # unique,count=np.unique(lbl,return_counts=True)
        # data_count=dict(zip(unique,count))
        # print(data_count)
        if tumor:
            for src, item in tumor_list:
                new_lbl[new_lbl == item] = totemplate[0]
        return new_lbl

class ToTemplatelabeld(MapTransform):
    '''
    Comment: spleen to 1
    '''
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.totemplate = ToTemplatelabel()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_index = int(d['name'][0:2])
        TUMOR = False
        tumor_list = None
        if dataset_index == 1 or dataset_index == 2:
            template_key = d['name'][0:2]
            pass
        elif dataset_index == 10:
            template_key = d['name'][0:2] + '_' + d['name'][17:19]
        else:
            template_key = d['name'][0:2]
        if template_key in ['04', '05', '10_03', '10_07', '14']:
            TUMOR = True
            tumor_list = POST_TUMOR_DICT[template_key]
        d['label'] = self.totemplate(d['label'], TEMPLATE[template_key], tumor=TUMOR, tumor_list=tumor_list)
        return d

class RL_Split(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, organ_list: List, name) -> NdarrayOrTensor:
        lbl_new = lbl.copy()
        for organ in organ_list:
            organ_index = organ
            right_index = organ
            left_index = organ + 1
            lbl_post = rl_split(lbl_new[0], organ_index, right_index, left_index, name)
            lbl_new[lbl_post == left_index] = left_index
        return lbl_new

class RL_Splitd(MapTransform):
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spliter = RL_Split()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_index = int(d['name'][0:2])
        # print(d['name'], dataset_index)
        if dataset_index in [5,8,13]:
            # print(d['name'], np.unique(d['label']))
            d['label'] = self.spliter(d['label'], [2], d['name'])
            # print(d['name'], np.unique(d['label']))
        elif dataset_index == 7:
            d['label'] = self.spliter(d['label'], [12], d['name'])
        elif dataset_index == 12:
            d['label'] = self.spliter(d['label'], [2, 16], d['name'])
        else:
            pass
        return d

def generate_label(input_lbl, num_classes, name, TEMPLATE, raw_lbl):
    """
    Convert class index tensor to one hot encoding tensor with -1 (ignored).
    Args:
         input: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    Comment: spleen to 0
    """
    shape = np.array(input_lbl.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    input_lbl = input_lbl.long()

    ## generate binary cross entropy label and assign -1 to ignored organ
    B = result.shape[0]
    for b in range(B):
        dataset_index = int(name[b][0:2])
        if dataset_index == 10:
            template_key = name[b][0:2] + '_' + name[b][17:19]
        else:
            template_key = name[b][0:2]
        
        # for organ split case
        if dataset_index == 5:
            organ_list = [2,3,26,32]
        elif dataset_index == 7:
            organ_list = [6,1,3,2,7,4,5,11,14,18,19,12,13,20,21,23,24]
        elif dataset_index == 8:
            organ_list = [6, 2, 3, 1, 11]
        elif dataset_index == 12:
            organ_list = [6,21,16,17,2,3]
        elif dataset_index == 13:
            organ_list = [6,2,3,1,11,8,9,7,4,5,12,13,25]
        else:
            organ_list = TEMPLATE[template_key]
        
        # -1 for organ not labeled
        for i in range(num_classes):
            if (i+1) not in organ_list:
                result[b, i] = -1
            else:
                result[b, i] = (input_lbl[b][0] ==  (i+1))
        
        # for tumor case
        if template_key in ['04', '05', '10_03', '10_07']:
            tumor_list = POST_TUMOR_DICT[template_key]
            for src, item in tumor_list:
                result[b, item - 1] = (raw_lbl[b][0] == src)

        if template_key in ['14']:
            tumor_lbl = torch.zeros(raw_lbl.shape)
            tumor_lbl[raw_lbl == 3] = 1
            tumor_lbl[raw_lbl == 4] = 1
            tumor_lbl[raw_lbl == 5] = 1
            result[b, organ_list[-1] - 1] = tumor_lbl[b][0]
    return result

label_process = Compose(
    [
        LoadImaged(keys=["image", "label", "label_raw"]),
        AddChanneld(keys=["image", "label", "label_raw"]),
        Orientationd(keys=["image", "label", "label_raw"], axcodes="RAS"),
        ToTemplatelabeld(keys=['label']),
        RL_Splitd(keys=['label']),
        Spacingd(
            keys=["image", "label", "label_raw"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=("bilinear", "nearest", "nearest"),), # process h5 to here
    ]
)

train_img = []
train_lbl = []
train_name = []

for line in open(ORGAN_LIST):
    key = get_key(line.strip().split()[0])
    if key in TRANSFER_LIST:
        train_img.append(ORGAN_DATASET_DIR + line.strip().split()[0])
        train_lbl.append(ORGAN_DATASET_DIR + line.strip().split()[1])
        train_name.append(line.strip().split()[1].split('.')[0])
data_dicts_train = [{'image': image, 'label': label, 'label_raw': label, 'name': name}
            for image, label, name in zip(train_img, train_lbl, train_name)]
print('train len {}'.format(len(data_dicts_train)))

train_dataset = Dataset(data=data_dicts_train, transform=label_process)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKER, 
                            collate_fn=list_data_collate)

for index, batch in enumerate(train_loader):
    x, y, y_raw, name = batch["image"], batch["label"], batch['label_raw'], batch['name']
    y = generate_label(y, NUM_CLASS, name, TEMPLATE, y_raw)
    name = batch['name'][0].replace('label', 'post_label')
    print(name)
    post_dir = ORGAN_DATASET_DIR + '/'.join(name.split('/')[:-1])
    store_y = y.numpy().astype(np.uint8)
    if not os.path.exists(post_dir):
        os.makedirs(post_dir)
    with h5py.File(ORGAN_DATASET_DIR + name + '.h5', 'w') as f:
        f.create_dataset('post_label', data=store_y, compression='gzip', compression_opts=9)
        f.close()
