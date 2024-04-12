import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader_without_gt
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ, save_results

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, val_transforms, args):
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    model.eval()
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].cuda(), batch["name"]
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        # use organ_list to indicate the saved organ
        organ_list = [i for i in range(1,33)]
        # organ_list = [26, 32]
        # if 'liver' in name[0]:
        #     organ_list = [6, 27]
        # elif 'kidney' in name[0]:
        #     organ_list = [2, 3, 26]
        # elif 'hepaticvessel' in name[0]:
        #     organ_list = [15, 29]
        # elif 'pancreas' in name[0]:
        #     organ_list = [11, 28]
        # elif 'colon' in name[0]:
        #     organ_list = [31]
        # elif 'lung' in name[0]:
        #     organ_list = [30]
        # elif 'spleen' in name[0]:
        #     organ_list = [1]

        pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
        pred_hard_post = torch.tensor(pred_hard_post)
        batch['results'] = pred_hard_post

        save_results(batch, args.result_save_path, val_transforms, organ_list)
            
        torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='Nvidia', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default='./pretrained_weights/swinunetr.pth', help='The path resume from checkpoint')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet]')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--data_root_path', default=None, help='data root path')
    parser.add_argument('--result_save_path', default=None, help='path for save result')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type= float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    # args.epoch = checkpoint['epoch']
    num_count = 0
    for key, value in load_dict.items():
        if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
            name = '.'.join(key.split('.')[1:])
            name = 'backbone.' + name
        else:
            name = '.'.join(key.split('.')[1:])
        store_dict[name] = value
        num_count += 1

    model.load_state_dict(store_dict)
    print('Use pretrained weights. load', num_count, 'params into', len(store_dict.keys()))

    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_loader_without_gt(args)

    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
