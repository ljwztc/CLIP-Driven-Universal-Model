# CLIP-Driven Universal Model
Universal Model is the first framework for both organ segmentation and tumor detection. We take the top spot of [MSD competition leaderboard](https://decathlon-10.grand-challenge.org/evaluation/challenge/leaderboard/).

<img src="teaser_fig.png" width = "480" height = "345" alt="" align=center />

## Paper
This repository provides the official implementation of top 1 solution in Medical Segmentation Decathlon

<b>CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection</b> <br/>
[Jie Liu](https://ljwztc.github.io)<sup>1</sup>, [Yixiao Zhang](https://scholar.google.com/citations?hl=en&user=lU3wroMAAAAJ)<sup>2</sup>, [Jie-Neng Chen](https://scholar.google.com/citations?hl=en&user=yLYj88sAAAAJ)<sup>2</sup>,  [Junfei Xiao](https://lambert-x.github.io)<sup>2</sup>, [Yongyi Lu](https://scholar.google.com/citations?hl=en&user=rIJ99V4AAAAJ)<sup>2</sup>, <br/>
[Yixuan Yuan](https://scholar.google.com.au/citations?user=Aho5Jv8AAAAJ&hl=en)<sup>1</sup>, [Alan Yuille](https://scholar.google.com/citations?user=FJ-huxgAAAAJ&hl=en)<sup>2</sup>, [Yucheng Tang](https://tangy5.github.io)<sup>3</sup>, [Zongwei Zhou](https://www.zongweiz.com)<sup>2</sup> <br/>
<sup>1 </sup>City University of Hong Kong,   <sup>2 </sup>Johns Hopkins University,   <sup>3 </sup>NVIDIA <br/>
[paper](https://arxiv.org/pdf/2301.00785.pdf) | [code](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | [slides](https://www.zongweiz.com/_files/ugd/deaea1_eb803117f2ee406fb83a253dd90cab8c.pdf) | poster | [talk](https://www.youtube.com/watch?v=bJpI9tCTsuA) | blog

## ⏳ Dataset Link
- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details)
- 05 [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block)
- 06 [Liver segmentation (3D-IRCADb)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
- 07 [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD)
- 08 [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- 09 [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org)
- 10 [Decathlon (Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 11 [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
- 12 [13 AbdomenCT 12organ](https://github.com/JunMa11/AbdomenCT-1K)

## 💡 Preparation
**Main Requirements**  
> connected-components-3d  
> h5py==3.6.0  
> monai==0.9.0  
> torch==1.11.0  
> tqdm  
> fastremap  

```
pip install -r requirements.txt
pip install 'monai[all]'
cd pretrained_weights/
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
cd ../
```

**Dataset Pre-Process**  
1. Download the dataset according to the dataset link and arrange the dataset according to the `dataset/dataset_list/PAOT.txt`.  
2. Modify the ORGAN_DATASET_DIR value in label_transfer.py (line 51) and NUM_WORKER (line 53)  
3. `python -W ignore label_transfer.py`


## 📦 Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py 
    --dist True 
    --data_root_path /mnt/medical_data/PublicAbdominalData/ 
    --resume out/epoch_10.pth 
    --num_workers 12 
    --num_samples 4 
    --cache_dataset 
    --cache_rate 0.6 
    --uniform_sample
```
## 📦 Validation
```
CUDA_VISIBLE_DEVICES=7 python -W ignore validation.py 
    --data_root_path /mnt/medical_data/PublicAbdominalData/ 
    --start_epoch 10 
    --end_epoch 40 
    --epoch_interval 10 
    --cache_dataset 
    --cache_rate 0.6
```
## 📦 Test
```
CUDA_VISIBLE_DEVICES=7 python -W ignore test.py 
    --resume ./out/epoch_61.pth 
    --data_root_path /mnt/medical_data/PublicAbdominalData/ 
    --store_result 
    --cache_dataset 
    --cache_rate 0.6
```

## 📒 To do
- [x] Code release
- [x] Dataset link
- [ ] Model release
- [ ] Pesudo label release
- [ ] Tutorials for generalizability, transferability, and extensibility

## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 📝 Citation

If you find this repository useful, please consider citing this paper:

